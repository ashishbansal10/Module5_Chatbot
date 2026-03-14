"""
inference_engine.py
===================
Inference functions for the fine-tuned chatbot.

All prompt formatting is delegated to model_loader.build_prompt()
and model_loader.get_stop_strings() — the same functions used during
training — so there is no train/inference format mismatch.

Public API
----------
    generate_response(model, tokenizer, query, model_config, ...)
        → ChatResponse

    generate_response_streaming(model, tokenizer, query, model_config, ...)
        → Iterator[str]   (token chunks, for Gradio)

    generate_response_batch(model, tokenizer, queries, model_config, ...)
        → List[ChatResponse]

    evaluate_dataset(model, tokenizer, test_dataset, model_config, ...)
        → List[ChatResponse]
        test_dataset : HuggingFace Dataset with 'prompt' and 'response' columns
"""

import time
from threading import Thread
from dataclasses import dataclass, field
from tqdm import tqdm
import torch
from transformers import TextIteratorStreamer

from model_loader import CHAT_STYLE_CONFIG, MODEL_CONFIGS, build_prompt, get_stop_strings


__all__ = [
    "ChatResponse",
    "generate_response",
    "generate_response_streaming",
    "generate_response_batch",
    "evaluate_dataset",
]


# ─────────────────────────────────────────────────────────────
# ChatResponse dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class ChatResponse:
    # Input
    query:            str
    reference:        str | None   # ground-truth answer if available, else None
    # Config labels
    model_type:       str          # "base" or "finetuned"
    chat_style:       str          # e.g. "conversational", "native"
    # Generated output
    response:         str
    tokens_generated: int
    generation_time:  float        # wall-clock seconds
    # Optional metadata
    metadata:         dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# Internal shared helpers
# ─────────────────────────────────────────────────────────────

def _build_gen_kwargs(
    model,
    tokenizer,
    prompt:             str,
    max_new_tokens:     int,
    temperature:        float,
    top_p:              float,
    top_k:              int,
    repetition_penalty: float,
    do_sample:          bool,
    device:             torch.device,
    streamer = None,
) -> tuple[dict, int]:
    """
    Tokenise prompt and build kwargs dict for model.generate().
    Returns (gen_kwargs, prompt_token_length).
    """
    inputs     = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_ids  = inputs["input_ids"].to(device)
    attn_mask  = inputs["attention_mask"].to(device)
    prompt_len = input_ids.shape[-1]

    gen_kwargs = dict(
        input_ids          = input_ids,
        attention_mask     = attn_mask,
        max_new_tokens     = max_new_tokens,
        temperature        = temperature,
        top_p              = top_p,
        top_k              = top_k,
        repetition_penalty = repetition_penalty,
        do_sample          = do_sample,
        pad_token_id       = tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id       = tokenizer.eos_token_id,
    )
    if streamer is not None:
        gen_kwargs["streamer"] = streamer

    return gen_kwargs, prompt_len


def _trim_at_stop_strings(text: str, stop_strings: list[str]) -> str:
    """Cut decoded output at the first stop string boundary."""
    for stop in stop_strings:
        if stop and stop in text:
            text = text.split(stop)[0]
    return text.strip()


# ─────────────────────────────────────────────────────────────
# 1.  generate_response
# ─────────────────────────────────────────────────────────────

def generate_response(
    model,
    tokenizer,
    model_config:       dict,
    query:              str,
    reference:          str | None                   = None,
    history:            list[tuple[str, str]] | None = None,
    max_new_tokens:     int   = 256,
    temperature:        float = 0.2,
    top_p:              float = 0.9,
    top_k:              int   = 50,
    repetition_penalty: float = 1.1,
    do_sample:          bool  = True,
    model_type:         str   = "finetuned",
    device:             torch.device = torch.device("cpu")
) -> ChatResponse:
    """
    Generate a single response (blocking).

    Args:
        model, tokenizer  : HuggingFace model (base or PEFT-wrapped) + tokenizer.
        model_config      : Entry from MODEL_CONFIGS, e.g. MODEL_CONFIGS["distilgpt2"].
        query             : User / customer input string.
        reference         : Ground-truth answer if available — stored in ChatResponse
                            for downstream evaluation, not used during generation.
        history           : Prior turns [(user, assistant), ...] for multi-turn chat.
                            Defaults to None (single-turn, matches training data).
        max_new_tokens    : Maximum tokens to generate.
        temperature       : Sampling temperature.  Lower = more deterministic.
        top_p             : Nucleus sampling probability mass.
        top_k             : Top-k sampling.
        repetition_penalty: Penalise repeated n-grams.
        do_sample         : False → greedy decoding (overrides temperature/top_p/top_k).
        model_type        : Label stored in ChatResponse — "base" or "finetuned".
        device            : torch.device("cpu" | "cuda" | etc.)

    Returns:
        ChatResponse
    """
    # build_prompt with response=None → PromptParts(.prompt, full_text=None)
    parts        = build_prompt(tokenizer, model_config, query, history=history)
    stop_strings = get_stop_strings(tokenizer, model_config)

    gen_kwargs, prompt_len = _build_gen_kwargs(
        model, tokenizer, parts.prompt,
        max_new_tokens, temperature, top_p, top_k, repetition_penalty, do_sample,
        device
    )

    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)
    elapsed = time.perf_counter() - t0

    new_ids       = output_ids[0][prompt_len:]
    response = tokenizer.decode(new_ids, skip_special_tokens=True)
    response = _trim_at_stop_strings(response, stop_strings)

    return ChatResponse(
        query            = query,
        reference        = reference,
        model_type       = model_type,
        chat_style       = model_config["chat_style"],
        response         = response,
        tokens_generated = len(new_ids),
        generation_time  = elapsed,
        metadata         = {
            "temperature":        temperature,
            "top_p":              top_p,
            "top_k":              top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample":          do_sample,
            "prompt_tokens":      prompt_len,
        },
    )


# ─────────────────────────────────────────────────────────────
# 2.  generate_response_streaming
# ─────────────────────────────────────────────────────────────

def generate_response_streaming(
    model,
    tokenizer,
    model_config:       dict,
    query:              str,
    history:            list[tuple[str, str]] | None = None,
    max_new_tokens:     int   = 256,
    temperature:        float = 0.2,
    top_p:              float = 0.9,
    top_k:              int   = 50,
    repetition_penalty: float = 1.1,
    device:             torch.device = torch.device("cpu"),
    verbose:            int   = 0
):
    """
    Token-by-token streaming generator — yields decoded string chunks.
    Intended for Gradio's streaming chat callback.

    Usage:
        for chunk in generate_response_streaming(model, tok, query, cfg):
            print(chunk, end="", flush=True)

    Notes:
        - Runs model.generate() in a background thread via TextIteratorStreamer.
        - Stops early if a stop string is detected in the accumulation buffer.
        - do_sample is always True here (streaming greedy is rarely useful).
    """

    parts        = build_prompt(tokenizer, model_config, query, history=history)
    stop_strings = get_stop_strings(tokenizer, model_config)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    gen_kwargs, _ = _build_gen_kwargs(
        model, tokenizer, parts.prompt,
        max_new_tokens, temperature, top_p, top_k, repetition_penalty, do_sample=True,
        device=device, streamer=streamer
    )

    thread = Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
    thread.start()

    buffer = ""
    for token in streamer:
        buffer += token
        # Check for stop string in accumulation buffer
        hit = next((s for s in stop_strings if s and s in buffer), None)
        if hit:
            # Yield the clean portion before the stop marker, then break
            clean = buffer.split(hit)[0].strip()
            if clean:
                yield clean
            break

        if verbose >= 3:
            print(token, end="", flush=True)

        yield token

    thread.join()


# ─────────────────────────────────────────────────────────────
# 3.  generate_response_batch
# ─────────────────────────────────────────────────────────────

def generate_response_batch(
    model,
    tokenizer,
    model_config:       dict,
    queries:            list[str],
    references:         list[str] | None = None,
    max_new_tokens:     int   = 256,
    temperature:        float = 0.2,
    top_p:              float = 0.9,
    top_k:              int   = 50,
    repetition_penalty: float = 1.1,
    do_sample:          bool  = True,
    model_type:         str   = "finetuned",
    device:             torch.device = torch.device("cpu"),
    verbose:            int   = 1
) -> list[ChatResponse]:
    """
    Run inference sequentially over a list of query strings.

    Args:
        queries    : List of user / customer input strings.
        references : Optional parallel list of ground-truth answers.
                     If provided, each ChatResponse.reference is populated.
                     Must be the same length as queries if given.
        verbose    : 0 -> silent, 1 -> basic progress, 2 -> per-sample progress, 3 -> per-token progress

    Returns:
        List[ChatResponse] in the same order as queries.
    """
    if references is not None:
        assert len(references) == len(queries), (
            f"references length ({len(references)}) must match "
            f"queries length ({len(queries)})"
        )

    results = []
    iter_queries = tqdm(enumerate(queries, 1), total=len(queries), desc="Generating") \
                        if verbose >= 1 else enumerate(queries, 1)

    for i, query in iter_queries:
        ref = references[i - 1] if references is not None else None
        if verbose >= 2:
            print(f"  [{i}/{len(queries)}] {query[:70]}{'…' if len(query) > 70 else ''}")

        resp = generate_response(
            model              = model,
            tokenizer          = tokenizer,
            model_config       = model_config,
            query              = query,
            reference          = ref,
            max_new_tokens     = max_new_tokens,
            temperature        = temperature,
            top_p              = top_p,
            top_k              = top_k,
            repetition_penalty = repetition_penalty,
            do_sample          = do_sample,
            model_type         = model_type,
            device             = device,
        )
        results.append(resp)

        if verbose >= 2:
            print(f"         → {resp.response[:80]}{'…' if len(resp.response) > 80 else ''}")
            print(f"           [{resp.generation_time:.2f}s  |  {resp.tokens_generated} tokens]")

    if verbose == 1:
        total_time = sum(r.generation_time for r in results)
        avg_time   = total_time / len(results)
        avg_tokens = sum(r.tokens_generated for r in results) / len(results)
        print(f"  [batch complete] {len(results)} samples | "
              f"avg time: {avg_time:.2f}s | avg tokens: {avg_tokens:.0f}")

    return results


# ─────────────────────────────────────────────────────────────
# 4.  evaluate_dataset
# ─────────────────────────────────────────────────────────────

def evaluate_dataset(
    model,
    tokenizer,
    model_config:       dict,
    test_dataset,
    max_new_tokens:     int   = 256,
    temperature:        float = 0.2,
    top_p:              float = 0.9,
    top_k:              int   = 50,
    repetition_penalty: float = 1.1,
    do_sample:          bool  = True,
    model_type:         str   = "finetuned",
    device:             torch.device = torch.device("cpu"),
    verbose:            int   = 1
) -> list[ChatResponse]:
    """
    Run inference over a HuggingFace Dataset and return ChatResponse per row.

    The test_dataset is expected to have 'prompt' and 'response' columns
    (the raw pre-tokenised columns from your data pipeline).
    The 'response' column is stored as ChatResponse.reference for
    downstream metric scoring — it is NOT fed into the model.

    Temperature defaults to 0.2 here (more deterministic) which is
    appropriate for evaluation.  Override if needed.

    Args:
        test_dataset : HuggingFace Dataset with columns 'prompt' and 'response'.
        model_config : Entry from MODEL_CONFIGS.
        model_type   : "base" or "finetuned" — label stored in each ChatResponse.
        verbose      : 0 -> silent, 1 -> basic progress, 2 -> per-sample progress, 3 -> per-token progress

    Returns:
        List[ChatResponse]  — same order as test_dataset rows.
        Pass this directly to evaluation.py's score_responses().

    Example (notebook):
        # Full test set — metrics
        base_responses = evaluate_dataset(base_model, tok, test_ds, cfg, model_type="base")
        ft_responses   = evaluate_dataset(ft_model,   tok, test_ds, cfg, model_type="finetuned")
        report = score_responses(base_responses, ft_responses)

        # 10-sample manual sheet
        small_ds = test_ds.select(range(10))
        base_r10 = evaluate_dataset(base_model, tok, small_ds, cfg, model_type="base")
        ft_r10   = evaluate_dataset(ft_model,   tok, small_ds, cfg, model_type="finetuned")
        generate_manual_eval_sheet(base_r10, ft_r10)
    """

    style_cfg  = CHAT_STYLE_CONFIG[model_config["chat_style"]]
    input_col  = style_cfg["input"]    # "prompt"
    output_col = style_cfg["output"]   # "response"

    queries    = test_dataset[input_col]
    references = test_dataset[output_col]

    if verbose >= 1:
        print(f"\n[evaluate_dataset] model_type={model_type}  |  {len(queries)} samples")
        print(f"  chat_style : {model_config['chat_style']}")
        print(f"  temperature: {temperature}  (low = deterministic for eval)")

    return generate_response_batch(
        model              = model, 
        tokenizer          = tokenizer,
        model_config       = model_config,
        queries            = queries,
        references         = references,
        max_new_tokens     = max_new_tokens,
        temperature        = temperature,
        top_p              = top_p,
        top_k              = top_k,
        repetition_penalty = repetition_penalty,
        do_sample          = do_sample,
        model_type         = model_type,
        device             = device,
        verbose            = verbose
    )
