"""
gradio_app.py
Customer Support Chatbot — Gradio Inference Application.
Reads all config from gradio_config.json at startup.

Launch:
    python gradio_app.py --config gradio_config.json --port 7860
    python gradio_app.py --config gradio_config.json --share
"""

import argparse
import gc
import os
import sys
import subprocess


def _ensure_torch():
    try:
        import torch 
    except ImportError:
        print("torch not found — installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "-q"])
_ensure_torch()

def _ensure_gradio():
    try:
        import gradio as gr
    except ImportError:
        print("gradio not found — installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio", "-q"])
_ensure_gradio()


from model_loader   import MODEL_CONFIGS, load_inference_model
from inference_engine import generate_response, generate_response_streaming

from gradio_utils   import (
    load_gradio_config,
    load_sample_queries,
    load_run_metrics,
    list_available_runs,
    make_run_key,
)


# ─────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Customer Support Chatbot — Gradio App")
    p.add_argument("--config", default="gradio_config.json")
    p.add_argument("--port",   type=int, default=7860)
    p.add_argument("--share",  action="store_true")
    p.add_argument("--debug",  action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Global app state
# ─────────────────────────────────────────────────────────────

class AppState:
    """Holds loaded model, tokenizer, and active run identity."""

    def __init__(self):
        self.model        = None
        self.tokenizer    = None
        self.device       = None
        self.model_id     = None
        self.dataset_key  = None
        self.config       = {}
        self.available_runs = []   # list of (model_id, dataset_key, label)

    def is_loaded(self, model_id, dataset_key):
        return (
            self.model is not None and
            self.model_id    == model_id and
            self.dataset_key == dataset_key
        )

    def cleanup(self):
        """Release current model from memory."""
        if self.model is not None:
            del self.model, self.tokenizer
            self.model       = None
            self.tokenizer   = None
            self.model_id    = None
            self.dataset_key = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load(self, model_id, dataset_key, device, debug=False):
        """Load model for the given run. Cleans up previous if different."""
        if self.is_loaded(model_id, dataset_key):
            return  # already loaded — reuse

        self.cleanup()

        run_entry       = self._get_run_entry(model_id, dataset_key)
        best_model_path = run_entry["best_model_path"]

        model, tokenizer, actual_device = load_inference_model(
            model_name      = self.config["models"][model_id]["model_name"],
            best_model_path = best_model_path,
            device          = device,
            load_adapter    = True,
            debug           = debug,
        )
        self.model       = model
        self.tokenizer   = tokenizer
        self.device      = actual_device
        self.model_id    = model_id
        self.dataset_key = dataset_key

    def load_base(self, model_id, dataset_key, device, debug=False):
        """Load base model (no adapter) for compare mode. Returns model, tokenizer, device."""
        run_entry       = self._get_run_entry(model_id, dataset_key)
        best_model_path = run_entry["best_model_path"]
        model, tokenizer, actual_device = load_inference_model(
            model_name      = self.config["models"][model_id]["model_name"],
            best_model_path = best_model_path,
            device          = device,
            load_adapter    = False,
            debug           = debug,
        )
        return model, tokenizer, actual_device

    def load_ft(self, model_id, dataset_key, device, debug=False):
        """Load finetuned model (with adapter) for compare mode. Returns model, tokenizer, device."""
        run_entry       = self._get_run_entry(model_id, dataset_key)
        best_model_path = run_entry["best_model_path"]
        model, tokenizer, actual_device = load_inference_model(
            model_name      = self.config["models"][model_id]["model_name"],
            best_model_path = best_model_path,
            device          = device,
            load_adapter    = True,
            debug           = debug,
        )
        return model, tokenizer, actual_device

    def _get_run_entry(self, model_id, dataset_key):
        return self.config["models"][model_id]["runs"][dataset_key]

    def get_model_config(self, model_id):
        """Return MODEL_CONFIGS entry for model_id."""
        return MODEL_CONFIGS[model_id]

    def get_sample_queries(self, model_id, dataset_key):
        run_entry = self._get_run_entry(model_id, dataset_key)
        return load_sample_queries(run_entry["model_dump_dir"])


STATE = AppState()


# ─────────────────────────────────────────────────────────────
# Helper — cleanup temp model after compare
# ─────────────────────────────────────────────────────────────

def _cleanup_temp(model, tokenizer):
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _parse_run_label(label, available_runs):
    """Recover (model_id, dataset_key) from dropdown label string."""
    for mod_id, dk, lbl in available_runs:
        if lbl == label:
            return mod_id, dk
    return None, None


# ─────────────────────────────────────────────────────────────
# Generation params — shared across tabs
# ─────────────────────────────────────────────────────────────

DEFAULT_TEMP   = 0.2
DEFAULT_TOP_P  = 0.9
DEFAULT_TOP_K  = 50
DEFAULT_REP    = 1.1
DEFAULT_TOKENS = 256


# ─────────────────────────────────────────────────────────────
# Tab 1 — Chat callbacks
# ─────────────────────────────────────────────────────────────

def chat_load_model(run_label, device):
    """Load selected model into STATE when user changes dropdown."""
    model_id, dataset_key = _parse_run_label(run_label, STATE.available_runs)
    if model_id is None:
        return "⚠️  Unknown run selected."
    try:
        STATE.load(model_id, dataset_key, device)
        return f"✅ Loaded: {run_label}"
    except Exception as e:
        return f"❌ Load failed: {e}"


def chat_respond(
    message, history, run_label,
    use_history, temperature, top_p, top_k, rep_penalty, max_tokens,
    device,
):
    """Streaming chat callback for Tab 1."""
    if not message.strip():
        yield history
        return

    model_id, dataset_key = _parse_run_label(run_label, STATE.available_runs)
    if model_id is None:
        yield history + [[message, "⚠️  No model selected."]]
        return

    # Ensure model is loaded
    STATE.load(model_id, dataset_key, device)
    model_config = STATE.get_model_config(model_id)

    # Build history for multi-turn
    history_pairs = None
    if use_history and history:
        history_pairs = [(h[0], h[1]) for h in history if h[1] is not None]

    partial = ""
    for chunk in generate_response_streaming(
        model               = STATE.model,
        tokenizer           = STATE.tokenizer,
        model_config        = model_config,
        query               = message,
        history             = history_pairs,
        max_new_tokens      = max_tokens,
        temperature         = temperature,
        top_p               = top_p,
        top_k               = top_k,
        repetition_penalty  = rep_penalty,
        device              = STATE.device,
    ):
        partial += chunk
        yield history + [[message, partial]]


def chat_clear(_history):
    return []


# ─────────────────────────────────────────────────────────────
# Tab 2 — Compare: Base vs Finetuned callbacks
# ─────────────────────────────────────────────────────────────

def compare_base_ft(
    query, run_label,
    temperature, top_p, top_k, rep_penalty, max_tokens,
    device,
):
    """Run query through base and finetuned of the same model sequentially."""
    if not query.strip():
        return "Please enter a query.", "", "", ""

    model_id, dataset_key = _parse_run_label(run_label, STATE.available_runs)
    if model_id is None:
        return "⚠️  No model selected.", "", "", ""

    model_config = STATE.get_model_config(model_id)

    gen_kwargs = dict(
        model_config       = model_config,
        query              = query,
        max_new_tokens     = max_tokens,
        temperature        = temperature,
        top_p              = top_p,
        top_k              = top_k,
        repetition_penalty = rep_penalty,
        do_sample          = True,
    )

    # ── Base model ──────────────────────────────────────────
    # Cleanup STATE model first — we need memory for base
    STATE.cleanup()
    base_model, base_tok, base_dev = STATE.load_base(model_id, dataset_key, device)
    base_resp = generate_response(
        model=base_model, tokenizer=base_tok,
        device=base_dev, model_type="base", **gen_kwargs
    )
    base_text = base_resp.response
    base_meta = f"{base_resp.generation_time:.2f}s  |  {base_resp.tokens_generated} tokens"
    _cleanup_temp(base_model, base_tok)

    # ── Finetuned model ─────────────────────────────────────
    ft_model, ft_tok, ft_dev = STATE.load_ft(model_id, dataset_key, device)
    ft_resp = generate_response(
        model=ft_model, tokenizer=ft_tok,
        device=ft_dev, model_type="finetuned", **gen_kwargs
    )
    ft_text = ft_resp.response
    ft_meta = f"{ft_resp.generation_time:.2f}s  |  {ft_resp.tokens_generated} tokens"
    _cleanup_temp(ft_model, ft_tok)

    return base_text, base_meta, ft_text, ft_meta


# ─────────────────────────────────────────────────────────────
# Tab 3 — Compare: Model vs Model callbacks
# ─────────────────────────────────────────────────────────────

def compare_two_models(
    query, run_label_1, run_label_2,
    temperature, top_p, top_k, rep_penalty, max_tokens,
    device,
):
    """Run query through two different finetuned models sequentially."""
    if not query.strip():
        return "Please enter a query.", "", "", ""

    mod_id1, dk1 = _parse_run_label(run_label_1, STATE.available_runs)
    mod_id2, dk2 = _parse_run_label(run_label_2, STATE.available_runs)

    if mod_id1 is None or mod_id2 is None:
        return "⚠️  Select both models.", "", "", ""

    results = []
    for model_id, dataset_key in [(mod_id1, dk1), (mod_id2, dk2)]:
        model_config = STATE.get_model_config(model_id)
        STATE.cleanup()
        ft_model, ft_tok, ft_dev = STATE.load_ft(model_id, dataset_key, device)
        resp = generate_response(
            model              = ft_model,
            tokenizer          = ft_tok,
            model_config       = model_config,
            query              = query,
            max_new_tokens     = max_tokens,
            temperature        = temperature,
            top_p              = top_p,
            top_k              = top_k,
            repetition_penalty = rep_penalty,
            do_sample          = True,
            model_type         = "finetuned",
            device             = ft_dev,
        )
        results.append((resp.response, f"{resp.generation_time:.2f}s  |  {resp.tokens_generated} tokens"))
        _cleanup_temp(ft_model, ft_tok)

    return results[0][0], results[0][1], results[1][0], results[1][1]


# ─────────────────────────────────────────────────────────────
# Tab 4 — About content builder
# ─────────────────────────────────────────────────────────────

def build_about_md(config):
    """Build markdown string for the About tab from config and metrics files."""
    lines = []

    # Author
    author = config.get("author", {})
    if author:
        lines.append("## Author")
        lines.append(f"**Name:** {author.get('name', 'N/A')}  ")
        lines.append(f"**Email:** {author.get('email', 'N/A')}  ")
        lines.append(f"**Institute:** {author.get('institute', 'N/A')}  ")
        lines.append(f"**Course:** {author.get('course', 'N/A')}  ")
        lines.append("")

    # Models
    lines.append("## Available Models")
    for model_id, model_entry in config.get("models", {}).items():
        lines.append(f"### {model_id}")
        lines.append(f"- **HuggingFace ID:** `{model_entry.get('model_name', 'N/A')}`")
        lines.append(f"- **Chat style:** `{model_entry.get('chat_style', 'N/A')}`")
        for dataset_key, run_entry in model_entry.get("runs", {}).items():
            dataset_display = config.get("datasets", {}).get(
                dataset_key, {}
            ).get("display_name", dataset_key)
            lines.append(f"- **Dataset:** {dataset_display} (`{dataset_key}`)")
            # Load metrics if available
            metrics = load_run_metrics(run_entry.get("model_dump_dir", ""), dataset_key)
            if metrics:
                # metrics.json may be a list of dicts (one per model_type)
                if isinstance(metrics, list):
                    for m in metrics:
                        mt = m.get("model_type", "?")
                        lines.append(
                            f"  - [{mt}] ROUGE-L: {m.get('rouge_l','N/A')}  "
                            f"BLEU: {m.get('bleu','N/A')}  "
                            f"Cosine: {m.get('mean_cos_sim','N/A')}"
                        )
                else:
                    mt = metrics.get("model_type", "?")
                    lines.append(
                        f"  - [{mt}] ROUGE-L: {metrics.get('rouge_l','N/A')}  "
                        f"BLEU: {metrics.get('bleu','N/A')}  "
                        f"Cosine: {metrics.get('mean_cos_sim','N/A')}"
                    )
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# UI builder
# ─────────────────────────────────────────────────────────────

def build_ui(config, device, debug=False):

    run_labels   = [lbl for _, _, lbl in STATE.available_runs]
    multi_run    = len(run_labels) >= 2
    first_label  = run_labels[0] if run_labels else ""

    # Pre-load first model into STATE
    if run_labels:
        mod_id, dk, _ = STATE.available_runs[0]
        try:
            STATE.load(mod_id, dk, device, debug=debug)
        except Exception as e:
            print(f"⚠️  Could not pre-load model: {e}")

    # Sample queries for first run
    first_samples = []
    if STATE.available_runs:
        mod_id, dk, _ = STATE.available_runs[0]
        first_samples = STATE.get_sample_queries(mod_id, dk)

    about_md = build_about_md(config)

    with gr.Blocks(title="Customer Support Chatbot", theme=gr.themes.Soft()) as demo:

        # ── App header ────────────────────────────────────────
        gr.Markdown("# Customer Support Chatbot")
        gr.Markdown(
            "Fine-tuned with LoRA · "
            f"Model: **{STATE.model_id or 'None'}** · "
            f"Device: **{str(device)}**"
        )

        # ── Shared generation params (always visible) ─────────
        with gr.Accordion("Generation Parameters", open=False):
            with gr.Row():
                s_temp   = gr.Slider(0.0, 1.5,  value=DEFAULT_TEMP,   step=0.05, label="Temperature")
                s_top_p  = gr.Slider(0.5, 1.0,  value=DEFAULT_TOP_P,  step=0.05, label="Top-p")
                s_top_k  = gr.Slider(1,   200,  value=DEFAULT_TOP_K,  step=1,    label="Top-k")
                s_rep    = gr.Slider(1.0, 2.0,  value=DEFAULT_REP,    step=0.05, label="Repetition penalty")
                s_tokens = gr.Slider(32,  512,  value=DEFAULT_TOKENS, step=16,   label="Max new tokens")

        shared_params = [s_temp, s_top_p, s_top_k, s_rep, s_tokens]

        # ── Tabs ──────────────────────────────────────────────
        with gr.Tabs():

            # ── Tab 1: Chat ───────────────────────────────────
            with gr.TabItem("💬 Chat"):
                with gr.Row():
                    dd_chat   = gr.Dropdown(
                        choices=run_labels, value=first_label,
                        label="Model / Dataset", scale=3
                    )
                    chk_hist  = gr.Checkbox(value=False, label="Remember conversation history", scale=1)

                chatbot = gr.Chatbot(label="Conversation", height=420, bubble_full_width=False)

                with gr.Row():
                    chat_in   = gr.Textbox(placeholder="Type your message…", show_label=False, scale=5)
                    send_btn  = gr.Button("Send ➤", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear 🗑", scale=1)

                if first_samples:
                    gr.Examples(
                        examples=[[q] for q in first_samples],
                        inputs=[chat_in],
                        label="Example queries"
                    )

                # Model reload on dropdown change
                status_chat = gr.Textbox(label="Status", interactive=False, visible=True)
                dd_chat.change(
                    fn=lambda lbl: chat_load_model(lbl, device),
                    inputs=[dd_chat],
                    outputs=[status_chat],
                )

                def _chat_submit(msg, hist, lbl, use_hist, *params):
                    yield from chat_respond(msg, hist, lbl, use_hist, *params, device)

                chat_in.submit(
                    _chat_submit,
                    inputs=[chat_in, chatbot, dd_chat, chk_hist] + shared_params,
                    outputs=[chatbot],
                ).then(lambda: "", outputs=[chat_in])

                send_btn.click(
                    _chat_submit,
                    inputs=[chat_in, chatbot, dd_chat, chk_hist] + shared_params,
                    outputs=[chatbot],
                ).then(lambda: "", outputs=[chat_in])

                clear_btn.click(chat_clear, inputs=[chatbot], outputs=[chatbot])

            # ── Tab 2: Compare Base vs Finetuned ─────────────
            with gr.TabItem("🔬 Base vs Fine-tuned"):
                gr.Markdown("Same query through the **base** and **fine-tuned** variants of one model.")
                dd_cmp1  = gr.Dropdown(choices=run_labels, value=first_label, label="Model / Dataset")
                cmp1_in  = gr.Textbox(label="Query", placeholder="Enter a customer query…", lines=2)
                cmp1_btn = gr.Button("Compare ▶", variant="primary")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Base model")
                        cmp1_base_out  = gr.Textbox(label="Response", lines=6, interactive=False)
                        cmp1_base_meta = gr.Textbox(label="Stats",    interactive=False)
                    with gr.Column():
                        gr.Markdown("#### Fine-tuned model")
                        cmp1_ft_out    = gr.Textbox(label="Response", lines=6, interactive=False)
                        cmp1_ft_meta   = gr.Textbox(label="Stats",    interactive=False)

                def _cmp1(*args):
                    return compare_base_ft(*args, device)

                cmp1_btn.click(
                    _cmp1,
                    inputs=[cmp1_in, dd_cmp1] + shared_params,
                    outputs=[cmp1_base_out, cmp1_base_meta, cmp1_ft_out, cmp1_ft_meta],
                )

            # ── Tab 3: Compare Model vs Model ─────────────────
            with gr.TabItem("⚖️ Model vs Model", visible=multi_run):
                gr.Markdown("Same query through **two different fine-tuned models**.")
                second_label = run_labels[1] if len(run_labels) > 1 else first_label
                with gr.Row():
                    dd_m1 = gr.Dropdown(choices=run_labels, value=first_label,  label="Model A")
                    dd_m2 = gr.Dropdown(choices=run_labels, value=second_label, label="Model B")
                cmp2_in  = gr.Textbox(label="Query", placeholder="Enter a customer query…", lines=2)
                cmp2_btn = gr.Button("Compare ▶", variant="primary")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Model A")
                        cmp2_m1_out  = gr.Textbox(label="Response", lines=6, interactive=False)
                        cmp2_m1_meta = gr.Textbox(label="Stats",    interactive=False)
                    with gr.Column():
                        gr.Markdown("#### Model B")
                        cmp2_m2_out  = gr.Textbox(label="Response", lines=6, interactive=False)
                        cmp2_m2_meta = gr.Textbox(label="Stats",    interactive=False)

                def _cmp2(*args):
                    return compare_two_models(*args, device)

                cmp2_btn.click(
                    _cmp2,
                    inputs=[cmp2_in, dd_m1, dd_m2] + shared_params,
                    outputs=[cmp2_m1_out, cmp2_m1_meta, cmp2_m2_out, cmp2_m2_meta],
                )

            # ── Tab 4: About ──────────────────────────────────
            with gr.TabItem("ℹ️ About"):
                gr.Markdown(about_md)

        # ── Footer ────────────────────────────────────────────
        author = config.get("author", {})
        if author:
            gr.Markdown(
                f"---\n"
                f"*{author.get('name','')}  ·  "
                f"{author.get('institute','')}  ·  "
                f"{author.get('course','')}  ·  "
                f"{author.get('email','')}*"
            )

    return demo


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    config = load_gradio_config(args.config)

    if not config:
        print(f"❌ Could not load config from '{args.config}'. Run the training notebook first.")
        sys.exit(1)

    # Populate global STATE
    STATE.config         = config
    STATE.available_runs = list_available_runs(args.config)

    if not STATE.available_runs:
        print("❌ No valid model runs found in config. Check best_model_path entries exist on disk.")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"🤖 Customer Support Chatbot")
    print(f"   Config      : {args.config}")
    print(f"   Runs found  : {len(STATE.available_runs)}")
    for mod_id, dk, lbl in STATE.available_runs:
        print(f"     • {lbl}")
    print(f"{'='*50}\n")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device      : {device}")

    demo = build_ui(config, device, debug=args.debug)
    demo.queue(max_size=3)
    demo.launch(
        server_port = args.port,
        share       = args.share,
        server_name = "0.0.0.0",
    )


if __name__ == "__main__":
    main()
