# Standard Libraries
import os
import json
import shutil

# Data and Plotting
import pandas as pd
import matplotlib.pyplot as plt
from packaging import version

# Deep Learning, GenAI & Model Loading
import torch
import transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    GenerationConfig,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig, 
    TaskType, 
    prepare_model_for_kbit_training, 
    PeftModel
)
from trl import SFTConfig, SFTTrainer


# Public API - Cleanly separated
__all__ = [
    "CHAT_STYLE_CONFIG",
    "MODEL_CONFIGS",
    "resolve_model_dtype_and_device",
    "update_model_and_tokenizer",
    "load_model_and_tokenizer",
    "prepare_chatbot_dataset",
    "apply_llm_finetune_strategy",
    "setup_training_engine",
    "run_training_engine",
    "save_training_results",
    "print_training_summary",
    "plot_training_results",
    "load_inference_model"
]


CHAT_STYLE_CONFIG = {
    "simple_qa": {
        "description": "Simple Q&A format for small models/datasets",
        "requires_special_tokens": False,
        "supports_system_prompt": False,
        "use_native_template": False,
        "input":  "prompt",
        "output": "response",
        "template": {
            "prefix": "Question: {input}\nAnswer: ",
            "suffix": "{output}{eos}",
        }
    },
    "conversational": {
        "description": "Natural role-label format, best for chatbot/ecommerce domain",
        "requires_special_tokens": False,
        "supports_system_prompt": False,
        "use_native_template": False,
        "input":  "prompt",
        "output": "response",
        "template": {
            "prefix": "Customer: {input}\nAgent: ",
            "suffix": "{output}{eos}",
        }
    },
    "alpaca": {
        "description": "Alpaca instruction format for mid-size models",
        "requires_special_tokens": False,
        "supports_system_prompt": True,
        "use_native_template": False,
        "input":  "prompt",
        "output": "response",
        "template": {
            "system": "### System:\n{system}\n\n",
            "prefix": "### Instruction:\n{input}\n\n### Response: ",
            "suffix": "{output}{eos}",
        }
    },
    "chatml": {
        "description": "ChatML format for models pretrained with im_start/end tokens",
        "requires_special_tokens": True,
        "supports_system_prompt": True,
        "use_native_template": False,
        "input":  "prompt",
        "output": "response",
        "template": {
            "system": "{bos}system\n{system}{eos}\n",
            "prefix": "{bos}user\n{input}{eos}\n{bos}assistant\n",
            "suffix": "{output}{eos}",
        }
    },
    "llama2": {
        "description": "Manual Llama2 [INST] format — use only if tokenizer has no built-in template",
        "requires_special_tokens": False,
        "supports_system_prompt": True,
        "use_native_template": False,
        "input":  "prompt",
        "output": "response",
        "template": {
            "system": "{bos}[INST] <<SYS>>\n{system}\n<</SYS>>\n\n",
            "prefix": "{bos}[INST] {input} [/INST]\n",
            "suffix": "{output}{eos}",
        }
    },
    "native": {
        "description": "Use tokenizer built-in chat_template via apply_chat_template — preferred for TinyLlama, Mistral, Llama2-chat etc.",
        "requires_special_tokens": False,
        "supports_system_prompt": True,
        "use_native_template": True,
        "input":  "prompt",
        "output": "response",
        "template": None                # tokenizer.apply_chat_template handles it
    }
}

#distilgpt2:                            Small model <= 125M parameters
#TinyLlama/TinyLlama-1.1B-Chat-v1.0:    Large model ~= 1.1B parameters

MODEL_CONFIGS = {
    "distilgpt2": {
        "model_id":       "distilgpt2",
        "use_4bit":       False,
        "use_8bit":       False,
        "target_modules": ["c_attn", "c_proj"],
        # LoRA
        "learning_rate":  2e-4,
        "lora_r":         16,
        "lora_alpha":     32,
        "dropout":        0.05,
        # Tokens
        # "bos_token":    "<|im_start|>",   # not needed for conversational style
        # "eos_token":    "<|im_end|>",     # not needed for conversational style
        # "unk_token":    "<|endoftext|>",  # native DistilGPT2 token, no override needed
        "pad_token":      "<|padding|>",    # dedicated pad, avoids eos==pad leakage
        "additional_special_tokens": [
            "<|padding|>"                   # only new token needed now
        ],
        # Chat
        "chat_style":     "conversational",
        "system_prompt":  None,             # conversational doesn't support system prompt
    },
    "tinyllama": {
        "model_id":       "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "use_4bit":       True,
        "use_8bit":       False,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # LoRA
        "learning_rate":  1e-4,
        "lora_r":         32,
        "lora_alpha":     64,
        "dropout":        0.05,
        "pad_token":      "<|padding|>",      # dedicated pad, avoids eos==pad leakage
        "additional_special_tokens": [
            "<|padding|>"                     # only new token needed
        ],
        # Chat
        "chat_style":     "native",           # use tokenizer.apply_chat_template
        "system_prompt":  "You are a helpful ecommerce customer support agent.",
    }
}


def resolve_model_dtype_and_device(device):
    """
    Shared helper — used by both load_model_and_tokenizer
    and load_inference_model.
    """
    is_cuda = device.type == "cuda" and torch.cuda.is_available()
    if is_cuda:
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map  = "auto"
    else:
        model_dtype = torch.float32
        device_map  = str(device)

    return model_dtype, device_map


def update_model_and_tokenizer(model, model_config, tokenizer, debug=False):
    """
    Safely updates model and tokenizer with custom special tokens.
    Behavior is driven by chat_style in model_config.
    
    Args:
        model:        The loaded HuggingFace model.
        model_config: The model config dict (from MODEL_CONFIGS).
        tokenizer:    The loaded HuggingFace tokenizer.
        debug (bool): If True, prints additional debug information.
    """
    chat_style = model_config.get("chat_style")
    style_cfg  = CHAT_STYLE_CONFIG.get(chat_style, None)

    # ----------------------------------------------------------------
    # Step 1: Early validation — fail fast before any work is done
    # ----------------------------------------------------------------

    model_transformers_ver = transformers.__version__
    assert model_transformers_ver is not None and \
       version.parse(model_transformers_ver) >= version.parse("4.46.0"), \
        f"❌ transformers >= 4.46.0 required for mean_resizing support. " \
        f"Found in model.config: '{model_transformers_ver}'"

    if style_cfg is None:
        raise ValueError(
            f"❌ Missing or Unknown chat_style: '{chat_style}'. "
            f"Must be one of {list(CHAT_STYLE_CONFIG.keys())}."
        )

    if style_cfg["requires_special_tokens"]:
        assert model_config.get("bos_token") is not None, \
            f"❌ chat_style='{chat_style}' requires bos_token in model_config!"
        assert model_config.get("eos_token") is not None, \
            f"❌ chat_style='{chat_style}' requires eos_token in model_config!"

    if style_cfg["use_native_template"]:
        assert tokenizer.chat_template is not None, \
            f"❌ chat_style='native' selected but tokenizer has no built-in chat_template!"

    pad_token = model_config.get("pad_token")
    assert pad_token is not None, \
        f"❌ pad_token missing from model_config — must be set to avoid eos==pad leakage!"

    print(f"\n{'='*50}")
    print(f"🔧 update_model_and_tokenizer")
    print(f"    chat_style      : {chat_style}")
    print(f"    Tokenizer Class : {tokenizer.__class__.__name__}")
    print(f"\n    Model & Tokenizer State BEFORE update")
    print(f"        Chat Template Exists      : {tokenizer.chat_template is not None}")
    print(f"        Vocab Size                : {len(tokenizer)}")
    print(f"        Special Tokens Map        : {tokenizer.special_tokens_map}")
    print(f"        Additional Special Tokens : {(ast := getattr(tokenizer, 'additional_special_tokens', []))} ({len(ast)})")
    print(f"        Embedding Size (input)    : {model.get_input_embeddings().weight.shape[0]}")
    print(f"        Embedding Size (output)   : {model.get_output_embeddings().weight.shape[0]}")

    print(f"{'='*50}")

    # ----------------------------------------------------------------
    # Step 2: Build update_dict based on chat_style
    # ----------------------------------------------------------------
    update_dict = {}

    if style_cfg["use_native_template"]:
        # Native style (e.g. TinyLlama) — tokenizer already has
        # bos/eos/unk configured correctly, only add pad token
        update_dict["pad_token"] = pad_token
        print(f"ℹ️  Native template — only adding pad token, preserving all native tokens.")

    else:
        # Non-native styles — build full update_dict from model_config
        # Priority: model_config value → existing tokenizer value
        critical_keys = ["bos_token", "eos_token", "unk_token", "pad_token"]

        # Dedup while preserving order — set() alone would lose order
        seen           = set()
        supported_keys = []
        for k in critical_keys + list(tokenizer.special_tokens_map.keys()):
            if k not in seen and k != "additional_special_tokens":
                seen.add(k)
                supported_keys.append(k)

        for key in supported_keys:
            # Explicit None check — avoids falsy empty string edge case
            val = model_config.get(key)
            if val is None:
                # Never fall back pad_token to tokenizer default
                # tokenizer default is often eos_token — causes label leakage
                if key == "pad_token":
                    continue
                val = getattr(tokenizer, key, None)
            if val is not None:
                update_dict[key] = val

    # ----------------------------------------------------------------
    # Shared: additional_special_tokens — applied to both branches
    # Dedup against tokens already assigned to bos/eos/unk/pad
    # to avoid double registration and inflated "Added N tokens" count
    # reserved uses set for O(1) lookup only — order preserved in list
    # ----------------------------------------------------------------
    reserved   = {update_dict.get("bos_token"), update_dict.get("eos_token"),
                  update_dict.get("unk_token"),  update_dict.get("pad_token")}
    additional = [
        t for t in model_config.get("additional_special_tokens", [])
        if t not in reserved
    ]
    if additional:
        update_dict["additional_special_tokens"] = additional

    if debug:
        print(f"\n[DEBUG] update_dict before add_special_tokens: {update_dict}")

    # ----------------------------------------------------------------
    # Step 3: Apply all special tokens in ONE call
    # ----------------------------------------------------------------
    num_added = tokenizer.add_special_tokens(update_dict)
    print(f"   Added {num_added} new tokens to tokenizer.")

    # ----------------------------------------------------------------
    # Step 4: Resize embeddings
    # mean_resizing=True (default since transformers 4.46) automatically
    # initializes new embeddings using mean + covariance of existing
    # embeddings. pad_to_multiple_of=8 aligns to GPU Tensor Core blocks.
    # ----------------------------------------------------------------
    old_vocab_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8, mean_resizing = True)
    new_vocab_size = model.get_input_embeddings().weight.shape[0]
    print(f"✅ Embeddings resized: {old_vocab_size} → {new_vocab_size} (+{new_vocab_size - old_vocab_size} slots, mean-init by HF)")

    if model_config.get("use_4bit") or model_config.get("use_8bit"):
        print(f"⚠️  Quantized model detected — verify embedding resize completed correctly above.")

    # ----------------------------------------------------------------
    # Step 5: Sync model config token IDs
    # ----------------------------------------------------------------
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    # ----------------------------------------------------------------
    # Step 6: Verification table
    # ----------------------------------------------------------------
    print(f"\n    Model & Tokenizer State AFTER update")
    print(f"        Chat Template Exists      : {tokenizer.chat_template is not None}")
    print(f"        Vocab Size                : {len(tokenizer)}")
    print(f"        Special Tokens Map        : {tokenizer.special_tokens_map}")
    print(f"        Additional Special Tokens : {(ast := getattr(tokenizer, 'additional_special_tokens', []))} ({len(ast)})")
    print(f"        Embedding Size (input)    : {model.get_input_embeddings().weight.shape[0]}")
    print(f"        Embedding Size (output)   : {model.get_output_embeddings().weight.shape[0]}")

    print(f"\n   {'Key':<25} | {'Model ID':<10} | {'Tok ID':<10} | {'Token String':<25} | Decoded")
    print(f"   {'-'*85}")
    for key in tokenizer.special_tokens_map.keys():
        map_str = tokenizer.special_tokens_map.get(key)

        if key == "additional_special_tokens":
            for i, sub_token in enumerate(map_str):
                t_id = tokenizer.convert_tokens_to_ids(sub_token)
                print(f"   {'additional_' + str(i):<25} | {'N/A':<10} | {t_id:<10} | {sub_token:<25} | {tokenizer.decode([t_id])}")
            continue

        m_id = getattr(model.config, f"{key}_id", None)
        t_id = getattr(tokenizer, f"{key}_id", None)
        decode_id = m_id if isinstance(m_id, int) else t_id
        try:
            decoded = tokenizer.decode([decode_id]) if isinstance(decode_id, int) else "N/A"
        except Exception:
            decoded = "Error"

        print(f"   {key:<25} | {str(m_id) if m_id is not None else 'N/A':<10} | {str(t_id):<10} | {str(map_str):<25} | {decoded}")

    # ----------------------------------------------------------------
    # Step 7: Sanity checks
    # ----------------------------------------------------------------
    assert tokenizer.pad_token_id != tokenizer.eos_token_id, \
        f"❌ pad_token_id ({tokenizer.pad_token_id}) == eos_token_id ({tokenizer.eos_token_id}). " \
        f"Ensure pad_token is a dedicated token in model_config, not eos_token!"
    assert model.config.pad_token_id == tokenizer.pad_token_id, \
        f"❌ model.config.pad_token_id ({model.config.pad_token_id}) out of sync " \
        f"with tokenizer.pad_token_id ({tokenizer.pad_token_id})!"
    assert model.get_input_embeddings().weight.shape[0] >= len(tokenizer), \
        f"❌ Embedding size ({model.get_input_embeddings().weight.shape[0]}) " \
        f"smaller than vocab size ({len(tokenizer)}) — resize failed!"

    print(f"\n✅ All sanity checks passed.")
    print(f"{'='*50}\n")

    return model, tokenizer


def load_model_and_tokenizer(model_id, model_config, device, debug=False):
    """
    Loads model and tokenizer, then updates special tokens
    based on chat_style in model_config.
    
    Args:
        model_id     (str):          HuggingFace model path.
        model_config (dict):         Model config from MODEL_CONFIGS.
        device       (torch.device): Execution device (e.g. 'cuda', 'cpu').
        debug        (bool):         If True, prints model architecture and config.
    
    Returns:
        model, tokenizer
    """

    # ----------------------------------------------------------------
    # Step 1: Early validation
    # ----------------------------------------------------------------
    assert model_id is not None and model_id != "", \
        "❌ model_id is empty or None!"
    assert model_config is not None, \
        "❌ model_config is None!"
    assert device is not None and device != "", \
        "❌ device is empty or None!"
    assert model_config.get("chat_style") in CHAT_STYLE_CONFIG, \
        f"❌ chat_style '{model_config.get('chat_style')}' not found in CHAT_STYLE_CONFIG!"

    print(f"\n{'='*50}")
    print(f"🚀 load_model_and_tokenizer")
    print(f"   model_id   : {model_id}")
    print(f"   device     : {device}")
    print(f"   chat_style : {model_config.get('chat_style')}")
    print(f"   debug      : {debug}")
    print(f"{'='*50}")

    # ----------------------------------------------------------------
    # Step 2: Setup dtype and device_map
    # ----------------------------------------------------------------
    # ✅ Use bfloat16/float16 on CUDA for efficiency
    # ✅ Respect device arg — don't override CPU with auto
    model_dtype, device_map = resolve_model_dtype_and_device(device)

    # ----------------------------------------------------------------
    # Step 3: Load tokenizer
    # ----------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    print(f"✅ Tokenizer loaded.")

    # ----------------------------------------------------------------
    # Step 4: Setup quantization
    # Only applies if CUDA is available AND config requests it
    # ----------------------------------------------------------------
    bnb_config  = None
    quant_label = "None (Full Precision)"

    if is_cuda:
        if model_config.get("use_4bit"):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=model_dtype,
                bnb_4bit_use_double_quant=True
            )
            quant_label = "4-bit (NF4)"
        elif model_config.get("use_8bit"):
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            quant_label = "8-bit"
    else:
        if model_config.get("use_4bit") or model_config.get("use_8bit"):
            print(f"⚠️  Quantization requested but CUDA unavailable — falling back to full precision.")

    # ----------------------------------------------------------------
    # Step 5: Load model
    # ----------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        dtype=model_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    print(f"✅ Model loaded.")

    if debug:
        print(f"\n[DEBUG] Model architecture:")
        print(model)
        print(f"\n[DEBUG] Model config:")
        print(model.config)

    # ----------------------------------------------------------------
    # Step 6: Update model and tokenizer special tokens
    # driven by chat_style in model_config
    # ----------------------------------------------------------------
    model, tokenizer = update_model_and_tokenizer(
        model=model,
        model_config=model_config,
        tokenizer=tokenizer,
        debug=debug
    )

    # ----------------------------------------------------------------
    # Step 7: Summary
    # ----------------------------------------------------------------
    memory_gb = model.get_memory_footprint() / (1024**3)

    print(f"\n{'='*50}")
    print(f"🚀 MODEL INITIALIZED")
    print(f"   📦 model_id              : {model_id}")
    print(f"   📍 Device / device_map   : {device} / {device_map}")
    print(f"   🔢 Precision             : {model_dtype}")
    print(f"   💎 Quant                 : {quant_label}")
    print(f"   💾 VRAM                  : {memory_gb:.2f} GB")
    print(f"   📖 Vocab                 : {len(tokenizer)}")
    print(f"   🎨 ChatStyle             : {model_config.get('chat_style')}")
    print(f"{'='*50}\n")

    return model, tokenizer


def prepare_chatbot_dataset(
    dataset_dict,
    model_config,
    tokenizer,
    max_length       = 1024,
    do_label_masking = True,
    debug            = False
):
    """
    Factory function that formats dataset into the correct chat style.
    Optionally applies label masking — loss computed on response tokens only.

    Args:
        dataset_dict     (DatasetDict): Raw dataset with at least 'train' split.
        model_config     (dict):        Model config from MODEL_CONFIGS.
        tokenizer:                      The loaded and updated HuggingFace tokenizer.
        max_length       (int):         Max token length. Default: 1024.
        do_label_masking (bool):        If True, masks prompt tokens in labels.
                                        If False, loss computed on full text (baseline).
        debug            (bool):        If True, prints additional debug information.

    Returns:
        DatasetDict: Processed dataset.
                     do_label_masking=True  → input_ids, attention_mask, labels
                     do_label_masking=False → text
    """

    # ----------------------------------------------------------------
    # Step 1: Early validation
    # ----------------------------------------------------------------
    assert dataset_dict is not None, \
        "❌ dataset_dict is None!"
    # Only train is mandatory to check, test/val may or may not be present
    assert "train" in dataset_dict, \
        "❌ dataset_dict missing 'train' split!"

    chat_style = model_config.get("chat_style")
    style_cfg  = CHAT_STYLE_CONFIG.get(chat_style, None)

    if style_cfg is None:
        raise ValueError(
            f"❌ Unknown chat_style: '{chat_style}'. "
            f"Must be one of {list(CHAT_STYLE_CONFIG.keys())}."
        )

    # Validate dataset has required columns
    input_col  = style_cfg["input"]
    output_col = style_cfg["output"]
    for split in dataset_dict.keys():
        assert input_col in dataset_dict[split].column_names, \
            f"❌ Input column '{input_col}' not found in '{split}' split!"
        assert output_col in dataset_dict[split].column_names, \
            f"❌ Output column '{output_col}' not found in '{split}' split!"
    
    # Validate native style has chat_template
    if style_cfg["use_native_template"]:
        assert tokenizer.chat_template is not None, \
            f"❌ chat_style='native' but tokenizer has no built-in chat_template!"

    # Validate that after bos/eos would be known:
    if not style_cfg["use_native_template"] and style_cfg["requires_special_tokens"]:
        assert tokenizer.bos_token is not None and tokenizer.bos_token != "", \
            f"❌ chat_style='{chat_style}' requires bos but tokenizer.bos_token is None!"
        assert tokenizer.eos_token is not None and tokenizer.eos_token != "", \
            f"❌ chat_style='{chat_style}' requires eos but tokenizer.eos_token is None!"

    # Warn if system_prompt set but style doesn't support it
    system_prompt = model_config.get("system_prompt")
    if system_prompt and not style_cfg["supports_system_prompt"]:
        print(f"⚠️  system_prompt set in model_config but chat_style='{chat_style}' "
              f"does not support it — ignoring system_prompt.")
        system_prompt = None

    print(f"\n{'='*50}")
    print(f"🛠️  prepare_chatbot_dataset")
    print(f"   chat_style       : {chat_style}")
    print(f"   input_col        : {input_col}")
    print(f"   output_col       : {output_col}")
    print(f"   system_prompt    : {system_prompt}")
    print(f"   max_length       : {max_length}")
    print(f"   do_label_masking : {do_label_masking}")
    print(f"{'='*50}")

    # Before defining _format_template
    template = style_cfg["template"]
    bos = tokenizer.bos_token or ""
    eos = tokenizer.eos_token or ""

    # ----------------------------------------------------------------
    # Step 2: Core tokenize + mask helper
    # Only called when do_label_masking=True
    # ----------------------------------------------------------------
    def _tokenize_and_mask(prompt_text, full_text):
        """
        Tokenizes full_text and masks prompt_text tokens with -100.

        Args:
            prompt_text (str): Prompt portion — masked in labels (no loss).
            full_text   (str): Full text (prompt + response) — model input.

        Returns:
            dict with input_ids, attention_mask, labels
        """
        full_enc = tokenizer(
            full_text,
            max_length     = max_length,
            truncation     = True,
            padding        = False,
            return_tensors = None,
        )

        prompt_enc = tokenizer(
            prompt_text,
            max_length         = max_length,
            truncation         = True,
            padding            = False,
            return_tensors     = None,
            add_special_tokens = False,  # avoid double BOS at boundary
        )

        input_ids      = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        prompt_len     = len(prompt_enc["input_ids"])

        labels              = input_ids.copy()
        labels[:prompt_len] = [-100] * prompt_len  # ✅ mask prompt — no loss here

        # Safety — response fully truncated edge case
        if all(l == -100 for l in labels):
            print(f"⚠️  All tokens masked after truncation — "
                  f"response may be fully truncated. Consider increasing max_length.")

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }

    # ----------------------------------------------------------------
    # Step 3: Native formatter
    # Uses tokenizer.apply_chat_template
    # ----------------------------------------------------------------
    def _format_native(example):
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user",      "content": str(example[input_col]).strip()})
        messages.append({"role": "assistant", "content": str(example[output_col]).strip()})

        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize              = False,
            add_generation_prompt = False
        )
        
        if do_label_masking:
            prompt_messages = messages[:-1]  # system + user only
            prompt_text     = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize              = False,
                add_generation_prompt = True   # ✅ adds assistant turn opener
            )
            return _tokenize_and_mask(prompt_text, full_text)
        else:
            return {"text": full_text}


    # ----------------------------------------------------------------
    # Step 4: Template formatter
    # Uses CHAT_STYLE_CONFIG template prefix/suffix
    # ----------------------------------------------------------------

    def _format_template(example):
        input_text    = str(example[input_col]).strip()
        output_text   = str(example[output_col]).strip()

        full_text = ""

        # Build prompt (everything BEFORE response)
        prompt_text = ""

        # Optional system block
        if system_prompt and template.get("system"):
            prompt_text += template["system"].format(
                system=system_prompt,
                bos=bos,
                eos=eos
            )

        # Prefix (question/instruction part)
        prompt_text += template["prefix"].format(
            input=input_text,
            bos=bos,
            eos=eos
        )

        # Suffix (answer/response part)
        # Full text = prompt + response
        full_text = prompt_text + template["suffix"].format(
            output=output_text,
            bos=bos,
            eos=eos
        )

        if do_label_masking:
            return _tokenize_and_mask(prompt_text, full_text)
        else:
            return {"text": full_text}

    # ----------------------------------------------------------------
    # Step 5: Apply formatter
    # remove_columns = original cols BEFORE map — never overlap with
    # formatter output keys (text / input_ids / attention_mask / labels)
    # so removing all original cols is safe without keep_cols filter
    # ----------------------------------------------------------------
    formatter      = _format_native   if style_cfg["use_native_template"] else _format_template
    formatter_name = "_format_native" if style_cfg["use_native_template"] else "_format_template"
    print(f"   formatter        : {formatter_name}")

    processed_dataset = dataset_dict.map(
        formatter,
        desc           = f"Formatting {'+ masking ' if do_label_masking else ''}[{chat_style}]",
        remove_columns = next(iter(dataset_dict.values())).column_names
    )

    # ----------------------------------------------------------------
    # Step 6: Debug preview — one sample per split
    # ----------------------------------------------------------------
    if debug:
        print(f"\n🚀 DATA PREVIEW")
        print(f"{'='*60}")

        for split in processed_dataset.keys():
            example = processed_dataset[split][0]

            if do_label_masking:
                input_ids        = example["input_ids"]
                labels           = example["labels"]
                full_decoded     = tokenizer.decode(input_ids,    skip_special_tokens=False)
                response_ids     = [t for t, l in zip(input_ids, labels) if l != -100]
                response_decoded = tokenizer.decode(response_ids, skip_special_tokens=False)
                n_masked         = sum(1 for l in labels if l == -100)
                n_unmasked       = sum(1 for l in labels if l != -100)

                print(f"--- {split.upper()} SAMPLE 1 (masking ON) ---")
                print(f"【full text decoded】 :\n{full_decoded}")
                print(f"【response only】     :\n{response_decoded}")
                print(f"【token counts】      : total={len(input_ids)}, masked={n_masked}, unmasked={n_unmasked}")
                print(f"【masking ratio】     : {n_masked/len(input_ids)*100:.1f}% masked")
            else:
                tokens = tokenizer(example["text"], return_tensors="pt")
                print(f"--- {split.upper()} SAMPLE 1 (masking OFF) ---")
                print(f"【text】        :\n{example['text']}")
                print(f"【token count】 : {tokens['input_ids'].shape[1]}")

            print(f"{'='*60}")

        print(f"\n[DEBUG] Column names after cleanup : {processed_dataset['train'].column_names}")

    # ----------------------------------------------------------------
    # Step 7: Summary
    # ----------------------------------------------------------------
    print(f"\n✅ Dataset ready")
    for split in processed_dataset.keys():
        print(f"   📊 {split:<10} : {len(processed_dataset[split])} rows")
    print(f"   columns          : {processed_dataset['train'].column_names}")
    print(f"{'='*50}\n")

    return processed_dataset


def apply_llm_finetune_strategy(model, model_config, strategy_name="lora", hp_overrides=None, debug=False):
    """
    Module 3: Applies fine-tuning strategy to model.
    Currently supported strategies: ['lora']

    Args:
        model         (nn.Module): The loaded HuggingFace model.
        model_config  (dict):      Model config from MODEL_CONFIGS.
        strategy_name (str):       Fine-tuning strategy. Default: 'lora'.
        hp_overrides  (dict):      Optional hyperparameter overrides.
        debug         (bool):      If True, prints additional debug info.

    Returns:
        model:       Model with strategy applied.
        peft_config: PEFT config object for SFTTrainer.
    """

    SUPPORTED_STRATEGIES = ["lora"]

    # ----------------------------------------------------------------
    # Step 1: Early validation
    # ----------------------------------------------------------------
    assert model is not None, \
        "❌ model is None!"
    assert model_config is not None, \
        "❌ model_config is None!"
    assert "target_modules" in model_config, \
        "❌ target_modules missing from model_config!"

    if strategy_name is None or strategy_name.strip() == "":
        raise ValueError(
            f"❌ strategy_name is empty or None. "
            f"Must be one of {SUPPORTED_STRATEGIES}."
        )
    if strategy_name.lower() not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"❌ Unsupported strategy_name: '{strategy_name}'. "
            f"Must be one of {SUPPORTED_STRATEGIES}."
        )

    hp_overrides = hp_overrides or {}

    # ----------------------------------------------------------------
    # Step 2: Inspect model object state
    # ----------------------------------------------------------------
    is_4bit     = getattr(model, "is_loaded_in_4bit", False)
    is_8bit     = getattr(model, "is_loaded_in_8bit", False)
    is_quantized = is_4bit or is_8bit
    model_dtype = next(model.parameters()).dtype

    print(f"\n{'='*50}")
    print(f"🎨 apply_llm_finetune_strategy")
    print(f"   strategy     : {strategy_name.upper()}")
    print(f"   quantized    : {is_quantized}")
    print(f"   dtype        : {model_dtype}")
    print(f"{'='*50}")

    # ----------------------------------------------------------------
    # Step 3: Resolve HP values
    # Priority: hp_overrides → model_config → hardcoded default
    # ----------------------------------------------------------------
    r       = hp_overrides.get("lora_r",     model_config.get("lora_r",     16))
    alpha   = hp_overrides.get("lora_alpha", model_config.get("lora_alpha", 32))
    dropout = hp_overrides.get("dropout",    model_config.get("dropout",    0.05))

    if debug:
        print(f"\n[DEBUG] Resolved HP values:")
        print(f"   lora_r         : {r:<6} {'(overridden)' if 'lora_r'     in hp_overrides else '(from config)'}")
        print(f"   lora_alpha     : {alpha:<6} {'(overridden)' if 'lora_alpha' in hp_overrides else '(from config)'}")
        print(f"   dropout        : {dropout:<6} {'(overridden)' if 'dropout'    in hp_overrides else '(from config)'}")
        print(f"   target_modules : {model_config['target_modules']}")

    # ----------------------------------------------------------------
    # Step 4: Apply LoRA strategy
    # ----------------------------------------------------------------
    if strategy_name.lower() == "lora":

        # Quantized model needs kbit preparation before LoRA
        if is_quantized:
            print(f"   💎 Quantized model detected — applying k-bit training preparation...")
            model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r              = r,
            lora_alpha     = alpha,
            target_modules = model_config["target_modules"],
            lora_dropout   = dropout,
            bias           = "none",
            task_type      = TaskType.CAUSAL_LM
        )

        # Disable cache — incompatible with gradient checkpointing during training
        model.config.use_cache = False

    # ----------------------------------------------------------------
    # Step 5: Summary
    # ----------------------------------------------------------------
    print(f"\n   {'='*45}")
    print(f"   ✅ Strategy Summary")
    print(f"   {'='*45}")
    print(f"   strategy         : {strategy_name.upper()}")
    print(f"   quantized        : {is_quantized}")
    print(f"   dtype            : {model_dtype}")
    print(f"   lora_r           : {r}")
    print(f"   lora_alpha       : {alpha}")
    print(f"   dropout          : {dropout}")
    print(f"   target_modules   : {model_config['target_modules']}")
    print(f"   use_cache        : {model.config.use_cache}")
    print(f"   {'='*45}\n")

    return model, peft_config


def setup_training_engine(
    model,
    tokenizer,
    dataset,
    model_config,
    peft_config,
    device,
    hp_overrides     = None,
    seed             = None,
    output_dir       = "./results",
    do_label_masking = True,
    debug            = False
):
    """
    Module 4: Configures and initializes the SFTTrainer training engine.
    Inspects model object to determine precision (bf16/fp16/fp32).

    Args:
        model            (nn.Module):    The loaded and LoRA-wrapped model.
        tokenizer:                       The loaded and updated tokenizer.
        dataset          (DatasetDict):  Processed dataset with at least 'train' split.
        model_config     (dict):         Model config from MODEL_CONFIGS.
        peft_config      (LoraConfig):   PEFT config from apply_llm_finetune_strategy.
        device           (torch.device): Execution device (e.g. 'cuda', 'cpu').
        hp_overrides     (dict):         Optional hyperparameter overrides.
                                         Friendly keys: batch_size etc.
        seed             (int):          seed for SFTConfig
        output_dir       (str):          Directory for checkpoints and results.
        do_label_masking (bool):         if true, use label masking technique to mask prompt in training phase
        debug            (bool):         If True, prints full SFTConfig via to_json_string().

    Returns:
        trainer (SFTTrainer): Configured trainer ready to run.
    """

    # ----------------------------------------------------------------
    # Step 1: Early validation
    # ----------------------------------------------------------------
    assert model        is not None, "❌ model is None!"
    assert tokenizer    is not None, "❌ tokenizer is None!"
    assert dataset      is not None, "❌ dataset is None!"
    assert model_config is not None, "❌ model_config is None!"
    assert peft_config  is not None, "❌ peft_config is None!"
    assert output_dir   is not None and output_dir != "", \
        "❌ output_dir is empty or None!"
    assert "train" in dataset, \
        "❌ dataset missing 'train' split!"

    hp_overrides = hp_overrides or {}

    # ----------------------------------------------------------------
    # Step 2: Resolve eval split
    # Priority: validation > val > test > dev > first non-train available
    # Fallback (Option C): use train split as eval proxy
    #   - loss always improves → early stopping won't trigger
    #   - but checkpoints saved and load_best_model_at_end still active
    # ----------------------------------------------------------------
    EVAL_SPLIT_PRIORITY = ["validation", "val", "test", "dev"]
    eval_split = next(
        (s for s in EVAL_SPLIT_PRIORITY if s in dataset.keys()),
        next((s for s in dataset.keys() if s != "train"), None)
    )

    if eval_split is None:
        print(f"⚠️  No eval split found — using train split as eval proxy.")
        print(f"⚠️  Early stopping will not trigger. load_best_model_at_end still active.")
        eval_dataset          = dataset["train"]
        metric_for_best_model = "loss"       # train loss — always improves
    else:
        eval_dataset          = dataset[eval_split]
        metric_for_best_model = "eval_loss"  # ✅ validation loss

    # ----------------------------------------------------------------
    # Step 3: Resolve eval/save strategy
    # save_strategy always synced to eval_strategy — mismatch impossible
    # Both must match for load_best_model_at_end=True
    # ----------------------------------------------------------------
    eval_strategy = hp_overrides.get("eval_strategy", "epoch")
    save_strategy = eval_strategy

    # ----------------------------------------------------------------
    # Step 4: Object-based precision detection
    # ----------------------------------------------------------------
    model_dtype = next(model.parameters()).dtype
    use_bf16    = (model_dtype == torch.bfloat16)
    use_fp16    = (model_dtype == torch.float16)

    # ----------------------------------------------------------------
    # Step 5: Hardware-based optimizer selection
    # PyTorch >= 2.0 confirmed — fused AdamW safe on CUDA
    # ----------------------------------------------------------------
    is_cuda   = "cuda" in str(device)
    optimizer = "adamw_torch_fused" if is_cuda else "adamw_torch"

    # ----------------------------------------------------------------
    # Step 6: Early stopping patience — not an SFTConfig param
    # Kept separate for EarlyStoppingCallback
    # ----------------------------------------------------------------
    early_stopping_patience = hp_overrides.get("early_stopping_patience", 3)

    # ----------------------------------------------------------------
    # Step 7: Print header
    # ----------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"⚙️  setup_training_engine")
    print(f"   device           : {device}")
    print(f"   output_dir       : {output_dir}")
    print(f"   do_label_masking : {do_label_masking}")
    print(f"   model_dtype      : {model_dtype}")
    print(f"   bf16 / fp16      : {use_bf16} / {use_fp16}")
    print(f"   optimizer        : {optimizer}")
    print(f"   eval_split       : {eval_split if eval_split else 'None — train used as proxy'}")
    print(f"   eval_strategy    : {eval_strategy}")
    print(f"   metric           : {metric_for_best_model}")
    print(f"   patience         : {early_stopping_patience}")

    # ----------------------------------------------------------------
    # Step 8: SFT Configuration
    # DistilGPT2 defaults (current):
    #   batch_size=8,  grad_accum=1,  effective_batch=8
    #   num_epochs=5,  max_length=1024, patience=3
    #
    # TinyLlama recommended (pass via hp_overrides):
    #   batch_size=4,  grad_accum=4,  effective_batch=16
    #   num_epochs=3,  max_length=1024, patience=2
    # ----------------------------------------------------------------
    sft_args = SFTConfig(
        output_dir                  = output_dir,
        seed                        = seed,

        # Sequence length
        max_length                  = hp_overrides.get("max_length",                  1024),
        packing                     = hp_overrides.get("packing",                     False),

        # Hyperparameters
        # learning_rate from model_config (LoRA-specific), rest from hp_overrides
        learning_rate               = hp_overrides.get("learning_rate",               model_config.get("learning_rate", 2e-4)),
        per_device_train_batch_size = hp_overrides.get("batch_size",                  8),
        gradient_accumulation_steps = hp_overrides.get("gradient_accumulation_steps", 1),
        num_train_epochs            = hp_overrides.get("num_train_epochs",            5),
        weight_decay                = hp_overrides.get("weight_decay",                0.01),

        # Validation & checkpointing
        eval_strategy               = eval_strategy,
        save_strategy               = save_strategy,
        eval_steps                  = hp_overrides.get("eval_steps",                  500),
        save_steps                  = hp_overrides.get("save_steps",                  500),
        save_total_limit            = 2,
        load_best_model_at_end      = True,
        metric_for_best_model       = metric_for_best_model,
        greater_is_better           = False,
        dataloader_num_workers      = 0,

        # Hardware-derived — not overrideable via hp_overrides
        optim                       = optimizer,
        bf16                        = use_bf16,
        fp16                        = use_fp16,

        # Logging
        logging_steps               = hp_overrides.get("logging_steps",               10),

        # SFT specifics
        **( {"dataset_kwargs": {"skip_prepare_dataset": True}}
           if do_label_masking else
           {"dataset_text_field": "text"}
        ),
        report_to                   = "none",
    )

    if debug:
        print(f"\n[DEBUG] Full SFTConfig:")
        print(sft_args.to_json_string())

    # ----------------------------------------------------------------
    # Step 9: Early stopping callback
    # Always active — won't trigger when train used as eval proxy
    # since loss always improves on train data
    # ----------------------------------------------------------------
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience
        )
    ]

    # ----------------------------------------------------------------
    # Step 10: Initialize SFTTrainer
    # ----------------------------------------------------------------
    trainer = SFTTrainer(
        model            = model,
        args             = sft_args,
        peft_config      = peft_config,
        train_dataset    = dataset["train"],
        eval_dataset     = eval_dataset,
        processing_class = tokenizer,
        callbacks        = callbacks
    )

    # ----------------------------------------------------------------
    # Step 11: Summary
    # ----------------------------------------------------------------
    print(f"\n   Trainable Parameters:")
    trainer.model.print_trainable_parameters()
    print(f"{'='*50}\n")

    return trainer


def run_training_engine(
    trainer,
    device,
    hp_overrides = None,
    debug        = False
):
    """
    Module 5: Runs the training loop on a configured SFTTrainer.

    Args:
        trainer      (SFTTrainer):   Configured trainer from setup_training_engine.
        device       (torch.device): Execution device (e.g. 'cuda', 'cpu').
        hp_overrides (dict):         Optional hyperparameter overrides for reference.
        debug        (bool):         If True, prints additional debug info.

    Returns:
        trainer (SFTTrainer): Trainer with completed training state.
    """

    # ----------------------------------------------------------------
    # Step 1: Early validation
    # ----------------------------------------------------------------
    assert trainer is not None, "❌ trainer is None!"
    assert device  is not None, "❌ device is None!"

    hp_overrides = hp_overrides or {}

    # ----------------------------------------------------------------
    # Step 2: Print header
    # ----------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"🚀 run_training_engine")
    print(f"   device         : {device}")
    print(f"{'='*50}")

    # ----------------------------------------------------------------
    # Step 3: Run training
    # ----------------------------------------------------------------
    train_result = trainer.train()

    # ----------------------------------------------------------------
    # Step 4: Summary
    # ----------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"✅ run_training_engine complete")
    print(f"   global_step    : {train_result.global_step}")
    print(f"   training_loss  : {train_result.training_loss:.4f}")
    print(f"   best_metric    : {trainer.state.best_metric}")
    print(f"   best_model_cp  : {trainer.state.best_model_checkpoint}")
    if debug:
        print(f"\n[DEBUG] Full train_result metrics:")
        for key, val in train_result.metrics.items():
            print(f"   {key:<35} : {val}")
    print(f"{'='*50}\n")

    return trainer


def save_training_results(
    trainer,
    tokenizer,
    output_dir,
    folder_name = "best_model",
    verify      = True,
    debug       = False
):
    """
    Module 6: Captures the full state of the model:
    Config, Adapters, Tokenizer, and Training History.

    Args:
        trainer     (SFTTrainer): Trainer with completed training state.
        tokenizer:                The loaded and updated tokenizer.
        output_dir  (str):        Root output directory.
        folder_name (str):        Subfolder name for saved model. Default: 'best_model'.
        verify      (bool):       If True, runs file verification after save.
        debug       (bool):       If True, prints additional debug info.

    Returns:
        save_path (str | None): Path to saved model, or None if verification failed.
    """

    # ----------------------------------------------------------------
    # Step 1: Early validation
    # ----------------------------------------------------------------
    assert trainer    is not None,                    "❌ trainer is None!"
    assert tokenizer  is not None,                    "❌ tokenizer is None!"
    assert output_dir is not None and output_dir != "", "❌ output_dir is empty or None!"
    assert folder_name is not None and folder_name != "", "❌ folder_name is empty or None!"

    save_path = os.path.join(output_dir, folder_name)
    os.makedirs(save_path, exist_ok=True)

    # ----------------------------------------------------------------
    # Step 2: Print header
    # ----------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"💾 save_training_results")
    print(f"   save_path      : {save_path}")
    print(f"   verify         : {verify}")
    print(f"{'='*50}")

    # ----------------------------------------------------------------
    # Step 3: Save LoRA adapters
    # trainer.save_model() saves:
    #   - adapter_model.safetensors
    #   - adapter_config.json
    #   - config.json (with quantization_config embedded by HuggingFace)
    # Note: config.json has resized vocab_size — NOT used at inference
    # Inference loads naked base model from hub then resizes manually
    # ----------------------------------------------------------------
    trainer.save_model(save_path)
    print("   ✅ LoRA adapters saved.")

    # ----------------------------------------------------------------
    # Step 4: Save base config & quantization config separately
    # Belt-and-suspenders — guaranteed clean source for inference reload
    # ----------------------------------------------------------------
    inner_model = (
        trainer.model.base_model.model
        if hasattr(trainer.model, "base_model")
        else trainer.model
    )

    if hasattr(inner_model, "config"):
        inner_model.config.save_pretrained(save_path)
        if hasattr(inner_model.config, "quantization_config"):
            inner_model.config.quantization_config.to_json_file(
                os.path.join(save_path, "quantization_config.json")
            )
        print("   ✅ quantization metadata saved.")

    # ----------------------------------------------------------------
    # Step 5: Save generation config
    # Ensures temperature, top_p etc. are consistent at inference
    # ----------------------------------------------------------------
    if hasattr(inner_model, "generation_config"):
        inner_model.generation_config.save_pretrained(save_path)
        print("   ✅ Generation config saved.")

    # ----------------------------------------------------------------
    # Step 6: Save tokenizer
    # ----------------------------------------------------------------
    tokenizer.save_pretrained(save_path)
    print("   ✅ Tokenizer and special tokens saved.")

    # ----------------------------------------------------------------
    # Step 7: Save trainer state
    # Useful for plotting loss curve later
    # ----------------------------------------------------------------
    trainer.state.save_to_json(os.path.join(save_path, "trainer_state.json"))
    print("   ✅ Training history (loss/steps) saved.")

    if debug:
        print(f"\n[DEBUG] Files saved:")
        for f in sorted(os.listdir(save_path)):
            fpath = os.path.join(save_path, f)
            size  = os.path.getsize(fpath) / 1024**2
            print(f"   {f:<45} : {size:.2f} MB")


    # ----------------------------------------------------------------
    # Step 8: Verification
    # Weights checked separately to handle both .safetensors and .bin
    # ----------------------------------------------------------------
    if verify:
        print(f"\n🔍 Verifying saved model files...")

        essentials = [
            "adapter_config.json",
            "tokenizer_config.json",
            "config.json"
        ]
        has_weights = (
            os.path.exists(os.path.join(save_path, "adapter_model.safetensors")) or
            os.path.exists(os.path.join(save_path, "adapter_model.bin"))
        )
        missing = [f for f in essentials if not os.path.exists(os.path.join(save_path, f))]

        if not missing and has_weights:
            print("   ✅ Verification passed — all essential files present.")

            # ----------------------------------------------------------------
            # Step 9: Cleanup checkpoint dirs — only after successful verify
            # ----------------------------------------------------------------
            print(f"   🗑️  Cleaning up checkpoints in {output_dir}...")
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isdir(item_path) and item.startswith("checkpoint-"):
                    shutil.rmtree(item_path)
            print(f"   ✅ Cleanup complete. Disk space reclaimed.")
            return save_path
        else:
            if missing:
                print(f"   ❌ Verification FAILED — missing files: {missing}")
            if not has_weights:
                print(f"   ❌ Verification FAILED — no adapter weights found "
                      f"(adapter_model.safetensors or adapter_model.bin)")
            print(f"   ⚠️  Checkpoints preserved for safety.")
            return None
    else:
        return save_path


def print_training_summary(trainer):

    if trainer is None:
        print("❌ trainer is None — no summary to print.")
        return

    print(f"\n{'='*50}")
    print(f"       📊 FINAL TRAINING SUMMARY REPORT")
    print(f"{'='*50}")

    print(f"🥇 Best Checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"🎯 Best Eval Loss:  {trainer.state.best_metric if trainer.state.best_metric else 'N/A'}")
    print(f"🚀 Global Steps:    {trainer.state.global_step}")

    if trainer.state.log_history:
        df = pd.DataFrame(trainer.state.log_history)

        if 'step' in df.columns:
            df         = df.dropna(subset=['step'])
            summary_df = df.groupby('step').first().reset_index()
        else:
            summary_df = df

        cols = [c for c in ['step', 'loss', 'eval_loss', 'learning_rate']
                if c in summary_df.columns]

        print("\n📈 Trend Highlights (Last 5 Logs):")
        print(summary_df[cols].tail(5).to_string(index=False))

    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    all_params       = sum(p.numel() for p in trainer.model.parameters())
    trainable_pct    = (100 * trainable_params / all_params) if all_params > 0 else 0.0

    print(f"\n🧠 Model Efficiency:")
    print(f"   Trainable Params : {trainable_params:,}")
    print(f"   Total Params     : {all_params:,}")
    print(f"   Trainable %      : {trainable_pct:.4f}%")
    print(f"{'='*50}\n")


def plot_training_results(trainer):
    """
    Visualizes training and validation loss history from trainer state.
    """

    if trainer is None:
        print("❌ trainer is None — nothing to plot.")
        return

    history = trainer.state.log_history

    train_loss  = [log["loss"]      for log in history if "loss"      in log]
    train_steps = [log["step"]      for log in history if "loss"      in log]
    val_loss    = [log["eval_loss"] for log in history if "eval_loss" in log]
    val_steps   = [log["step"]      for log in history if "eval_loss" in log]

    if not train_loss and not val_loss:
        print("⚠️  No loss data available to plot.")
        return

    plt.figure(figsize=(10, 6))

    if train_loss:
        plt.plot(train_steps, train_loss,
                 label='Training Loss (Steps)',
                 color='#1f77b4', alpha=0.4, linestyle='--')

    if val_loss:
        plt.plot(val_steps, val_loss,
                 label='Validation Loss (Epoch)',
                 color='#ff7f0e', marker='o', linewidth=2)

        best_val_loss = min(val_loss)
        best_step     = val_steps[val_loss.index(best_val_loss)]  # first occurrence
        loss_range    = max(val_loss) - min(val_loss)
        offset        = max(loss_range * 0.1, 0.05)

        plt.annotate(
            f'Best Model\nLoss: {best_val_loss:.4f}',
            xy         = (best_step, best_val_loss),
            xytext     = (best_step, best_val_loss + offset),
            arrowprops = dict(facecolor='black', shrink=0.05),
            horizontalalignment = 'center'
        )

    plt.title("Fine-Tuning Progress: Training vs. Validation Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    print(f"✅ Plotting complete. Best Validation Loss: {min(val_loss):.4f}" 
          if val_loss else 
          "✅ Plotting complete. No validation loss recorded.")
    

def load_inference_model(
    model_name,
    best_model_path,
    device,
    load_adapter = True,
    debug = False
):
    """
    Reloads base model and attaches trained LoRA adapters for inference.
    Loads naked base model from hub to avoid vocab_size mismatch,
    then resizes embeddings to match saved tokenizer.

    Args:
        model_name      (str):          HuggingFace model id (e.g. 'distilgpt2').
        best_model_path (str):          Path to saved model folder.
        device          (torch.device): Execution device (e.g. 'cuda', 'cpu').
        load_adapter    (bool):         If True, attaches LoRA adapters (finetuned).
                                        If False, returns naked base model (baseline).
        debug           (bool):         If True, prints additional debug info.

    Returns:
        model:          Model ready for inference.
        tokenizer:      Tokenizer with all special tokens.
        actual_device:  Device model is loaded on.
    """

    # ----------------------------------------------------------------
    # Step 1: Early validation
    # ----------------------------------------------------------------
    assert model_name      is not None and model_name != "", \
        "❌ model_name is empty or None!"
    assert best_model_path is not None and best_model_path != "", \
        "❌ best_model_path is empty or None!"
    assert os.path.exists(best_model_path), \
        f"❌ best_model_path does not exist: {best_model_path}"

    # ----------------------------------------------------------------
    # Step 2: Resolve dtype and device_map
    # ----------------------------------------------------------------
    model_dtype, device_map = resolve_model_dtype_and_device(device)
    is_cuda = "cuda" in str(device) and torch.cuda.is_available()

    print(f"\n{'='*50}")
    print(f"🏗️  load_inference_model")
    print(f"   model_name      : {model_name}")
    print(f"   best_model_path : {best_model_path}")
    print(f"   load_adapter    : {load_adapter}")
    print(f"   device          : {device}")
    print(f"   model_dtype     : {model_dtype}")
    print(f"   device_map      : {device_map}")
    print(f"{'='*50}")

    # ----------------------------------------------------------------
    # Step 3: Load tokenizer from saved path
    # Includes all custom special tokens from training
    # ----------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(best_model_path)
    print(f"   ✅ Tokenizer loaded — vocab size: {len(tokenizer)}")

    # ----------------------------------------------------------------
    # Step 4: Resolve quantization config from saved file
    # ----------------------------------------------------------------
    bnb_config  = None
    quant_label = "None (Full Precision)"

    if is_cuda:
        quant_path = os.path.join(best_model_path, "quantization_config.json")
        if os.path.exists(quant_path):
            try:
                with open(quant_path, "r") as f:
                    q_dict = json.load(f)

                if q_dict.get("load_in_4bit"):
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit              = True,
                        bnb_4bit_quant_type       = q_dict.get("bnb_4bit_quant_type",       "nf4"),
                        bnb_4bit_compute_dtype    = model_dtype,
                        bnb_4bit_use_double_quant = q_dict.get("bnb_4bit_use_double_quant", True),
                    )
                    quant_label = "4-bit (NF4)"

                elif q_dict.get("load_in_8bit"):
                    bnb_config  = BitsAndBytesConfig(load_in_8bit=True)
                    quant_label = "8-bit"

                else:
                    print(f"   ℹ️  quantization_config.json found but no 4bit/8bit flag — full precision.")

            except Exception as e:
                print(f"   ⚠️  Failed to read quantization_config.json — "
                      f"falling back to full precision. Error: {e}")
        else:
            print(f"   ℹ️  No quantization_config.json — loading full precision.")
    else:
        print(f"   ℹ️  CPU device — skipping quantization.")

    # ----------------------------------------------------------------
    # Step 5: Load NAKED base model from hub
    # Intentionally NOT using best_model_path config —
    # config.json has resized vocab_size which causes shape mismatch crash
    # Embeddings resized manually in Step 6
    # ----------------------------------------------------------------
    print(f"   📥 Loading naked base model: {model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        dtype               = model_dtype,
        device_map          = device_map,
        low_cpu_mem_usage   = True,
        trust_remote_code   = True
    )
    print(f"   ✅ Base model loaded.")

    model_transformers_ver = transformers.__version__
    assert model_transformers_ver is not None and \
       version.parse(model_transformers_ver) >= version.parse("4.46.0"), \
        f"❌ transformers >= 4.46.0 required for mean_resizing. " \
        f"Found: '{model_transformers_ver}'"

    # ----------------------------------------------------------------
    # Step 6: Resize embeddings + mean-init new slots
    # Matches update_model_and_tokenizer logic from training exactly
    # mean_resizing=True handles new slot init automatically (transformers >= 4.46)
    # ----------------------------------------------------------------
    old_vocab = base_model.get_input_embeddings().weight.shape[0]
    base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8, mean_resizing = True)
    new_vocab = base_model.get_input_embeddings().weight.shape[0]
    print(f"   ✅ Embeddings resized: {old_vocab} → {new_vocab} (+{new_vocab - old_vocab} slots, mean-init by HF)")

    # ----------------------------------------------------------------
    # Step 7: Sync token IDs
    # Prevents generation from never stopping or using wrong pad token
    # ----------------------------------------------------------------
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.eos_token_id = tokenizer.eos_token_id
    base_model.config.bos_token_id = tokenizer.bos_token_id
    print(f"   ✅ Token IDs synced — "
          f"pad={tokenizer.pad_token_id}, "
          f"eos={tokenizer.eos_token_id}, "
          f"bos={tokenizer.bos_token_id}")

    # ----------------------------------------------------------------
    # Step 8: Attach LoRA adapters
    # ----------------------------------------------------------------
    if load_adapter:
        print(f"   🧠 Attaching LoRA adapters...")
        model = PeftModel.from_pretrained(base_model, best_model_path)
        print(f"   ✅ LoRA adapters attached.")
    else:
        model = base_model
        print(f"   ℹ️  load_adapter=False — returning naked base model.")


    # ----------------------------------------------------------------
    # Step 9: Load generation config
    # ----------------------------------------------------------------
    try:
        gen_config = GenerationConfig.from_pretrained(best_model_path)
        model.generation_config = gen_config
        print(f"   ✅ Generation config loaded.")
    except Exception as e:
        print(f"   ⚠️  No generation_config.json — using model defaults. ({e})")

    # ----------------------------------------------------------------
    # Step 10: Set inference mode
    # ----------------------------------------------------------------
    model.eval()
    model.config.use_cache = True   # re-enable — disabled during training
    actual_device = next(model.parameters()).device
    if actual_device is None:
        actual_device = device

    if debug:
        print(f"\n[DEBUG] Inference model state:")
        print(f"   vocab_size     : {len(tokenizer)}")
        print(f"   embedding_size : {model.get_input_embeddings().weight.shape[0]}")
        print(f"   actual_device  : {actual_device}")
        print(f"   use_cache      : {model.config.use_cache}")
        print(f"   pad_token_id   : {model.config.pad_token_id}")
        print(f"   eos_token_id   : {model.config.eos_token_id}")
        print(f"   bos_token_id   : {model.config.bos_token_id}")

    # ----------------------------------------------------------------
    # Step 11: Summary
    # ----------------------------------------------------------------
    memory_gb = model.get_memory_footprint() / (1024**3)

    print(f"\n{'='*50}")
    print(f"✅ load_inference_model complete")
    print(f"   model       : {model_name}")
    print(f"   adapter     : {'LoRA attached' if load_adapter else 'None (base)'}")
    print(f"   quant       : {quant_label}")
    print(f"   memory      : {memory_gb:.2f} GB")
    print(f"   vocab       : {len(tokenizer)}")
    print(f"   device      : {actual_device}")
    print(f"{'='*50}\n")

    return model, tokenizer, actual_device
