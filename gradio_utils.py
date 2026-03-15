"""
gradio_utils.py
Utility functions for Gradio app configuration, naming, and launch management.
Used by both Bansal_Ashish_IITD_AIML05_Module5_chatbot.ipynb and gradio_app.py.
"""

import os
import json
import re
import time
import hashlib
import subprocess
import sys

__all__ = [
    # Naming & key generation
    "make_dataset_key",
    "make_run_key",
    "make_model_dump_dir",
    # Config file management
    "load_gradio_config",
    "save_gradio_config",
    "upsert_gradio_config",
    # Disk reading helpers
    "load_sample_queries",
    "save_sample_queries",
    "load_run_metrics",
    "list_available_runs",
    # Subprocess launcher
    "launch_gradio",
    # Shell script generator
    "write_launch_scripts",
]

# ─────────────────────────────────────────────────────────────
# Naming & key generation
# ─────────────────────────────────────────────────────────────

def make_dataset_key(dataset_name):
    """
    Unique short key from dataset name.
    Works for file paths, URLs, HuggingFace ids, or any string identifier.
    Format: {basename}_{md5[:6]}
    """
    # Strip trailing slashes, take last segment after any separator
    clean = dataset_name.rstrip("/").rstrip("\\")
    # Split on both / and \ to handle URLs, HF ids, and local paths
    basename = re.split(r"[/\\]", clean)[-1]
    # Remove file extension if present
    basename = basename.split(".")[0]
    # Truncate to 20 chars for readability
    basename = basename[:20]
    # Hash full original string for uniqueness
    hash6 = hashlib.md5(dataset_name.encode()).hexdigest()[:6]
    return f"{basename}_{hash6}"


def make_run_key(model_id, dataset_key):
    """Composite key used in config and directory naming: {model_id}__{dataset_key}"""
    return f"{model_id}__{dataset_key}"


def make_model_dump_dir(model_id, dataset_key, base_dir=None):
    """Full path to the model-dataset artefact directory."""
    if base_dir is None:
        base_dir = os.getcwd()
    return os.path.join(base_dir, make_run_key(model_id, dataset_key))


# ─────────────────────────────────────────────────────────────
# Config file management
# ─────────────────────────────────────────────────────────────

def load_gradio_config(config_path="gradio_config.json"):
    """Load gradio_config.json from disk. Returns empty dict if file does not exist."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return json.load(f)


def save_gradio_config(config, config_path="gradio_config.json"):
    """Write config dict to gradio_config.json."""
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✅ gradio_config.json saved → {config_path}")


def upsert_gradio_config(
    model_id,
    model_config,
    dataset_name,
    model_dump_dir,
    author       = None,
    config_path  = "gradio_config.json",
):
    """
    Add or update one model-dataset run entry in gradio_config.json.
    All other existing entries are preserved.

    Args:
        model_id       : Key from MODEL_CONFIGS e.g. "distilgpt2"
        model_config   : Entry from MODEL_CONFIGS dict
        dataset_name   : Full dataset path/name used in notebook
        model_dump_dir : Root artefact directory for this run
        author         : AUTHOR dict from notebook (optional, written once)
        config_path    : Path to gradio_config.json
    """
    config      = load_gradio_config(config_path)
    dataset_key = make_dataset_key(dataset_name)
    best_model_path = os.path.join(model_dump_dir, "best_model")

    # Author — written once, not overwritten if already present
    if author and "author" not in config:
        config["author"] = author

    # Dataset registry
    if "datasets" not in config:
        config["datasets"] = {}
    config["datasets"][dataset_key] = {
        "dataset_key":  dataset_key,
        "full_name":    os.path.basename(dataset_name),
        "display_name": os.path.basename(dataset_name).split(".")[0].replace("_", " ").title(),
    }

    # Model registry
    if "models" not in config:
        config["models"] = {}
    if model_id not in config["models"]:
        config["models"][model_id] = {
            "model_name": model_config["model_name"],
            "chat_style": model_config["chat_style"],
            "runs":       {},
        }

    # Run entry — upsert under model > runs > dataset_key
    config_dir   = os.path.dirname(os.path.abspath(config_path))
    rel_dump_dir = os.path.relpath(model_dump_dir, config_dir)

    config["models"][model_id]["runs"][dataset_key] = {
        "model_dump_dir":  rel_dump_dir,
        "best_model_path": os.path.join(rel_dump_dir, "best_model"),
    }

    save_gradio_config(config, config_path)

    print(f"   model_id   : {model_id}")
    print(f"   dataset_key: {dataset_key}")
    print(f"   dump_dir   : {model_dump_dir}")
    return config


# ─────────────────────────────────────────────────────────────
# Disk reading helpers
# ─────────────────────────────────────────────────────────────

def load_sample_queries(model_dump_dir):
    """
    Load pre-built sample queries from samples/sample_queries.json.
    Returns list of query strings, or empty list if file not found.
    """
    path = os.path.join(model_dump_dir, "samples", "sample_queries.json")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def save_sample_queries(queries, model_dump_dir):
    """Save list of query strings to samples/sample_queries.json."""
    samples_dir = os.path.join(model_dump_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    path = os.path.join(samples_dir, "sample_queries.json")
    with open(path, "w") as f:
        json.dump(queries, f, indent=2)
    print(f"✅ Sample queries saved → {path}")


def load_run_metrics(model_dump_dir, dataset_key):
    """
    Load evaluation metrics from eval/{dataset_key}_metrics.json.
    Returns dict or None if file not found.
    """
    path = os.path.join( model_dump_dir, "eval", f"{dataset_key}_metrics.json" )
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def list_available_runs(config_path="gradio_config.json"):
    """
    Return list of (model_id, dataset_key, display_label) tuples
    for all valid runs in gradio_config.json.
    Only includes runs where best_model_path exists on disk.
    """
    config = load_gradio_config(config_path)
    runs   = []
    for model_id, model_entry in config.get("models", {}).items():
        for dataset_key, run_entry in model_entry.get("runs", {}).items():
            best_path = run_entry.get("best_model_path", "")
            if os.path.exists(best_path):
                dataset_display = config.get("datasets", {}).get(dataset_key, {}).get("display_name", dataset_key)
                label = f"{model_id}  /  {dataset_display}"
                runs.append((model_id, dataset_key, label))
    return runs


# ─────────────────────────────────────────────────────────────
# Subprocess launcher
# ─────────────────────────────────────────────────────────────

def launch_gradio(
    config_path = "gradio_config.json",
    port        = 7860,
    share       = True,
    inbrowser   = True,
    detach      = True,
    log_file    = "gradio.log"
):
    """
    Launch gradio_app.py as a subprocess.
    detach=True → independent process, notebook cell completes immediately.
    detach=False → blocking, notebook waits (useful for debugging).
    """
    cmd = [
        sys.executable, "gradio_app.py",
        "--config", config_path,
        "--port",   str(port),
    ]
    if share:
        cmd.append("--share")
    if inbrowser:
        cmd.append("--inbrowser")

    if detach:
        log = open(log_file, "w") if log_file else subprocess.DEVNULL
        proc = subprocess.Popen(
            cmd,
            stdout            = log,
            stderr            = log,
            start_new_session = True,  # detach from parent process group
        )
        print(f"✅ Gradio launched (PID: {proc.pid})")
        print(f"   Local  : http://localhost:{port}")
        if share:
            print(f"   Waiting for public URL...")
            time.sleep(15)
            try:
                with open(log_file) as f:
                    for line in f:
                        if "gradio.live" in line or "127.0.0.1" in line:
                            print(f"   {line.strip()}")
            except Exception:
                print(f"   Check {log_file} for public URL")
        return proc
    else:
        subprocess.run(cmd)
        return None


# ─────────────────────────────────────────────────────────────
# Shell script generator
# ─────────────────────────────────────────────────────────────

def write_launch_scripts(config_path="gradio_config.json", port=7860):
    """
    Write platform-specific launcher scripts alongside gradio_app.py.
    Creates launch_gradio.sh (Linux/Mac) and launch_gradio.bat (Windows).
    """
    sh_content = f"""#!/bin/bash
# Launch Gradio Customer Support Chatbot
cd "$(dirname "$0")"
python gradio_app.py --config {config_path} --port {port}
"""
    bat_content = f"""@echo off
REM Launch Gradio Customer Support Chatbot
cd /d "%~dp0"
python gradio_app.py --config {config_path} --port {port}
pause
"""
    with open("launch_gradio.sh", "w") as f:
        f.write(sh_content)
    os.chmod("launch_gradio.sh", 0o755)

    with open("launch_gradio.bat", "w") as f:
        f.write(bat_content)

    print("✅ Launch scripts written:")
    print("   launch_gradio.sh  (Linux / Mac)")
    print("   launch_gradio.bat (Windows)")
    return ["launch_gradio.sh", "launch_gradio.bat"]
