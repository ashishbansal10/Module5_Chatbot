import os
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Used in visual_atlas for heatmap/boxplot

# HuggingFace Datasets
from datasets import load_dataset, Dataset, DatasetDict

# Scikit-learn utilities
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

# Sentence Transformers
from sentence_transformers import SentenceTransformer


# Public API - Cleanly separated
__all__ = [
    "load_data",
    "analyze_data",
    "map_schema",
    "semantic_analysis",
    "audit_report",
    "visual_atlas",
    "sample_data",
    "validate_sample",
    "sampling_report",
    "split_data",
    "save_sampled_data",
    "save_splits"
]

# ============================================================
# DATA PIPELINE HELPER ROUTINES
# ============================================================

# ============================================================
# UTILITY
# ============================================================
def get_file_basename(dataset_name):
    return os.path.basename(dataset_name).split(".")[0]

# ============================================================
# 1: INGESTOR
# ============================================================
def load_data(dataset_name):
    """
    Loads dataset from HuggingFace or local file.
    Drops fully duplicate rows and all-null rows.

    Args:
        dataset_name (str): HuggingFace dataset name or local file path.

    Returns:
        pd.DataFrame: Raw DataFrame with original column names.
    """
    if os.path.exists(dataset_name):
        ext = os.path.splitext(dataset_name)[-1].lower()
        if ext == ".csv":
            df = pd.read_csv(dataset_name)
        elif ext in [".json", ".jsonl"]:
            df = pd.read_json(dataset_name, lines=(ext == ".jsonl"))
        else:
            raise ValueError(f"❌ Unsupported file type: {ext}")
    else:
        df = pd.DataFrame(load_dataset(dataset_name, split="train"))

    before = len(df)
    df = df.drop_duplicates().dropna(how="all").reset_index(drop=True)

    print(f"✅ load_data: {len(df):,} rows ({before - len(df):,} dropped)")
    return df


# ============================================================
# 2: ANALYSIS
# ============================================================
def analyze_data(df, base_name):
    """
    Prints basic dataset statistics and plots intent distribution.
    Saves plot to <basename>.analysis.png and displays inline.

    Args:
        df        (pd.DataFrame): Cleaned DataFrame.
        base_name (str):          Dataset name for file naming.
    """
    plot_file  = f"{base_name}.analysis.png"
    intent_col = "intent"

    print(f"{'='*50}")
    print(f"📊 analyze_data")
    print(f"{'='*50}")

    print(f"\n   Shape          : {df.shape}")
    print(f"   Null values    :\n{df.isnull().sum().to_string()}")
    print(f"\n--- info ---")
    df.info()
    print(f"\n--- sample row ---")
    row = df.iloc[0]
    for col, val in row.items():
        print(f"   {col:<12} : {val}")

    # Intent summary
    intent_counts_display = df[intent_col].value_counts().sort_index()
    print(f"\n--- intents ({df[intent_col].nunique()}) ---")
    print(f"   Avg q/intent : {intent_counts_display.mean():.1f}")
    top3    = intent_counts_display.sort_values(ascending=False).head(3)
    bottom3 = intent_counts_display.sort_values(ascending=False).tail(3)
    print(f"   Top 3:")
    for intent, count in top3.items():
        print(f"   {intent:<40} : {count:>5,}")
    print(f"   Bottom 3:")
    for intent, count in bottom3.items():
        print(f"   {intent:<40} : {count:>5,}")

    # Intent distribution
    intent_counts = df[intent_col].value_counts().sort_values()
    n_intents     = len(intent_counts)
    mean_count    = intent_counts.mean()
    std_count     = intent_counts.std()

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, n_intents * 0.25)))
    fig.suptitle(f"Intent Distribution — {base_name}", fontsize=14)

    axes[0].barh(intent_counts.index, intent_counts.values, color="steelblue", alpha=0.8)
    axes[0].axvline(mean_count, color="red",    linestyle="--", label=f"Mean={mean_count:.0f}")
    axes[0].axvline(mean_count - std_count, color="orange", linestyle=":", label=f"-1σ={mean_count-std_count:.0f}")
    axes[0].axvline(mean_count + std_count, color="orange", linestyle=":", label=f"+1σ={mean_count+std_count:.0f}")
    axes[0].set_xlabel("Query Count")
    axes[0].set_title("Queries per Intent")
    axes[0].legend(fontsize=8)

    bins = max(10, int((intent_counts.max() - intent_counts.min()) / 20))
    axes[1].hist(intent_counts.values, bins=bins, color="coral", alpha=0.8, edgecolor="white")
    axes[1].set_xlim(0, intent_counts.max() * 1.1)
    axes[1].axvline(mean_count, color="red", linestyle="--", label=f"Mean={mean_count:.0f}")
    axes[1].set_xlabel("Query Count")
    axes[1].set_ylabel("Number of Intents")
    axes[1].set_title("Distribution of Intent Sizes")
    axes[1].legend(fontsize=8)

    balance_ratio = intent_counts.min() / intent_counts.max()
    label         = "Balanced" if balance_ratio > 0.5 else "⚠️ Imbalanced"
    fig.text(0.5, 0.01, f"{label} (min/max ratio: {balance_ratio:.2f})",
             ha="center", fontsize=11)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ analyze_data saved: {plot_file}\n")


# ============================================================
# 3: MAP SCHEMA
# ============================================================
def map_schema(df, schema_map):
    """
    Renames source columns to output schema names.
    Drops other columns, cleans empty strings, deduplicates on query.

    Args:
        df         (pd.DataFrame): Raw DataFrame.
        schema_map (dict):         Column mapping config.

    Returns:
        pd.DataFrame: DataFrame with output column names only.
    """
    rename_map = {
        schema_map["q"]["in"]: schema_map["q"]["out"],
        schema_map["a"]["in"]: schema_map["a"]["out"],
        schema_map["i"]["in"]: schema_map["i"]["out"],
    }

    missing = [c for c in rename_map if c not in df.columns]
    assert not missing, f"❌ Columns not found: {missing}"

    q_out = schema_map["q"]["out"]
    a_out = schema_map["a"]["out"]

    df = df.rename(columns=rename_map)[list(rename_map.values())].copy()
    df = df[df[q_out].str.strip().ne("")]
    df = df[df[a_out].str.strip().ne("")]
    df = df.drop_duplicates(subset=[q_out]).reset_index(drop=True)

    print(f"✅ map_schema: {len(df):,} rows — cols: {list(df.columns)}")
    return df


# ============================================================
# 4: SEMANTIC ANALYSIS
# ============================================================
def _compute_data_hash(df, dataset_name):
    source_sig  = dataset_name
    size_sig    = str(len(df))
    field_sig   = ",".join(sorted(df.columns.tolist()))
    interval    = max(1, len(df) // 100)
    content_sig = df.iloc[::interval].head(100)["prompt"].to_string()
    combined    = f"{source_sig}|{size_sig}|{field_sig}|{content_sig}"
    return hashlib.md5(combined.encode()).hexdigest()[:8]


def semantic_analysis(df, dataset_name, cache_base_name, device="cuda", target_cluster_size=50, min_clusters=2, max_clusters=20):
    """
    Encodes queries, L2-normalizes, and clusters within each intent.
    Cluster count k scales dynamically with intent size.
    Caches normalized embeddings to disk.

    Args:
        df                  (pd.DataFrame): DataFrame with prompt, response, intent cols.
        dataset_name        (str):          Dataset name for cache file naming.
        cache_base_name     (str):          base name for cache file naming. Pass None to disable.
        device              (str):          'cuda' or 'cpu'.
        target_cluster_size (int):          Target rows per cluster (controls k).
        min_clusters        (int):          Minimum clusters per intent.
        max_clusters        (int):          Maximum clusters per intent.

    Returns:
        df         (pd.DataFrame): DataFrame with semantic_cluster column added.
        embeddings (np.ndarray):   L2-normalized embeddings.
    """

    q_col      = "prompt"
    intent_col = "intent"

    cache_file = None
    if cache_base_name is not None:
        data_hash  = _compute_data_hash(df, dataset_name)
        cache_file = f"{cache_base_name}.{data_hash}.embeddings.npy"

        # Cleanup stale cache
        for f in os.listdir("."):
            if f.startswith(cache_base_name) and f.endswith(".embeddings.npy") and f != cache_file:
                os.remove(f)
                print(f"🗑️  Removed stale cache: {f}")

    # Encode or load normalized cache
    if cache_file is not None and os.path.exists(cache_file):
        embeddings = np.load(cache_file)
        print(f"✅ Embeddings loaded from cache: {cache_file}")
    else:
        batch_size = 128 if "cuda" in str(device) else 32
        model      = SentenceTransformer("all-MiniLM-L12-v2", device=device)
        raw        = model.encode(
            df[q_col].tolist(),
            show_progress_bar = True,
            convert_to_numpy  = True,
            batch_size        = batch_size
        )
        embeddings = normalize(raw)          # L2 normalize before caching

        if cache_file is not None:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            np.save(cache_file, embeddings)
            print(f"✅ Embeddings encoded, normalized, cached: {cache_file}")

    # Dynamic k clustering per intent
    df         = df.copy()
    df["_idx"] = range(len(df))

    def _cluster_group(group):
        n = len(group)
        if n <= 1:
            return pd.Series([0], index=group.index)
        k    = max(min_clusters, min(max_clusters, n // target_cluster_size))
        embs = embeddings[group["_idx"].values]
        ids  = MiniBatchKMeans(
            n_clusters   = k,
            random_state = 42,
            n_init       = 3
        ).fit_predict(embs)
        return pd.Series(ids, index=group.index)

    df["semantic_cluster"] = df.groupby(intent_col, group_keys=False).apply(_cluster_group)
    df = df.drop(columns=["_idx"]).reset_index(drop=True)

    n_natures = df.groupby([intent_col, "semantic_cluster"]).ngroups
    print(f"✅ semantic_analysis: {n_natures} natures across {df[intent_col].nunique()} intents\n")

    return df, embeddings

# ============================================================
# 5: AUDIT REPORT & VISUAL ATLAS
# ============================================================
def _build_data_report(
    df_original,
    dataset_name,
    base_name,
    report_file,
    save_json  = True,
    df_sampled = None,
):
    """
    Core report builder for audit and sampling modes.
    df_sampled=None → audit mode. df_sampled=DataFrame → sampling mode.
    """
    is_sampling  = df_sampled is not None
    intent_col   = "intent"
    has_clusters = "semantic_cluster" in df_original.columns
 
    # ── Base counts ───────────────────────────────────────────
    orig_counts    = df_original[intent_col].value_counts().sort_values(ascending=False)
    sampled_counts = df_sampled[intent_col].value_counts() if is_sampling else None
 
    # ── Nature info ───────────────────────────────────────────
    if has_clusters:
        orig_natures_per_intent = df_original.groupby(intent_col)["semantic_cluster"].nunique()
        orig_nature_sizes       = df_original.groupby([intent_col, "semantic_cluster"]).size()
        total_natures           = df_original.groupby([intent_col, "semantic_cluster"]).ngroups
        avg_natures_per_intent  = round(float(orig_natures_per_intent.mean()), 2)
        avg_q_per_nature        = round(len(df_original) / total_natures, 1)
        min_nature_size         = int(orig_nature_sizes.min())
        max_nature_size         = int(orig_nature_sizes.max())
 
        if is_sampling:
            sampled_natures_per_intent = df_sampled.groupby(intent_col)["semantic_cluster"].nunique()
            sampled_total_natures      = df_sampled.groupby([intent_col, "semantic_cluster"]).ngroups
 
    # ── Balance ───────────────────────────────────────────────
    balance_ratio         = round(orig_counts.min() / orig_counts.max(), 3)
    balance_label         = "✅ Balanced" if balance_ratio > 0.5 else "⚠️  Imbalanced"
    avg_q_per_intent      = round(len(df_original) / df_original[intent_col].nunique(), 1)
    top3                  = orig_counts.head(3)
    bottom3               = orig_counts.tail(3)
 
    if is_sampling:
        sampled_balance_ratio = round(sampled_counts.min() / sampled_counts.max(), 3)
        sampled_balance_label = "✅ Balanced" if sampled_balance_ratio > 0.5 else "⚠️  Imbalanced"
 
    # ── Intent distribution list ──────────────────────────────
    intent_distribution = []
    for intent in orig_counts.index:
        entry = {
            "intent" : intent,
            "count"  : int(orig_counts[intent]),
        }
        if has_clusters:
            n_nat = int(orig_natures_per_intent.get(intent, 0))
            entry["num_natures"]     = n_nat
            entry["avg_q_per_nature"] = round(
                orig_counts[intent] / n_nat, 1
            ) if n_nat > 0 else 0
        if is_sampling:
            s_count = int(sampled_counts.get(intent, 0))
            entry["sampled_count"]   = s_count
            entry["retention_pct"]   = round(s_count / orig_counts[intent] * 100, 1)
            if has_clusters:
                s_nat = int(sampled_natures_per_intent.get(intent, 0))
                entry["sampled_natures"]      = s_nat
                entry["nature_coverage_pct"]  = round(
                    s_nat / orig_natures_per_intent.get(intent, 1) * 100, 1
                )
        intent_distribution.append(entry)
 
    # ── Build JSON ────────────────────────────────────────────
    report = {}
 
    report["meta"] = {
        "base_name"  : base_name,
        "source"     : dataset_name,
    }
 
    report["data_summary"] = {
        "original_rows"  : int(len(df_original)),
        "total_intents"  : int(df_original[intent_col].nunique()),
        **({"total_natures": int(total_natures)} if has_clusters else {}),
        **({"sampled_rows"  : int(len(df_sampled)),
            "squeeze_factor": round(len(df_original) / len(df_sampled), 1),
            "reduction_pct" : round(100 - len(df_sampled) / len(df_original) * 100, 2),
           } if is_sampling else {}),
    }
 
    report["intent_summary"] = {
        "avg_q_per_intent": avg_q_per_intent,
        "min_intent"    : {"name": str(orig_counts.index[-1]), "count": int(orig_counts.iloc[-1])},
        "max_intent"    : {"name": str(orig_counts.index[0]),  "count": int(orig_counts.iloc[0])},
        "balance_ratio" : balance_ratio,
        "top_3_intents"   : [{"intent": k, "count": int(v)} for k, v in top3.items()],
        "bottom_3_intents": [{"intent": k, "count": int(v)} for k, v in bottom3.items()],
        **({"sampled_intents"     : int(df_sampled[intent_col].nunique()),
            "missing_intents"     : list(
                set(df_original[intent_col].unique()) -
                set(df_sampled[intent_col].unique())
            ),
            "intent_coverage_rate": round(
                df_sampled[intent_col].nunique() /
                df_original[intent_col].nunique() * 100, 2
            ),
            "sampled_balance_ratio": sampled_balance_ratio,
           } if is_sampling else {}),
    }
 
    if has_clusters:
        report["nature_summary"] = {
            "total_natures"         : int(total_natures),
            "avg_natures_per_intent": avg_natures_per_intent,
            "avg_q_per_nature"      : avg_q_per_nature,
            "min_nature_size"       : min_nature_size,
            "max_nature_size"       : max_nature_size,
            **({"sampled_natures"      : int(sampled_total_natures),
                "nature_coverage_rate" : round(sampled_total_natures / total_natures * 100, 2),
               } if is_sampling else {}),
        }
 
    report["intent_distribution"] = intent_distribution
 
    # ── Save JSON ─────────────────────────────────────────────
    if save_json:
        with open(report_file, "w") as f:
            json.dump(report, f, indent=4)
        print(f"✅ Report saved → {report_file}")
 
    # ── Terminal print ────────────────────────────────────────
    mode = "SAMPLING REPORT" if is_sampling else "AUDIT REPORT"
    print(f"\n{'='*65}")
    print(f"📊 {mode}: {base_name}")
    print(f"{'='*65}")
 
    # data_summary
    print(f"\n── Data Summary ──")
    print(f"   Original rows  : {len(df_original):,}")
    if is_sampling:
        print(f"   Sampled rows   : {len(df_sampled):,}  "
              f"(squeeze {round(len(df_original)/len(df_sampled),1)}x, "
              f"-{round(100 - len(df_sampled)/len(df_original)*100,1)}%)")
    if has_clusters:
        print(f"   Total natures  : {total_natures}")
 
    # intent_summary
    print(f"\n── Intent Summary ──")
    print(f"   Total intents  : {df_original[intent_col].nunique()}")
    print(f"   Avg q/intent   : {avg_q_per_intent}")
    print(f"   Min intent     : {orig_counts.index[-1]} ({orig_counts.iloc[-1]:,})")
    print(f"   Max intent     : {orig_counts.index[0]}  ({orig_counts.iloc[0]:,})")
    print(f"   Balance ratio  : {balance_ratio}  {balance_label}")
    if is_sampling:
        print(f"   Sampled intents: {df_sampled[intent_col].nunique()} / {df_original[intent_col].nunique()}"
              f"  (coverage {round(df_sampled[intent_col].nunique()/df_original[intent_col].nunique()*100,1)}%)")
        missing = list(set(df_original[intent_col].unique()) - set(df_sampled[intent_col].unique()))
        if missing:
            print(f"   Missing intents: {missing}")
        print(f"   Sampled balance: {sampled_balance_ratio}  {sampled_balance_label}")
 
    # nature_summary
    if has_clusters:
        print(f"\n── Nature Summary ──")
        print(f"   Total natures      : {total_natures}")
        print(f"   Avg natures/intent : {avg_natures_per_intent}")
        print(f"   Avg q/nature       : {avg_q_per_nature}")
        print(f"   Nature size        : min={min_nature_size}  max={max_nature_size}")
        if is_sampling:
            print(f"   Sampled natures    : {sampled_total_natures} / {total_natures}"
                  f"  (coverage {round(sampled_total_natures/total_natures*100,1)}%)")
 
    # intent_distribution — top 3 and bottom 3
    def _print_intent_row(entry):
        line = f"   {entry['intent']:<40} | count: {entry['count']:>5}"
        if has_clusters:
            line += f"  | natures: {entry['num_natures']:>3}"
        if is_sampling:
            line += f"  | sampled: {entry['sampled_count']:>4} ({entry['retention_pct']}%)"
            if has_clusters:
                line += f"  | s_natures: {entry['sampled_natures']:>3} ({entry['nature_coverage_pct']}%)"
        print(line)
 
    print(f"\n── Top 3 Intents ──")
    for entry in intent_distribution[:3]:
        _print_intent_row(entry)
 
    print(f"\n── Bottom 3 Intents ──")
    for entry in intent_distribution[-3:]:
        _print_intent_row(entry)
 
    print(f"\n{'='*65}\n")


def audit_report(df, dataset_name, base_name, save_json=True):
    """Audit report for master dataset — terminal print + optional JSON."""
    report_file = f"{base_name}.audit_report.json"
    _build_data_report(
        df_original  = df,
        dataset_name = dataset_name,
        base_name    = base_name,
        report_file  = report_file,
        save_json    = save_json,
        df_sampled   = None,
    )


def visual_atlas(df, embeddings, base_name):
    """
    Generates 2 actionable plots to inform sampling decisions.
    Plot 3 (nature distribution) shown only if semantic_cluster present.
    Saves to <basename>.atlas.png and displays inline.

    Args:
        df         (pd.DataFrame): DataFrame with prompt, response, intent cols.
        embeddings (np.ndarray):   L2-normalized query embeddings.
        base_name  (str):          Base name for file naming.
    """
    output_file   = f"{base_name}.atlas.png"
    intent_col    = "intent"
    has_clusters  = "semantic_cluster" in df.columns
    n_plots       = 2 if has_clusters else 1
    intent_counts = df[intent_col].value_counts().sort_values()

    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, max(6, len(intent_counts) * 0.22)))
    if n_plots == 1:
        axes = [axes]
    fig.suptitle(f"Semantic Atlas: {base_name}", fontsize=14, fontweight="bold")

    # ----------------------------------------------------------------
    # Plot 1: Intent similarity heatmap
    # ----------------------------------------------------------------
    ax1   = axes[0]
    top10 = df[intent_col].value_counts().nlargest(10).index
    means = [embeddings[df[df[intent_col] == n].index].mean(axis=0) for n in top10]
    sim   = cosine_similarity(means)
    sns.heatmap(sim, xticklabels=top10, yticklabels=top10,
                annot=True, fmt=".2f", cmap="magma",
                ax=ax1, annot_kws={"size": 7})
    ax1.set_title("1. Intent Similarity (Top 10)")
    ax1.tick_params(axis="x", rotation=45, labelsize=7)
    ax1.tick_params(axis="y", rotation=0,  labelsize=7)

    # ----------------------------------------------------------------
    # Plot 2: Nature size distribution (box plot per intent)
    # ----------------------------------------------------------------
    if has_clusters:
        ax2          = axes[1]
        nature_sizes = df.groupby([intent_col, "semantic_cluster"]).size().reset_index(name="count")
        top15        = df[intent_col].value_counts().nlargest(15).index
        plot_data    = nature_sizes[nature_sizes[intent_col].isin(top15)]
        sns.boxplot(data=plot_data, y=intent_col, x="count",
                    order=top15, ax=ax2, hue=intent_col, palette="Set2", orient="h", legend=False)
        ax2.set_title("2. Nature Size Distribution (Top 15 Intents)")
        ax2.set_xlabel("Queries per Nature")
        ax2.set_ylabel("")
        ax2.tick_params(axis="y", labelsize=7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ visual_atlas saved: {output_file}\n")


# ============================================================
# 6: SAMPLER
# ============================================================
def sample_data(df, sample_size=1000, target_n_per_nature=2, random_state=42):
    """
    Samples dataset using squeeze logic — importance-weighted per nature.
    Prints sampling summary inline. Saves <basename>.sampled_report.json.
    Requires semantic_cluster column.

    Args:
        df                  (pd.DataFrame): DataFrame with semantic_cluster.
        sample_size         (int):          Target number of rows.
        target_n_per_nature (int):          Target samples per nature cluster.
        random_state        (int):          Random seed.

    Returns:
        pd.DataFrame: Sampled DataFrame.
    """
    assert "semantic_cluster" in df.columns, \
        "❌ semantic_cluster missing — run semantic_analysis() first"

    intent_col  = "intent"

    if sample_size is None or sample_size >= len(df):
        print(f"✅ sample_data: sample_size >= dataset — returning full dataset ({len(df):,} rows)")
        return df.reset_index(drop=True)

    nature_groups  = df.groupby([intent_col, "semantic_cluster"])
    nature_sizes   = nature_groups.size()
    total_natures  = len(nature_sizes)
    max_s, min_s   = nature_sizes.max(), nature_sizes.min()
    ideal_total    = total_natures * target_n_per_nature
    squeeze_factor = min(1.0, sample_size / ideal_total)

    final_selection = []
    for (intent, cluster), group in nature_groups:
        importance   = (len(group) - min_s) / (max_s - min_s) if max_s != min_s else 1.0
        dynamic_take = int(np.floor(target_n_per_nature * importance * squeeze_factor))
        take_n       = min(len(group), max(1, dynamic_take))
        final_selection.append(group.sample(n=take_n, random_state=random_state))

    selected = pd.concat(final_selection)

    gap = sample_size - len(selected)
    if gap > 0:
        remaining = df[~df.index.isin(selected.index)]
        if not remaining.empty:
            selected = pd.concat([
                selected,
                remaining.sample(n=min(gap, len(remaining)), random_state=random_state)
            ])
    elif gap < 0:
        print(f"⚠️  Overshoot by {abs(gap)} rows — accepting as-is")

    selected = selected.reset_index(drop=True)
    intent_counts     = selected[intent_col].value_counts()
    selected["_nature"] = (
        selected[intent_col].astype(str) + "_n" +
        selected["semantic_cluster"].astype(str)
    )
    n_natures_sampled = selected["_nature"].nunique()

    print(f"✅ sample_data: {len(selected):,} rows sampled  "
          f"| nature coverage {n_natures_sampled}/{total_natures} "
          f"({n_natures_sampled/total_natures*100:.1f}%)  "
          f"| squeeze {len(df)/len(selected):.1f}x")

    if squeeze_factor < 1.0:
        print(f"   ⚠️  Budget squeeze active: {squeeze_factor:.2f}")

    print(f"\n   Top 3 intents in sample:")
    for intent, count in intent_counts.head(3).items():
        print(f"   {intent:<40} : {count:>4} ({count/len(selected)*100:.1f}%)")

    print(f"\n   Bottom 3 intents by volume:")
    for intent, count in intent_counts.tail(3).items():
        print(f"   {intent:<40} : {count:>4} ({count/len(selected)*100:.1f}%)")

    selected = selected.drop(columns=["_nature"])
    return selected

def validate_sample(df_original, df_sampled):
    """
    Validates sampled DataFrame against original.
    Checks intent coverage, min samples per intent, nature coverage.
    Runs before split_data — both DataFrames must have intent col.

    Args:
        df_original (pd.DataFrame): Full original DataFrame.
        df_sampled  (pd.DataFrame): Sampled DataFrame.
    """
    intent_col      = "intent"
    has_clusters    = "semantic_cluster" in df_sampled.columns
    orig_intents    = set(df_original[intent_col].unique())
    sampled_intents = set(df_sampled[intent_col].unique())
    missing_intents = orig_intents - sampled_intents
    intent_counts   = df_sampled[intent_col].value_counts()
    under_5         = intent_counts[intent_counts < 5]

    print(f"\n{'='*55}")
    print(f"✅ SAMPLE VALIDATION")
    print(f"{'='*55}")
    print(f"   Original rows    : {len(df_original):,}")
    print(f"   Sampled rows     : {len(df_sampled):,}  "
          f"({len(df_sampled)/len(df_original)*100:.1f}% of original)")

    # Intent coverage
    if missing_intents:
        print(f"   ⚠️  Missing intents : {len(missing_intents)}")
        for i in list(missing_intents)[:5]:
            print(f"      • {i}")
    else:
        print(f"   Intent coverage  : {len(sampled_intents)}/{len(orig_intents)} ✅")

    # Min samples per intent
    if len(under_5) > 0:
        print(f"   ⚠️  Intents with <5 samples : {len(under_5)}")
        for intent, count in under_5.items():
            print(f"      • {intent} → {count}")
    else:
        print(f"   Min per intent   : {intent_counts.min()} ✅")

    # Nature coverage
    if has_clusters:
        orig_natures   = df_original.groupby(
            [intent_col, "semantic_cluster"]).ngroups \
            if "semantic_cluster" in df_original.columns else None
        sample_natures = df_sampled.groupby([intent_col, "semantic_cluster"]).ngroups
        if orig_natures:
            print(f"   Nature coverage  : {sample_natures}/{orig_natures} "
                  f"({sample_natures/orig_natures*100:.1f}%) ✅")
        else:
            print(f"   Sampled natures  : {sample_natures}")
    else:
        print(f"   ⚠️  semantic_cluster missing — nature coverage not checked")

    print(f"{'='*55}\n")


def sampling_report(df_original, df_sampled, dataset_name, base_name, save_json=True):
    """Sampling report — terminal print + optional JSON + distribution plot."""
    report_file = f"{base_name}.sampling_report.json"
    plot_file   = f"{base_name}.sampling.png"
 
    _build_data_report(
        df_original  = df_original,
        dataset_name = dataset_name,
        base_name    = base_name,
        report_file  = report_file,
        save_json    = save_json,
        df_sampled   = df_sampled,
    )
 
    # ── Plot ──────────────────────────────────────────────────
    intent_col     = "intent"
    orig_counts    = df_original[intent_col].value_counts()
    sampled_counts = df_sampled[intent_col].value_counts()
    intents        = orig_counts.sort_values(ascending=True).index.tolist()
    orig_vals      = [orig_counts.get(i, 0)    for i in intents]
    sample_vals    = [sampled_counts.get(i, 0) for i in intents]
    n              = len(intents)
    y              = range(n)
 
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, n * 0.25)), sharey=True)
    fig.suptitle(f"Sampling Report: {base_name}", fontsize=13, fontweight="bold")
 
    axes[0].barh(list(y), orig_vals, color="steelblue", alpha=0.8)
    axes[0].axvline(sum(orig_vals)/n, color="red", linestyle="--",
                    linewidth=1, label=f"Mean={sum(orig_vals)/n:.0f}")
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(intents, fontsize=7)
    axes[0].set_title(f"Original ({len(df_original):,} rows)")
    axes[0].set_xlabel("Query Count")
    axes[0].legend(fontsize=8)
 
    axes[1].barh(list(y), sample_vals, color="coral", alpha=0.8)
    axes[1].axvline(sum(sample_vals)/n, color="red", linestyle="--",
                    linewidth=1, label=f"Mean={sum(sample_vals)/n:.0f}")
    axes[1].set_title(f"Sampled ({len(df_sampled):,} rows)")
    axes[1].set_xlabel("Query Count")
    axes[1].legend(fontsize=8)
 
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Sampling plot saved → {plot_file}\n")


# ============================================================
# 7: SPLITTER
# ============================================================
def split_data(df, training_cols, test_size=0.10, val_size=0.10, random_state=42):
    """
    Stratified split into train/val/test.
    Converts final splits to DatasetDict with training_cols only.
    Rare intents (count < 5) are protected → always go to train.

    Args:
        df            (pd.DataFrame): Sampled DataFrame.
        training_cols (list):         Columns to keep in final DatasetDict.
        test_size     (float):        Test split ratio.
        val_size      (float):        Val split ratio.
        random_state  (int):          Random seed.

    Returns:
        DatasetDict: train / val / test splits.
    """
    intent_col    = "intent"
    t_size        = test_size or 0
    v_size        = val_size  or 0
    total_holdout = t_size + v_size

    def _finalize(d):
        return Dataset.from_pandas(
            d[training_cols].reset_index(drop=True)
        )

    if total_holdout == 0:
        return DatasetDict({"train": _finalize(df)})

    # Protect rare intents
    counts      = df[intent_col].value_counts()
    rare        = counts[counts < 5].index
    rare_df     = df[df[intent_col].isin(rare)]
    to_split_df = df[~df[intent_col].isin(rare)]

    if len(rare) > 0:
        print(f"⚠️  {len(rare)} rare intents protected → train only")

    train_df, holdout_df = train_test_split(
        to_split_df,
        test_size    = total_holdout,
        stratify     = to_split_df[intent_col],
        random_state = random_state
    )
    train_df = pd.concat([train_df, rare_df]).sample(frac=1, random_state=random_state)
    results  = {"train": _finalize(train_df)}

    if v_size > 0 and t_size > 0:
        val_df, test_df = train_test_split(
            holdout_df,
            test_size    = 1 - (v_size / total_holdout),
            stratify     = holdout_df[intent_col],
            random_state = random_state
        )
        results["val"]  = _finalize(val_df)
        results["test"] = _finalize(test_df)
    elif v_size > 0:
        results["val"]  = _finalize(holdout_df)
    elif t_size > 0:
        results["test"] = _finalize(holdout_df)

    final = DatasetDict(results)

    for split, ds in final.items():
        print(f"   {split:<10} : {len(ds):,} rows  cols: {ds.column_names}")
    print(f"✅ split_data complete\n")

    return final


# ============================================================
# 8: SAVE & COLLECT DATA FILES
# ============================================================
def save_sampled_data(df, base_name):
    out_file = f"{base_name}.sampled.jsonl"
    df.to_json(out_file, orient="records", lines=True)
    print(f"✅ save_sampled_data: {len(df):,} rows → {out_file}")


def save_splits(dataset_dict, base_name):
    for split, ds in dataset_dict.items():
        out_file = f"{base_name}.{split}.jsonl"
        ds.to_json(out_file)
        print(f"✅ {split:<10} → {out_file}  ({len(ds):,} rows)")
