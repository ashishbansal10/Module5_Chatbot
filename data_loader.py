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
    "save_splits",
    "collect_generated_files"
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
def analyze_data(df, dataset_name):
    """
    Prints basic dataset statistics and plots intent distribution.
    Saves plot to <basename>.analysis.png and displays inline.

    Args:
        df           (pd.DataFrame): Cleaned DataFrame.
        dataset_name (str):          Dataset name for file naming.
    """
    base       = get_file_basename(dataset_name)
    plot_file  = f"{base}.analysis.png"
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
    for intent, count in intent_counts_display.items():
        print(f"   {intent:<40} : {count:>5,}")

    # Intent distribution
    intent_counts = df[intent_col].value_counts().sort_values()
    n_intents     = len(intent_counts)
    mean_count    = intent_counts.mean()
    std_count     = intent_counts.std()

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, n_intents * 0.25)))
    fig.suptitle(f"Intent Distribution — {base.upper()}", fontsize=14)

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
    label         = "✅ Balanced" if balance_ratio > 0.5 else "⚠️ Imbalanced"
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


def semantic_analysis(df, dataset_name, device="cuda", target_cluster_size=50, min_clusters=2, max_clusters=20):
    """
    Encodes queries, L2-normalizes, and clusters within each intent.
    Cluster count k scales dynamically with intent size.
    Caches normalized embeddings to disk.

    Args:
        df                  (pd.DataFrame): DataFrame with prompt, response, intent cols.
        dataset_name        (str):          Dataset name for cache file naming.
        device              (str):          'cuda' or 'cpu'.
        target_cluster_size (int):          Target rows per cluster (controls k).
        min_clusters        (int):          Minimum clusters per intent.
        max_clusters        (int):          Maximum clusters per intent.

    Returns:
        df         (pd.DataFrame): DataFrame with semantic_cluster column added.
        embeddings (np.ndarray):   L2-normalized embeddings.
    """

    base       = get_file_basename(dataset_name)
    data_hash  = _compute_data_hash(df, dataset_name)
    cache_file = f"{base}.{data_hash}.embeddings.npy"

    assert device in ("cuda", "cpu"), \
        f"❌ device must be 'cuda' or 'cpu', got: '{device}'"

    # Cleanup stale cache
    for f in os.listdir("."):
        if f.startswith(base) and f.endswith(".embeddings.npy") and f != cache_file:
            os.remove(f)
            print(f"🗑️  Removed stale cache: {f}")

    q_col      = "prompt"
    intent_col = "intent"

    # Encode or load normalized cache
    if os.path.exists(cache_file):
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
def audit_report(df, dataset_name, save_json=True):
    """
    Computes and prints semantic audit report.
    Nature/cluster stats included only if semantic_cluster column present.

    Args:
        df           (pd.DataFrame): DataFrame with prompt, response, intent cols.
        dataset_name (str):          Dataset name for file naming.
        save_json    (bool):         If True, saves report to disk.
    """
    base         = get_file_basename(dataset_name)
    report_file  = f"{base}.report.json"
    intent_col   = "intent"
    q_col        = "prompt"
    has_clusters = "semantic_cluster" in df.columns

    df            = df.copy()
    intent_counts = df[intent_col].value_counts()

    print(f"\n{'='*65}")
    print(f"📊 AUDIT REPORT: {base.upper()}")
    print(f"{'='*65}")
    print(f"   Records        : {len(df):,}")
    print(f"   Intents        : {df[intent_col].nunique()}")
    print(f"   Avg Q/Intent   : {intent_counts.mean():.1f}  "
          f"(min={intent_counts.min()}, max={intent_counts.max()})")
    balance_ratio = intent_counts.min() / intent_counts.max()
    balance_label = "✅ Balanced" if balance_ratio > 0.5 else "⚠️  Imbalanced"
    print(f"   Balance Ratio  : {balance_ratio:.2f}  {balance_label}")

    # Intent table
    print(f"\n   {'Intent':<40} | {'Queries':>7}")
    print(f"   {'-'*52}")
    for intent, count in intent_counts.items():
        print(f"   {intent:<40} | {count:>7,}")

    # Nature stats — only if semantic_cluster present
    if has_clusters:
        df["_nature"] = df[intent_col].astype(str) + "_n" + df["semantic_cluster"].astype(str)
        diversity     = df.groupby(intent_col)["_nature"].nunique()
        total_natures = df["_nature"].nunique()
        nature_sizes  = df.groupby("_nature").size()

        print(f"\n   Natures        : {total_natures}")
        print(f"   Avg Q/Nature   : {len(df)/total_natures:.1f}  "
              f"(min={nature_sizes.min()}, max={nature_sizes.max()})")
        print(f"   Avg Natures/Intent : {diversity.mean():.1f}")

        print(f"\n   {'Intent':<40} | {'Natures':>7} | {'Queries':>7}")
        print(f"   {'-'*60}")
        for intent in diversity.sort_values(ascending=False).head(10).index:
            print(f"   {intent:<40} | {diversity[intent]:>7} | {intent_counts[intent]:>7,}")

        print(f"\n   Top Natures (by density):")
        for nature, count in df["_nature"].value_counts().head(5).items():
            examples = df[df["_nature"] == nature][q_col].head(2).tolist()
            print(f"\n   [{nature}] — {count} queries")
            for ex in examples:
                print(f"     > {ex}")
    else:
        print(f"\n   ⚠️  semantic_cluster not found — run semantic_analysis() for nature stats")

    print(f"\n{'='*65}\n")

    if save_json:
        report = {
            "meta"  : {"source": dataset_name, "base_name": base},
            "stats" : {
                "total_rows"      : int(len(df)),
                "unique_intents"  : int(df[intent_col].nunique()),
                "intent_densities": intent_counts.to_dict(),
                "balance_ratio"   : round(balance_ratio, 3),
            }
        }
        if has_clusters:
            report["stats"].update({
                "total_unique_natures"  : int(total_natures),
                "avg_natures_per_intent": round(float(diversity.mean()), 2),
            })
        with open(report_file, "w") as f:
            json.dump(report, f, indent=4)
        print(f"✅ audit_report saved: {report_file}\n")


def visual_atlas(df, embeddings, dataset_name):
    """
    Generates 3 actionable plots to inform sampling decisions.
    Plot 3 (nature distribution) shown only if semantic_cluster present.
    Saves to <basename>.atlas.png and displays inline.

    Args:
        df           (pd.DataFrame): DataFrame with prompt, response, intent cols.
        embeddings   (np.ndarray):   L2-normalized query embeddings.
        dataset_name (str):          Dataset name for file naming.
    """
    base          = get_file_basename(dataset_name)
    output_file   = f"{base}.atlas.png"
    intent_col    = "intent"
    has_clusters  = "semantic_cluster" in df.columns
    n_plots       = 3 if has_clusters else 2
    intent_counts = df[intent_col].value_counts().sort_values()

    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, max(6, len(intent_counts) * 0.22)))
    fig.suptitle(f"Semantic Atlas: {base.upper()}", fontsize=14, fontweight="bold")

    # ----------------------------------------------------------------
    # Plot 1: Intent volume bar chart
    # ----------------------------------------------------------------
    ax1 = axes[0]
    mean_val = intent_counts.mean()
    ax1.barh(intent_counts.index, intent_counts.values, color="steelblue", alpha=0.8)
    ax1.axvline(mean_val, color="red", linestyle="--", linewidth=1, label=f"Mean={mean_val:.0f}")
    ax1.set_title("1. Intent Volume")
    ax1.set_xlabel("Query Count")
    ax1.legend(fontsize=8)
    balance_ratio = intent_counts.min() / intent_counts.max()
    label         = "✅ Balanced" if balance_ratio > 0.5 else "⚠️ Imbalanced"
    ax1.annotate(f"{label}  (min/max={balance_ratio:.2f})",
                 xy=(0.5, -0.04), xycoords="axes fraction",
                 ha="center", fontsize=9)

    # ----------------------------------------------------------------
    # Plot 2: Intent similarity heatmap
    # ----------------------------------------------------------------
    ax2   = axes[1]
    top10 = df[intent_col].value_counts().nlargest(10).index
    means = [embeddings[df[df[intent_col] == n].index].mean(axis=0) for n in top10]
    sim   = cosine_similarity(means)
    sns.heatmap(sim, xticklabels=top10, yticklabels=top10,
                annot=True, fmt=".2f", cmap="magma",
                ax=ax2, annot_kws={"size": 7})
    ax2.set_title("2. Intent Similarity (Top 10)")
    ax2.tick_params(axis="x", rotation=45, labelsize=7)
    ax2.tick_params(axis="y", rotation=0,  labelsize=7)

    # ----------------------------------------------------------------
    # Plot 3: Nature size distribution (box plot per intent)
    # ----------------------------------------------------------------
    if has_clusters:
        ax3          = axes[2]
        nature_sizes = df.groupby([intent_col, "semantic_cluster"]).size().reset_index(name="count")
        top15        = df[intent_col].value_counts().nlargest(15).index
        plot_data    = nature_sizes[nature_sizes[intent_col].isin(top15)]
        sns.boxplot(data=plot_data, y=intent_col, x="count",
                    order=top15, ax=ax3, palette="Set2", orient="h")
        ax3.set_title("3. Nature Size Distribution (Top 15 Intents)")
        ax3.set_xlabel("Queries per Nature")
        ax3.set_ylabel("")
        ax3.tick_params(axis="y", labelsize=7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ visual_atlas saved: {output_file}\n")


# ============================================================
# 6: SAMPLER
# ============================================================
def sample_data(df, dataset_name, sample_size=1000, target_n_per_nature=2, random_state=42):
    """
    Samples dataset using squeeze logic — importance-weighted per nature.
    Prints sampling summary inline. Saves <basename>.sampled_report.json.
    Requires semantic_cluster column.

    Args:
        df                  (pd.DataFrame): DataFrame with semantic_cluster.
        dataset_name        (str):          Dataset name for file naming.
        sample_size         (int):          Target number of rows.
        target_n_per_nature (int):          Target samples per nature cluster.
        random_state        (int):          Random seed.

    Returns:
        pd.DataFrame: Sampled DataFrame.
    """
    assert "semantic_cluster" in df.columns, \
        "❌ semantic_cluster missing — run semantic_analysis() first"

    base        = get_file_basename(dataset_name)
    report_file = f"{base}.sampled_report.json"
    intent_col  = "intent"
    q_col       = "prompt"

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

    print(f"\n   Top 5 intents in sample:")
    for intent, count in intent_counts.head(5).items():
        print(f"   {intent:<40} : {count:>4} ({count/len(selected)*100:.1f}%)")

    summary = {
        "compression": {
            "original_rows" : int(len(df)),
            "sampled_rows"  : int(len(selected)),
            "squeeze_factor": round(len(df) / len(selected), 1),
            "reduction_pct" : round(100 - len(selected) / len(df) * 100, 2),
        },
        "retention": {
            "original_natures": int(total_natures),
            "sampled_natures" : int(n_natures_sampled),
            "coverage_rate"   : round(n_natures_sampled / total_natures * 100, 2),
        },
        "distribution": [
            {"intent": k, "count": int(v), "pct": round(v / len(selected) * 100, 2)}
            for k, v in intent_counts.head(10).items()
        ]
    }
    with open(report_file, "w") as f:
        json.dump(summary, f, indent=4)

    selected = selected.drop(columns=["_nature"])
    print(f"\n✅ Sampling report saved: {report_file}\n")
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

def sampling_report(df_original, df_sampled, dataset_name):
    """
    Generates sampling quality report — JSON + intent distribution plot.
    Compares original vs sampled intent distribution side by side.
    Saves <basename>.sampling_report.json and <basename>.sampling.png.

    Args:
        df_original  (pd.DataFrame): Full original DataFrame.
        df_sampled   (pd.DataFrame): Sampled DataFrame.
        dataset_name (str):          Dataset name for file naming.
    """
    base        = get_file_basename(dataset_name)
    report_file = f"{base}.sampling_report.json"
    plot_file   = f"{base}.sampling.png"
    intent_col  = "intent"
    has_clusters= "semantic_cluster" in df_sampled.columns

    orig_counts    = df_original[intent_col].value_counts().sort_index()
    sampled_counts = df_sampled[intent_col].value_counts().sort_index()

    n_natures_orig    = df_original.groupby(
        [intent_col, "semantic_cluster"]).ngroups if has_clusters and \
        "semantic_cluster" in df_original.columns else None
    n_natures_sampled = df_sampled.groupby(
        [intent_col, "semantic_cluster"]).ngroups if has_clusters else None

    report = {
        "compression": {
            "original_rows" : int(len(df_original)),
            "sampled_rows"  : int(len(df_sampled)),
            "squeeze_factor": round(len(df_original) / len(df_sampled), 1),
            "reduction_pct" : round(100 - len(df_sampled) / len(df_original) * 100, 2),
        },
        "intent_coverage": {
            "original_intents" : int(df_original[intent_col].nunique()),
            "sampled_intents"  : int(df_sampled[intent_col].nunique()),
            "missing_intents"  : list(
                set(df_original[intent_col].unique()) -
                set(df_sampled[intent_col].unique())
            ),
        },
        "distribution": [
            {
                "intent"        : intent,
                "original_count": int(orig_counts.get(intent, 0)),
                "sampled_count" : int(sampled_counts.get(intent, 0)),
                "retention_pct" : round(
                    sampled_counts.get(intent, 0) /
                    orig_counts.get(intent, 1) * 100, 1
                ),
            }
            for intent in orig_counts.index
        ],
    }
    if n_natures_orig and n_natures_sampled:
        report["nature_coverage"] = {
            "original_natures": int(n_natures_orig),
            "sampled_natures" : int(n_natures_sampled),
            "coverage_rate"   : round(n_natures_sampled / n_natures_orig * 100, 2),
        }

    with open(report_file, "w") as f:
        json.dump(report, f, indent=4)

    # ----------------------------------------------------------------
    # Plot — original vs sampled intent distribution
    # ----------------------------------------------------------------
    intents     = orig_counts.index.tolist()
    orig_vals   = [orig_counts.get(i, 0)    for i in intents]
    sample_vals = [sampled_counts.get(i, 0) for i in intents]
    n           = len(intents)
    y           = range(n)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, n * 0.25)), sharey=True)
    fig.suptitle(f"Sampling Report: {base.upper()}", fontsize=13, fontweight="bold")

    # Original
    axes[0].barh(list(y), orig_vals, color="steelblue", alpha=0.8)
    axes[0].axvline(sum(orig_vals)/n, color="red", linestyle="--",
                    linewidth=1, label=f"Mean={sum(orig_vals)/n:.0f}")
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(intents, fontsize=7)
    axes[0].set_title(f"Original ({len(df_original):,} rows)")
    axes[0].set_xlabel("Query Count")
    axes[0].legend(fontsize=8)

    # Sampled
    axes[1].barh(list(y), sample_vals, color="coral", alpha=0.8)
    axes[1].axvline(sum(sample_vals)/n, color="red", linestyle="--",
                    linewidth=1, label=f"Mean={sum(sample_vals)/n:.0f}")
    axes[1].set_title(f"Sampled ({len(df_sampled):,} rows)")
    axes[1].set_xlabel("Query Count")
    axes[1].legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ sampling_report: {report_file} + {plot_file}\n")


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
def save_sampled_data(df, dataset_name):
    base     = get_file_basename(dataset_name)
    out_file = f"{base}.sampled.jsonl"
    df.to_json(out_file, orient="records", lines=True)
    print(f"✅ save_sampled_data: {len(df):,} rows → {out_file}")


def save_splits(dataset_dict, dataset_name):
    base = get_file_basename(dataset_name)
    for split, ds in dataset_dict.items():
        out_file = f"{base}.{split}.jsonl"
        ds.to_json(out_file)
        print(f"✅ {split:<10} → {out_file}  ({len(ds):,} rows)")


def collect_generated_files(dataset_name):
    base = get_file_basename(dataset_name)
    generated_files = sorted([
        f for f in os.listdir(".")
        if os.path.isfile(f) and f.startswith(base)
    ])
    return generated_files
