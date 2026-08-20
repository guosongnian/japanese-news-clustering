"""日文新闻的无监督聚类、簇数选择与代表标题分析。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

SEED = 42
STABILITY_SEEDS = (42, 123, 456)
DATA_URL = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
DATA_SHA256 = "b17606ed8c670013a3809100a9e6104701baab62cc019abc262111bd2acf1063"

def sha256(path: Path) -> str:
    """计算文件的 SHA-256。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_data(data_dir: Path) -> Path:
    """下载并校验官方 Livedoor News Corpus。"""
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "ldcc-20140209.tar.gz"
    extracted = data_dir / "text"
    if not archive.exists():
        urllib.request.urlretrieve(DATA_URL, archive)
    if sha256(archive) != DATA_SHA256:
        raise ValueError("Livedoor 数据校验失败。")
    if not extracted.exists():
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(data_dir, filter="data")
    return extracted


def load_articles(root: Path) -> pd.DataFrame:
    """读取文章正文；编辑栏目标签只用于最终外部评价。"""
    rows = []
    for path in sorted(root.glob("*/*.txt")):
        if path.name == "LICENSE.txt":
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(lines) >= 3:
            rows.append(
                {
                    "category": path.parent.name,
                    "title": lines[2].strip(),
                    "content": " ".join(lines[3:]).strip(),
                    "url": lines[0].strip(),
                    "file": path.name,
                }
            )
    return pd.DataFrame(rows)


def mean_pairwise_ari(label_sets: list[np.ndarray]) -> float:
    """计算不同初始化所得聚类之间的平均一致性。"""
    scores = [
        adjusted_rand_score(label_sets[i], label_sets[j])
        for i in range(len(label_sets))
        for j in range(i + 1, len(label_sets))
    ]
    return float(np.mean(scores)) if scores else 1.0


def select_k(
    embedding: np.ndarray,
    candidates: list[int],
    seeds: tuple[int, ...] = STABILITY_SEEDS,
) -> pd.DataFrame:
    """同时考虑轮廓系数和不同初始化之间的稳定性。"""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(embedding), min(2500, len(embedding)), replace=False)
    rows = []
    for k in candidates:
        label_sets = []
        silhouettes = []
        for seed in seeds:
            model = MiniBatchKMeans(
                n_clusters=k,
                n_init=10,
                batch_size=512,
                random_state=seed,
            ).fit(embedding)
            label_sets.append(model.labels_)
            silhouettes.append(
                silhouette_score(
                    embedding[idx],
                    model.labels_[idx],
                    metric="cosine",
                )
            )
        silhouette_mean = float(np.mean(silhouettes))
        stability = mean_pairwise_ari(label_sets)
        rows.append(
            {
                "k": k,
                "silhouette_mean": silhouette_mean,
                "silhouette_std": float(np.std(silhouettes)),
                "stability_ari": stability,
                "selection_score": silhouette_mean * stability,
            }
        )
    return pd.DataFrame(rows)


def representative_titles(
    data: pd.DataFrame,
    embedding: np.ndarray,
    model: MiniBatchKMeans,
    n: int = 5,
) -> pd.DataFrame:
    """返回每个簇中与中心最接近的标题。"""
    rows = []
    for cluster in range(model.n_clusters):
        idx = np.where(model.labels_ == cluster)[0]
        center = model.cluster_centers_[cluster]
        similarity = embedding[idx] @ center / (
            np.linalg.norm(embedding[idx], axis=1) * np.linalg.norm(center) + 1e-12
        )
        top = idx[np.argsort(-similarity)[:n]]
        rows.append(
            {
                "cluster": cluster,
                "size": len(idx),
                "representative_titles": " | ".join(
                    data.iloc[top].title.tolist()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("size", ascending=False)

def run(root: Path, quick: bool = False) -> dict[str, object]:
    """运行文本表示、选簇、最终聚类和诊断输出。"""
    sns.set_theme(style="whitegrid")
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    data = load_articles(ensure_data(root / "data"))
    if quick:
        sample_size = min(150, data.groupby("category").size().min())
        data = (
            data.groupby("category", group_keys=False)
            .sample(n=sample_size, random_state=SEED)
            .reset_index(drop=True)
        )
    print(f"文章标题: {len(data)}；真实栏目数（仅用于最终评价）: {data.category.nunique()}")
    representation = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 4),
                    min_df=3,
                    max_df=0.95,
                    sublinear_tf=True,
                    max_features=80_000,
                ),
            ),
            ("svd", TruncatedSVD(n_components=100, random_state=SEED)),
            ("normalize", Normalizer(copy=False)),
        ]
    )
    # 标题与正文前 1,000 字共同表示主题；栏目标签不参与训练或选簇。
    documents = data.title + " " + data.content.str.slice(0, 1000)
    embedding = representation.fit_transform(documents)
    candidates = list(range(6, 21)) if not quick else [7, 9, 11]
    seeds = STABILITY_SEEDS if not quick else (SEED,)
    k_table = select_k(embedding, candidates, seeds=seeds)
    best_row = k_table.sort_values(
        ["selection_score", "silhouette_mean"], ascending=False
    ).iloc[0]
    best_k = int(best_row["k"])
    model = MiniBatchKMeans(
        n_clusters=best_k,
        n_init=30,
        batch_size=512,
        random_state=SEED,
    ).fit(embedding)
    result = {
        "documents": len(data),
        "selected_k": best_k,
        "silhouette_mean": float(best_row["silhouette_mean"]),
        "silhouette_std": float(best_row["silhouette_std"]),
        "stability_ari": float(best_row["stability_ari"]),
        "selection_score": float(best_row["selection_score"]),
        "ARI_against_editorial_categories": float(
            adjusted_rand_score(data.category, model.labels_)
        ),
        "NMI_against_editorial_categories": float(
            normalized_mutual_info_score(data.category, model.labels_)
        ),
    }
    print(json.dumps(result,ensure_ascii=False,indent=2))
    summary = representative_titles(data, embedding, model)
    print(summary.to_string(index=False))
    k_table.to_csv(output / "cluster_count_selection.csv", index=False)
    summary.to_csv(output / "cluster_summary.csv", index=False)
    data.assign(cluster=model.labels_).to_csv(
        output / "article_clusters.csv", index=False
    )
    with (output / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].errorbar(
        k_table["k"],
        k_table["silhouette_mean"],
        yerr=k_table["silhouette_std"],
        marker="o",
        capsize=3,
    )
    axes[0].set(xlabel="k", ylabel="Mean silhouette", title="Cluster-count selection")
    coords = embedding[:, :2]
    sns.scatterplot(
        x=coords[:, 0],
        y=coords[:, 1],
        hue=model.labels_,
        palette="tab10",
        s=18,
        alpha=0.7,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("SVD projection colored by cluster")
    fig.tight_layout()
    fig.savefig(
        output / "clustering_diagnostics.png", dpi=160, bbox_inches="tight"
    )
    plt.close(fig)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    run(Path(__file__).resolve().parent, arguments.quick)
