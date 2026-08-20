# 日文新闻主题聚类

本项目对 7,367 篇 Livedoor 新闻的标题和正文片段进行无监督主题聚类。流程不使用栏目标签训练模型：首先用日文字符 TF-IDF 表示文本，再通过 TruncatedSVD 压缩维度，根据轮廓系数选择簇数，最后输出每个簇最有代表性的新闻标题。

## 结果

稳定性修正后的选簇分数在候选范围内选择 `k=9`：

| 指标 | 结果 |
|---|---:|
| 平均 Silhouette | **0.162** |
| 初始化稳定性 ARI | **0.633** |
| 选簇分数 | **0.102** |
| ARI（与编辑栏目对照） | **0.488** |
| NMI（与编辑栏目对照） | **0.647** |

真实栏目只用于最终外部评价，不参与特征学习、簇数选择或 K-Means 训练。聚类能够识别体育、电影娱乐、Android、科技、女性生活等主题，但也会产生跨栏目的主题簇；这符合“内容主题”和“编辑栏目”并不完全相同的现实。

## 方法

- 从官方地址下载 Livedoor News Corpus，并执行 SHA-256 校验。
- 使用标题与正文前 1,000 字，避免长文对内存和运行时间造成过大压力。
- 采用 2–4 字符 n-gram，无需依赖容易失效的日文分词器或大型语言模型。
- TF-IDF 后通过 100 维 TruncatedSVD 获得稠密语义表示。
- 在随机抽取的 2,500 篇文章上计算余弦轮廓系数，比较 `k=6...20`。
- 对每个候选簇数使用三个随机种子，并以不同运行之间的 ARI 衡量聚类稳定性。
- 使用 `平均轮廓系数 × 稳定性 ARI` 作为选簇分数，避免单纯追逐更大的簇数。
- 使用 MiniBatchKMeans 训练最终模型。
- 根据文档与簇中心的余弦相似度，为每个簇选取五条代表标题。
- ARI 和 NMI 仅作为聚类完成后的外部参考。

## 运行

建议使用 Python 3.11 或更高版本；GitHub Actions 使用 Python 3.12 自动检查依赖、模块导入和语法。

```bash
git clone https://github.com/guosongnian/japanese-news-clustering.git
cd japanese-news-clustering
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python news_clustering.py
```

快速环境检查：`python news_clustering.py --quick`。

完整分析见 [`notebooks/japanese_news_clustering.ipynb`](notebooks/japanese_news_clustering.ipynb)，数据许可与署名要求见 [`NOTICE.md`](NOTICE.md)。

## 局限

- 轮廓系数偏低，说明新闻主题之间存在重叠；簇数选择应结合稳定性和代表标题解释，而不能只看单一指标。
- SVD 二维投影只用于辅助观察，不能完整表达 100 维空间中的距离。
- 字符 n-gram 易于复现，但不具备大型语言模型的上下文语义能力。
- 新闻语料来自特定时期，不能代表当前新闻分布。
