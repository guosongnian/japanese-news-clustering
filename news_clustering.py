"""日文新闻标题的无监督聚类、簇数选择与代表标题分析。"""
from __future__ import annotations
import argparse, hashlib, json, os, tarfile, urllib.request
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

SEED=42
DATA_URL="https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
DATA_SHA256="b17606ed8c670013a3809100a9e6104701baab62cc019abc262111bd2acf1063"

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def ensure_data(data_dir:Path)->Path:
    data_dir.mkdir(parents=True,exist_ok=True); archive=data_dir/"ldcc-20140209.tar.gz"; extracted=data_dir/"text"
    if not archive.exists(): urllib.request.urlretrieve(DATA_URL,archive)
    if DATA_SHA256!="TO_BE_FILLED" and sha256(archive)!=DATA_SHA256: raise ValueError("Livedoor 数据校验失败。")
    if not extracted.exists():
        with tarfile.open(archive,"r:gz") as tar: tar.extractall(data_dir,filter="data")
    return extracted

def load_articles(root:Path)->pd.DataFrame:
    rows=[]
    for path in sorted(root.glob("*/*.txt")):
        if path.name=="LICENSE.txt": continue
        lines=path.read_text(encoding="utf-8",errors="replace").splitlines()
        if len(lines)>=3:
            rows.append({"category":path.parent.name,"title":lines[2].strip(),"content":" ".join(lines[3:]).strip(),"url":lines[0].strip(),"file":path.name})
    return pd.DataFrame(rows)

def select_k(embedding:np.ndarray,candidates:list[int])->pd.DataFrame:
    rng=np.random.default_rng(SEED); idx=rng.choice(len(embedding),min(2500,len(embedding)),replace=False)
    rows=[]
    for k in candidates:
        model=MiniBatchKMeans(n_clusters=k,n_init=10,batch_size=512,random_state=SEED).fit(embedding)
        rows.append({"k":k,"silhouette":silhouette_score(embedding[idx],model.labels_[idx],metric="cosine")})
    return pd.DataFrame(rows)

def representative_titles(data:pd.DataFrame,embedding:np.ndarray,model:MiniBatchKMeans,n:int=5)->pd.DataFrame:
    rows=[]
    for cluster in range(model.n_clusters):
        idx=np.where(model.labels_==cluster)[0]; center=model.cluster_centers_[cluster]
        similarity=embedding[idx]@center/(np.linalg.norm(embedding[idx],axis=1)*np.linalg.norm(center)+1e-12)
        top=idx[np.argsort(-similarity)[:n]]
        rows.append({"cluster":cluster,"size":len(idx),"representative_titles":" | ".join(data.iloc[top].title.tolist())})
    return pd.DataFrame(rows).sort_values("size",ascending=False)

def run(root:Path,quick:bool=False)->dict:
    sns.set_theme(style="whitegrid"); output=root/"outputs"; output.mkdir(exist_ok=True)
    data=load_articles(ensure_data(root/"data"))
    if quick: data=data.groupby("category",group_keys=False).sample(n=min(150,data.groupby("category").size().min()),random_state=SEED).reset_index(drop=True)
    print(f"文章标题: {len(data)}；真实栏目数（仅用于最终评价）: {data.category.nunique()}")
    representation=Pipeline([
        ("tfidf",TfidfVectorizer(analyzer="char",ngram_range=(2,4),min_df=3,max_df=.95,sublinear_tf=True,max_features=80_000)),
        ("svd",TruncatedSVD(n_components=100,random_state=SEED)),
        ("normalize",Normalizer(copy=False)),
    ])
    # 标题与正文前 1,000 字共同表示主题；栏目标签不参与训练或选簇。
    documents=data.title+" "+data.content.str.slice(0,1000)
    embedding=representation.fit_transform(documents)
    candidates=[6,7,8,9,10,11,12] if not quick else [7,9,11]
    k_table=select_k(embedding,candidates); best_k=int(k_table.loc[k_table.silhouette.idxmax(),"k"])
    model=MiniBatchKMeans(n_clusters=best_k,n_init=30,batch_size=512,random_state=SEED).fit(embedding)
    result={"documents":len(data),"selected_k":best_k,"silhouette":float(k_table.silhouette.max()),"ARI_against_editorial_categories":float(adjusted_rand_score(data.category,model.labels_)),"NMI_against_editorial_categories":float(normalized_mutual_info_score(data.category,model.labels_))}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    summary=representative_titles(data,embedding,model); print(summary.to_string(index=False))
    k_table.to_csv(output/"cluster_count_selection.csv",index=False); summary.to_csv(output/"cluster_summary.csv",index=False)
    data.assign(cluster=model.labels_).to_csv(output/"article_clusters.csv",index=False)
    with (output/"metrics.json").open("w",encoding="utf-8") as f: json.dump(result,f,ensure_ascii=False,indent=2)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5)); sns.lineplot(data=k_table,x="k",y="silhouette",marker="o",ax=axes[0]); axes[0].set_title("Cluster-count selection")
    coords=embedding[:,:2]; sns.scatterplot(x=coords[:,0],y=coords[:,1],hue=model.labels_,palette="tab10",s=18,alpha=.7,legend=False,ax=axes[1]); axes[1].set_title("SVD projection colored by cluster"); fig.tight_layout(); fig.savefig(output/"clustering_diagnostics.png",dpi=160,bbox_inches="tight"); plt.close(fig)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--quick",action="store_true"); a=p.parse_args(); run(Path(__file__).resolve().parent,a.quick)
