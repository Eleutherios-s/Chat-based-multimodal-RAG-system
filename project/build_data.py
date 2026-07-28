"""
build_data.py -- run the project's preprocessing by CALLING 01_Preprocessing.ipynb.

This does not re-implement the notebook: `load_notebook` (from rag_backend) imports 01's own
functions (run_sql, encode_texts, build_faiss_index, load_videos, build_clip_index, ...) and
this script just orchestrates them over the datasets in Compressed_Data/ and writes the
artifacts the chatbot (02), RAG (03) and the UI read:

    wikihow.db
    Data/Embeddings/{title_index,combined_index,clip_index}.faiss
    Data/HowTo100M/clip_metadata.csv  +  Data/HowTo100M/videos/<id>.mp4

Run:  python build_data.py            (AML_MAX_ARTICLES=0 -> embed the full 214k WikiHow set)
"""

import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "11_User_Interface"))   # the loader lives with the UI backend
from rag_backend import load_notebook
DATA, EMB = ROOT / "Data", ROOT / "Data/Embeddings"
BUNDLE, VIDEO_DIR = ROOT / "Data/HowTo100M", ROOT / "Data/HowTo100M/videos"
for d in (EMB, VIDEO_DIR):
    d.mkdir(parents=True, exist_ok=True)

# full WikiHow is 214k; embedding all of it takes ~1h on CPU/MPS, so default to a diverse
# random sample (the assignment allows a batch). AML_MAX_ARTICLES=0 -> full dataset.
MAX_ARTICLES = int(os.environ.get("AML_MAX_ARTICLES", "20000")) or None
DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# import notebook 01's functions (definitions only -- no cells are executed for real)
nb = load_notebook(str(ROOT / "01_Preprocessing.ipynb"), namespace={"device": DEVICE})
DB = str(ROOT / "wikihow.db")


def build_text():
    if Path(DB).exists() and (EMB / "title_index.faiss").exists() and (EMB / "combined_index.faiss").exists():
        print("[build] text artifacts exist -> skip (delete them to rebuild)"); return
    with zipfile.ZipFile(ROOT / "Compressed_Data/wikihow-cleaned.zip") as z:
        z.extractall(DATA)
    df = pd.read_csv(DATA / "wikihow-cleaned/wikihow-cleaned.csv").dropna(subset=["title"])
    if MAX_ARTICLES and len(df) > MAX_ARTICLES:
        df = df.sample(n=MAX_ARTICLES, random_state=42)
    df = df.reset_index(drop=True)
    df["summary"] = df["summary"].fillna("")
    print(f"[build] {len(df)} WikiHow articles")

    # SQLite db -- via the notebook's run_sql (id == row order == FAISS position)
    if Path(DB).exists():
        Path(DB).unlink()
    nb.run_sql(DB, "CREATE TABLE wikihow_articles (id int PRIMARY KEY, summary varchar, "
                   "title varchar, text varchar)", commit=True)
    nb.run_sql(DB, "INSERT INTO wikihow_articles VALUES (?, ?, ?, ?)",
               list(df[["summary", "title", "text"]].itertuples(name=None)), many=True, commit=True)

    # embeddings + FAISS -- via the notebook's encode_texts / build_faiss_index
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(TEXT_MODEL, device=DEVICE)
    title_emb = nb.encode_texts(model, df["title"].tolist(), DEVICE, batch_size=512)
    comb_emb = nb.encode_texts(model, (df["title"] + ". " + df["summary"]).tolist(), DEVICE, batch_size=512)
    faiss.write_index(nb.build_faiss_index(np.asarray(title_emb, dtype="float32")), str(EMB / "title_index.faiss"))
    faiss.write_index(nb.build_faiss_index(np.asarray(comb_emb, dtype="float32")), str(EMB / "combined_index.faiss"))
    print("[build] wikihow.db + title_index.faiss + combined_index.faiss")


def build_video():
    if not (BUNDLE / "available_videos.csv").exists():
        with zipfile.ZipFile(ROOT / "Compressed_Data/howto100m_bundle.zip") as z:
            z.extractall(DATA)
    # everything below is notebook 01's code: load_clip_model / load_videos / build_clip_index
    clip_model, clip_proc = nb.load_clip_model(nb.CLIP_MODEL_NAME, DEVICE)
    videos = nb.load_videos(str(BUNDLE), str(VIDEO_DIR), limit=None)
    print(f"[build] {len(videos)} videos on disk")
    index, metadata = nb.build_clip_index(videos, clip_model, clip_proc, DEVICE)
    faiss.write_index(index, str(EMB / "clip_index.faiss"))
    metadata.to_csv(BUNDLE / "clip_metadata.csv", index=False)
    print(f"[build] {index.ntotal} clips -> clip_index.faiss + clip_metadata.csv")


if __name__ == "__main__":
    print(f"[build] device={DEVICE}; calling 01_Preprocessing.ipynb functions")
    build_text()
    build_video()
    print("[build] DONE. Run the UI:  cd 11_User_Interface && python app.py")
