# Project: Chat-based multimodal RAG system

An end-to-end how-to assistant: it retrieves relevant WikiHow **articles** and HowTo100M
**video clips** for a natural-language question, grounds a chat model on them, and shows
everything in a transparent UI.

```
01_Preprocessing.ipynb   WikiHow -> SQLite + MiniLM FAISS indices;  HowTo100M -> CLIP FAISS clip index
02_Chatbot_Setup.ipynb   TinyLlama prompt assembly + reply generation
03_RAG.ipynb             ArticleRetriever + VideoRetriever + end-to-end RAG pipeline
build_data.py            calls 01's functions to build the artifacts (see below)
11_User_Interface/       Gradio UI (Assignment 11) that connects to the above
```

The pipeline logic lives entirely in the notebooks. `load_notebook` (in
`11_User_Interface/rag_backend.py`) imports a notebook's definitions (its `def`/`class`/constants,
skipping the demo cells), so both `build_data.py` and the UI **call** the notebook code rather
than duplicating it:

- `build_data.py` calls `01`'s `encode_texts`, `build_faiss_index`, `load_videos`, `build_clip_index`, …
- `11_User_Interface/rag_backend.py` instantiates `03`'s `ArticleRetriever` / `VideoRetriever` and
  calls its `build_prompt` / `generate_reply` / `parse_steps`, with `SYSTEM_PROMPT` and the token
  budget (`MAX_CONTEXT_TOKENS`) read from `03` / `02`.

## Running the whole project

Everything is inference-only (frozen models). Models download automatically from Hugging
Face on first use (MiniLM ~90 MB, CLIP ~600 MB, TinyLlama ~2.2 GB).

```bash
pip install -r requirements.txt

# 1) datasets -> Compressed_Data/  (not committed; see Assignment 8)
#    WikiHow (cleaned):  https://cloud.uni-konstanz.de/index.php/s/Y35jG9SDLGXA5y2   -> wikihow-cleaned.zip
#    HowTo100M subset:   https://cloud.uni-konstanz.de/index.php/s/7yqcgzYQ4TfnBF8   -> howto100m_bundle.zip

# 2) build the artifacts (wikihow.db + FAISS text/CLIP indices + clip_metadata.csv + clips)
python build_data.py                 # ~10 min: 20k-article sample + a small video batch
#   AML_MAX_ARTICLES=0 python build_data.py   # embed the FULL 214k WikiHow set (~1 h)

# 3) launch the UI
cd 11_User_Interface && python app.py           # http://127.0.0.1:7860
```

`build_data.py` runs `01_Preprocessing.ipynb`'s own functions (via `load_notebook`) and writes,
at the paths `03_RAG.ipynb` and the UI read from:

```
wikihow.db
Data/Embeddings/{title_index,combined_index,clip_index}.faiss
Data/HowTo100M/clip_metadata.csv
Data/HowTo100M/videos/<id>.mp4
```

The UI (`11_User_Interface/`) loads these automatically and runs the **real** MiniLM + CLIP +
TinyLlama pipeline. Datasets, indices, videos and model weights are gitignored and never
committed — they are downloaded / rebuilt by the steps above.

> Notes: `build_data.py` embeds a 20k-article random sample by default so it finishes in
> minutes; the code supports the full 214k set via `AML_MAX_ARTICLES=0`. The HowTo100M bundle
> is 10 GB, so `Compressed_Data/howto100m_bundle.zip` here holds a small real batch of clips
> (the assignment allows a 2–5 video batch); point it at the full bundle to process all 342.

## Interface

The two images are Figma mockups of the interface, once with the articles and once with the
videos tab selected for the Top-K retrieval column.

The system inspection panel includes:
- Context size (token budget)
    - If we exceed the max context size, the chat model breaks down so we need this in order to balance the context input. Note that in the mockup only one article and one video are portrayed. This is purely for demonstration purposes so it does not dominate the image. In the full application, this would correspond to the retrieved top-k results
- Injected context
    - Lets us know which additional context was added into the model both for reproducibility/causality and a factor for context size
- Retrieval settings
    - Provides basic data for what is used in the embedding retrieval. These values are not strictly necessary since we as developers should be aware of them, but informative in a general sense for new users or if we ourselves explore the project again at some later point in time. These parameters have a masssive effect on output quality
    - This lets us identify potential bottlenecks in generation time
