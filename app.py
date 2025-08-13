# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image
import io, numpy as np, torch

from pm_lib import (
    read_products, load_clip, embed_images, embed_texts,
    build_index, search_topk, log_jsonl
)

PRODUCTS_PATH = "data/products.jsonl"
LOG_PATH = "logs.jsonl"

app = FastAPI(title="Product Matching MVP", version="0.1")

# -------- Startup: load metadata, model, embeddings, index --------
products = read_products(PRODUCTS_PATH)
if not products:
    raise RuntimeError("No products found in data/products.jsonl")

model, preprocess = load_clip()
catalog_paths = [p["image_path"] for p in products]
catalog_embs = embed_images(model, preprocess, catalog_paths)   # [N,512], normalized
index_mat = build_index(catalog_embs)                          # just the matrix

class TextQuery(BaseModel):
    text: str

@app.get("/health")
def health():
    return {
        "status": "ok",
        "num_products": len(products),
        "embedding_dim": int(index_mat.shape[1]),
    }

@app.post("/match-image")
async def match_image(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # embed query image (CPU)
    with torch.no_grad():
        x = preprocess(img).unsqueeze(0)   # [1,3,224,224]
        e = model.encode_image(x)
        e = e / e.norm(dim=-1, keepdim=True)
    q = e.numpy().astype("float32")        # [1,512]

    scores, idxs = search_topk(index_mat, q, k=3)
    top = products[int(idxs[0])]
    log_jsonl(LOG_PATH, {
        "type": "image",
        "filename": file.filename,
        "top1_id": top["product_id"],
        "score": float(scores[0]),
    })
    return {
        "query": {"filename": file.filename},
        "top1": {"score": float(scores[0]), "product": top},
        "topk": [
            {"rank": i+1, "score": float(scores[i]), "product": products[int(idxs[i])]}
            for i in range(len(idxs))
        ],
    }

@app.post("/match-text")
def match_text(payload: TextQuery):
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    t = embed_texts(model, [text])         # [1,512], normalized
    scores, idxs = search_topk(index_mat, t, k=3)
    top = products[int(idxs[0])]
    log_jsonl(LOG_PATH, {
        "type": "text",
        "text": text,
        "top1_id": top["product_id"],
        "score": float(scores[0]),
    })
    return {
        "query": {"text": text},
        "top1": {"score": float(scores[0]), "product": top},
        "topk": [
            {"rank": i+1, "score": float(scores[i]), "product": products[int(idxs[i])]}
            for i in range(len(idxs))
        ],
    }
