import json, time
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
import torch, clip

DEVICE = "cpu"  # CPU-only for a light setup

def read_products(jsonl_path: str) -> List[Dict]:
    items = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def load_clip():
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()
    return model, preprocess

def embed_images(model, preprocess, image_paths: List[str]) -> np.ndarray:
    """
    Returns: np.ndarray of shape [N, 512], L2-normalized
    """
    vecs = []
    with torch.no_grad():
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            x = preprocess(img).unsqueeze(0)  # CPU
            e = model.encode_image(x)
            e = e / e.norm(dim=-1, keepdim=True)  # normalize for cosine
            vecs.append(e.numpy())
    return np.vstack(vecs).astype("float32")

def embed_texts(model, texts: List[str]) -> np.ndarray:
    """
    Returns: np.ndarray of shape [B, 512], L2-normalized
    """
    with torch.no_grad():
        tok = clip.tokenize(texts)
        e = model.encode_text(tok)
        e = e / e.norm(dim=-1, keepdim=True)
    return e.numpy().astype("float32")

def build_index(embs: np.ndarray) -> np.ndarray:
    """
    For small catalogs, the 'index' is just the normalized embedding matrix itself.
    """
    return embs  # [N, 512]

def search_topk(index_mat: np.ndarray, q: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cosine similarity on normalized vectors = dot product.
    index_mat: [N, 512]
    q: [512] or [1, 512]
    Returns (scores, indices) for top-k (first query).
    """
    if q.ndim == 1:
        q = q[None, :]  # [1, 512]
    sims = index_mat @ q[0].astype("float32")  # [N]
    topk_idx = np.argsort(-sims)[:k]
    topk_scores = sims[topk_idx]
    return topk_scores, topk_idx

def log_jsonl(path: str, obj: Dict):
    obj = dict(obj); obj["ts"] = int(time.time() * 1000)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
