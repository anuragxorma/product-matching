# **Product Matching Prototype (MVP)**

## **Project Overview**
This project is a minimal product matching prototype that, given an image or a short text description, returns the closest matching product from a small catalog.  
It demonstrates the core logic of an image/text search system using embeddings and nearest neighbor search, without the heavy infrastructure that would be used in production.

---

## **Goal**
Build a tiny product matching system:  
> **Given an image or a text query, return the closest product from a small catalog using embeddings and similarity search.**

---

## **What Was Built (MVP)**

### **1. Small Catalog + Metadata**
- ~17 product images (`data/images/`)  
  *(examples: shoes, mugs, laptop, sunglasses, chairs, etc.)*
- `products.jsonl` with product metadata:
    ```json
    {
  "product_id": "shoes1",
  "name": "Brown Leather Boots",
  "category": "shoes",
  "price": 79.99,
  "image_path": "data/images/shoes1.jpg",
  "description": "brown leather boots"
    }

### **2. Embeddings with CLIP (CPU)**
- Pre-trained CLIP (ViT-B/32) model from OpenAI.
- Generates 512-dimensional embeddings for:
  - Images (product photos).
  - Text (queries).
- L2-normalized embeddings for cosine similarity.

---

### **3. Vector Search (NumPy Cosine Similarity)**
- All product embeddings stored in memory.
- For each query:
  - Embed → Normalize → Cosine similarity search → Return top-1 match.

---

### **4. Simple API (FastAPI)**
- `/health` → Quick check.
- `/match-image` (multipart file) → Returns top-1 product + metadata.
- `/match-text` (JSON body) → Returns top-1 product + metadata.

---

### **5. Lightweight Logging**
- Every query logged to `logs.jsonl`:
    ```json
    {"type":"image","query":"data/images/shoes1.jpg","top1_id":"shoes1","score":0.9999,"timestamp":"2025-08-13T10:23:45"}

## **6. Dockerized App**
Single container with:
- API server.
- Small dataset.
- All dependencies pre-installed.
- Run with one command.

**Success Criteria**
- Runs locally and in Docker.
- Upload image / send text → get top-1 match + log entry.
- Reproducible setup for reviewers.

---

## **What Was NOT Implemented (Intentionally) & Why**
| Feature Skipped | Why Skipped | How Addressed |
|-----------------|-------------|---------------|
| **NVIDIA Triton Serving** | Heavy container orchestration, unnecessary for small CPU demo | Added `DESIGN.md` section on how to serve with Triton in production |
| **TensorRT Quantization (.plan)** | Requires GPU & calibration dataset | Provided ONNX export + `trtexec --fp16` instructions in `DESIGN.md` |
| **External Vector DB (Qdrant, Pinecone)** | Overkill for 17 items | In-memory NumPy now; migration path in `DESIGN.md` |
| **MongoDB for Metadata/Logs** | Adds infra complexity | Used `products.jsonl` and `logs.jsonl` instead; Mongo schema documented in `DESIGN.md` |
| **Batching, Caching, Async** | Performance fine on tiny dataset | Suggested optimizations in `DESIGN.md` |

---

## Step-by-Step Tasks That Were Completed
**Set up project & environment**
- Python 3.11 virtualenv.
- Installed: `torch`, `torchvision`, `Pillow`, `numpy`, `faiss-cpu`, `openai-clip`, `fastapi`, `uvicorn`.

**Prepared data**
- Collected 17 product images.
- Created `products.jsonl` with ID, name, category, price, description.

**Core logic**
- Loaded CLIP model on CPU.
- Computed normalized embeddings for products.
- Implemented cosine similarity search.

**API**
- `/health`, `/match-image`, `/match-text` endpoints via FastAPI.

**Docker**
- Base: `python:3.11-slim`.
- Installed dependencies.
- Copied code & data.
- Ran API on port 9000.

---

## **How This Still Meets the Original Brief**
| Original Brief Item | MVP Implementation |
|---------------------|--------------------|
| **Vector DB** | NumPy cosine similarity (can upgrade later) |
| **NoSQL (Mongo)** | JSONL metadata/logs; Mongo schema in `DESIGN.md` |
| **VLM & Vision Encoder** | CLIP image & text encoders |
| **Quantization (TensorRT)** | Documented in `DESIGN.md` |
| **Triton Deployment** | Documented in `DESIGN.md` |
| **Matching Pipeline** | Implemented |
| **Logging & Errors** | Implemented |
| **Bonus (Docker)** | Implemented |

---

## **Running Locally (Without Docker)**

**Clone repo & enter directory**
```bash
git clone https://github.com/anuragxorma/product-matching
cd product-matching
```

**Create virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Install dependencies**
```bash
pip install torch torchvision pillow numpy faiss-cpu git+https://github.com/openai/CLIP.git fastapi uvicorn
```

**Run demo script**
```bash
python pm_demo.py
```

**Run API**
```bash
uvicorn app:app --reload --port 9000
```

**Test**
```bash
curl http://localhost:9000/health
```

## Running with Docker

### Build image
```bash
docker build -t product-matching:cpu .
```
### Run container with caching

```bash
mkdir -p ~/.cache/clip ~/.cache/torch

docker run --rm -p 9000:9000 \
-v $PWD/data:/app/data:ro \
-v $HOME/.cache/clip:/root/.cache/clip \
-v $HOME/.cache/torch:/root/.cache/torch \
product-matching:cpu
```

### Test API
```bash
curl http://localhost:9000/health
```

## Future Improvements
- Switch to Triton Inference Server for scalable model serving.
- Quantize models with TensorRT for faster inference on GPU.
- Use an external vector database (Qdrant, Pinecone) for large catalogs.
- Integrate MongoDB for scalable metadata and logging.
- Add batching, caching, and async for high throughput.
- Add a web UI for easier interaction.

---

## Project Structure
```bash
product-matching/
│── data/
│   ├── images/               # Product images
│   ├── products.jsonl        # Product metadata
│── notebooks/                # Optional experiments
│── app.py                    # FastAPI app
│── pm_demo.py                 # Simple demo script
│── logs.jsonl                 # Query logs
│── Dockerfile                 # Docker build file
│── README.md                  # This file
│── DESIGN.md                  # Production upgrade plan
```

---

## Additional Resources
For details on scaling this prototype to production including NVIDIA Triton serving, TensorRT quantization, MongoDB integration, and vector database migration — please see **[DESIGN.md](./DESIGN.md)**.

---



