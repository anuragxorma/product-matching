# DESIGN.md – Production Upgrade Plan

## Purpose
This document describes how the current product matching MVP can be scaled and modified for production environments, including components from the original brief that were intentionally omitted.

---

## 1. Current MVP Architecture

**Components Implemented**
- **Data Storage**: Metadata (`products.jsonl`) and logs (`logs.jsonl`).
- **Model**: Pre-trained CLIP (ViT-B/32) for image and text embeddings.
- **Similarity Search**: In-memory cosine similarity (NumPy).
- **API**: FastAPI endpoints `/match-image`, `/match-text`, `/health`.
- **Deployment**: Local run or Docker container.

---

## 2. Features Skipped & Why

| Feature                           | Why Skipped                                         | MVP Alternative                                    |
|-----------------------------------|-----------------------------------------------------|----------------------------------------------------|
| **NVIDIA Triton Serving**         | Heavy container orchestration, unnecessary for CPU-only prototype | Documented setup plan for future GPU-based serving |
| **TensorRT Quantization (.plan)** | Requires GPU + calibration data                     | Provided ONNX export + `trtexec --fp16` example    |
| **External Vector DB (Qdrant, Pinecone)** | Overkill for 17 items                              | In-memory NumPy search; migration path documented  |
| **MongoDB for Metadata/Logs**     | Adds infra complexity                               | JSONL files for simplicity; schema outlined below |
| **Batching, Caching, Async**      | Not needed for small dataset                        | Performance already sufficient in MVP              |            |

---

## 3. Future Scaling Plan

### 3.1 Model Serving with NVIDIA Triton
- **Why**: Handles concurrent requests efficiently; supports multiple models & versions.
- **Plan**:
  1. Export CLIP model to ONNX.
  2. Convert to TensorRT plan for GPU inference.
  3. Deploy with Triton Inference Server container.
  4. Expose endpoints for inference over gRPC/HTTP.

---

### 3.2 Quantization with TensorRT
- **Why**: Reduce inference latency and memory footprint on GPU.
- **Plan**:
  1. Export to ONNX.
  2. Run `trtexec --onnx=model.onnx --fp16 --saveEngine=model.plan`.
  3. Load `.plan` file in Triton for serving.

---

### 3.3 Vector Database Integration
- **Options**: Qdrant, Pinecone, Weaviate, FAISS server.
- **Benefits**: Scales similarity search to millions of vectors.
- **Plan**:
  1. Store product embeddings in vector DB.
  2. Replace NumPy search with vector DB queries.
  3. Use metadata filters (e.g., category, price range).

---

### 3.4 MongoDB for Metadata & Logging
- **Why**: JSONL is fine for small data, but MongoDB supports:
  - Indexed queries
  - Complex filtering
  - Persistent logs
- **Proposed Schema**:
```json
{
  "product_id": "shoes1",
  "name": "Brown Leather Boots",
  "category": "Footwear",
  "price": 79.99,
  "image_path": "images/shoes1.jpg",
  "description": "Brown leather boots",
  "embedding_vector": [ ... ]
}
```
### 3.5 Performance Optimizations
- Add batch processing for multiple queries.
- Use caching for repeated queries.
- Implement async I/O in FastAPI for higher throughput.

---

### 3.6 User Interface
- Build a web UI with file upload and search bar.
- Display top matches with score and product details.
- Integrate drag-and-drop for image uploads.

---

## 4. Security & Deployment Considerations
- Use Docker multi-stage builds for smaller images.
- Add request validation and size limits to prevent abuse.
- For production, deploy behind a reverse proxy (e.g., Nginx).
- Add monitoring (Prometheus, Grafana) for API performance.

---

## 5. Testing Plan
**Unit tests for:**
- Embedding generation
- Similarity computation
- API endpoints

**Integration test:**
- End-to-end query → match → log entry.

**Load testing:**
- Use locust or k6 when scaling.

---

## 6. Summary
This MVP demonstrates:
- End-to-end image/text product matching with CLIP.
- Minimal infrastructure for quick demonstration.
- Clear upgrade paths for production readiness.

The focus was on core functionality first, with scalability documented rather than implemented, aligning with rapid prototyping goals.

