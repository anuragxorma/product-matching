from pm_lib import (
    read_products, load_clip, embed_images, embed_texts,
    build_index, search_topk, log_jsonl
)

PRODUCTS = "data/products.jsonl"
LOGS = "logs.jsonl"

def main():
    # Load metadata + model
    products = read_products(PRODUCTS)
    model, preprocess = load_clip()

    # Build catalog embeddings + index
    image_paths = [p["image_path"] for p in products]
    embs = embed_images(model, preprocess, image_paths)   # [N,512], normalized
    index = build_index(embs)

    # Test: image query (use first product image)
    q_emb = embed_images(model, preprocess, [products[0]["image_path"]])  # [1,512]
    scores, idxs = search_topk(index, q_emb, k=3)
    top = products[int(idxs[0])]
    print("Top-1 image match:", top["name"], "| score:", float(scores[0]))
    log_jsonl(LOGS, {"type": "image", "query": "self", "top1_id": top["product_id"], "score": float(scores[0])})

    # Test: text query
    query_text = "ceramic mug"
    t_emb = embed_texts(model, [query_text])  # [1,512]
    scores2, idxs2 = search_topk(index, t_emb, k=3)
    top2 = products[int(idxs2[0])]
    print("Top-1 text match:", top2["name"], "| score:", float(scores2[0]))
    log_jsonl(LOGS, {"type": "text", "query": query_text, "top1_id": top2["product_id"], "score": float(scores2[0])})

if __name__ == "__main__":
    main()
