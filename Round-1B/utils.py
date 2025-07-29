import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np
import os

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def extract_text_chunks(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("blocks")
        for b in blocks:
            text = b[4].strip()
            if len(text.split()) >= 20:
                title = next((line.strip() for line in text.split('\n') if line.strip() and len(line.strip()) > 10), text[:50])
                chunks.append({
                    "document":os.path.basename(pdf_path),
                    "page": page_num,
                    "text": text,
                    "title": title
                })
    return chunks

def get_embeddings(texts):
    if not texts:
        return np.array([])  # Important to avoid error
    return model.encode(texts, show_progress_bar=False)

def rank_chunks(chunks, chunk_embeddings, query_embedding, top_k=5):
    if len(chunk_embeddings) == 0:
        return []

    sims = cosine_similarity([query_embedding], chunk_embeddings)[0]
    for idx, sim in enumerate(sims):
        chunks[idx]["score"] = sim

    # Sort all chunks by similarity
    sorted_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)

    # ✅ Select top chunks from DIFFERENT documents
    seen_docs = set()
    top_ranked = []
    for chunk in sorted_chunks:
        doc = chunk["document"]
        if doc not in seen_docs:
            top_ranked.append(chunk)
            seen_docs.add(doc)
        if len(top_ranked) == top_k:
            break
    return top_ranked

def summarize_text(text, num_sentences=2):
    sentences = text.replace("\n", " ").split(". ")
    summary = ". ".join(sentences[:num_sentences])
    return summary.strip() + "." if summary else ""
