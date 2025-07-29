import os
import json
from datetime import datetime
from utils import extract_text_chunks, get_embeddings, rank_chunks, summarize_text

def main():
    # Load persona.json
    with open("persona_job.json", "r", encoding="utf-8") as f:
        persona_data = json.load(f)

    input_folder = "./input"  # folder with all PDFs
    pdf_files = [f for f in os.listdir(input_folder) if f.endswith(".pdf")]

    all_chunks = []
    for filename in pdf_files:
        print(f"🔍 Processing: {filename}")
        filepath = os.path.join(input_folder, filename)
        chunks = extract_text_chunks(filepath)
        all_chunks.extend(chunks)

    texts = [chunk["text"] for chunk in all_chunks]
    chunk_embeddings = get_embeddings(texts)
    query = persona_data["job_to_be_done"]
    query_embedding = get_embeddings([query])[0]

    top_chunks = rank_chunks(all_chunks, chunk_embeddings, query_embedding, top_k=5)

    extracted_sections = []
    subsection_analysis = []

    for rank, chunk in enumerate(top_chunks, 1):
        section = {
            "document": chunk["document"],
            "section_title": chunk["title"],
            "importance_rank": rank,
            "page_number": chunk["page"]
        }
        extracted_sections.append(section)

        summary = summarize_text(chunk["text"])
        subsection_analysis.append({
            "document": chunk["document"],
            "page_number": chunk["page"],
            "refined_text": summary
        })

    final_output = {
        "metadata": {
            "input_documents": pdf_files,
            "persona": persona_data["persona"],
            "job_to_be_done": persona_data["job_to_be_done"],
            "processing_timestamp": datetime.utcnow().isoformat() + "Z"
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print("✅ Output saved to output.json")

if __name__ == "__main__":
    main()
