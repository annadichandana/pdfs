import os
import fitz
import json
import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_persona_task(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        persona = lines[0].split(":", 1)[1].strip()
        job = lines[1].split(":", 1)[1].strip()
    return persona, job

def extract_sections_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    sections = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        for para in text.split('\n\n'):
            cleaned = re.sub(r'\s+', ' ', para).strip()
            if len(cleaned) > 50:
                sections.append({
                    "document": os.path.basename(pdf_path),
                    "page": page_num,
                    "text": cleaned
                })
    return sections

def rank_sections(sections, query):
    if not sections:
        return []
    texts = [s["text"] for s in sections]
    vectorizer = TfidfVectorizer().fit([query] + texts)
    query_vec = vectorizer.transform([query])
    section_vecs = vectorizer.transform(texts)
    similarities = cosine_similarity(query_vec, section_vecs)[0]
    for i, score in enumerate(similarities):
        sections[i]["score"] = score
    return sorted(sections, key=lambda x: x["score"], reverse=True)

def write_output(sections, persona, job, output_path):
    metadata = {
        "documents": list(set(s["document"] for s in sections)),
        "persona": persona,
        "job_to_be_done": job,
        "timestamp": datetime.datetime.now().isoformat()
    }
    extracted_sections = []
    subsection_analysis = []
    for i, sec in enumerate(sections[:5]):
        extracted_sections.append({
            "document": sec["document"],
            "page": sec["page"],
            "section_title": sec["text"][:50] + "...",
            "importance_rank": i + 1
        })
        subsection_analysis.append({
            "document": sec["document"],
            "refined_text": sec["text"],
            "page": sec["page"]
        })
    result = {
        "metadata": metadata,
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

def main():
    input_dir = "/app/input"
    output_file = "/app/output/output.json"
    persona_path = os.path.join(input_dir, "persona_task.txt")
    persona, job = read_persona_task(persona_path)
    query = f"{persona}. {job}"
    all_sections = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(input_dir, filename)
            all_sections += extract_sections_from_pdf(filepath)
    if not all_sections:
        return
    ranked = rank_sections(all_sections, query)
    write_output(ranked, persona, job, output_file)

if __name__ == "__main__":
    main()
