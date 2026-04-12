import pdfplumber
import json
import logging
import time
import re
from tqdm import tqdm
from openai import OpenAI


# Global Settings
logging.getLogger("pdfminer").setLevel(logging.ERROR)

api_key = "YOUR_OWN_API_KEY"
#replace with your own openai API key
client = OpenAI(api_key=api_key)

# ----------------------------
# PDF to Text Extraction
# ----------------------------
def pdf_to_pages(pdf_path):
    """Extract text from PDF, one string per page"""
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                pages_text.append(text.strip())
            else:
                pages_text.append(f"[Page {i} has no extractable text]")
    return pages_text


def build_raw_knowledge():
    """Build raw text corpus from PDF files"""
    pdf_files = [
        "Japan_Neurology_guideline2018.pdf",
        "NICE_UK_Epilepsy_Standard_2025.pdf",
        "SIGN_Scotland_Epilepsy_Standard_2018.pdf",
        "AES_Epilepsy_Guidelines.pdf",
        "ILAE_Epilepsy_Guidelines.pdf",
        "Epilepsia2010-Definition-of-drug-resistant-epilepsy-ILAE.pdf",
        "Epilepsia-2020-Lhatoo-Big-data-in-epilepsy-Clinical-ILAE.pdf",
        "Epilepsia-2021-Epilepsy-care-during-COVID19.pdf",
        "Epilepsia-2022-Schulze-diagnostic-hospital-documentation.pdf",
        "Epilepsia-2022-Operational-Classification.pdf",
        "JES_15_104.pdf",
        "JES_16_71.pdf",
        "JES_18_A000175.pdf",
        "JES_18_A000175.pdf"
    ]
    knowledge_base = []

    for pdf in pdf_files:
        pages = pdf_to_pages(pdf)
        for page_no, text in enumerate(pages, start=1):
            knowledge_base.append({
                "source_doc": pdf,
                "page_no": page_no,
                "text": text
            })
        print(f"Processed {pdf}, extracted {len(pages)} pages")

    json_file = "knowledge.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
    print(f"Generated {json_file}, total {len(knowledge_base)} pages of text")

    return json_file


# ----------------------------
# Utility Functions
# ----------------------------
def _extract_json_substring(text: str):
    text = str(text or "").strip()
    if not text:
        return ""
    # Remove markdown code fences if present.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    # Prefer object, then array.
    obj_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if obj_match:
        return obj_match.group(0)
    arr_match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if arr_match:
        return arr_match.group(0)
    return text


def safe_json_parse(text):
    """Parse JSON robustly; return (parsed, ok)."""
    if isinstance(text, (dict, list)):
        return text, True
    try:
        return json.loads(text), True
    except Exception:
        pass
    try:
        repaired = _extract_json_substring(text)
        return json.loads(repaired), True
    except Exception:
        return {"raw_output": str(text)}, False


def load_json_lines(path):
    """Load knowledge.json as a list of records."""
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for d in data:
            docs.append(d)
    return docs


def infer_source_type(source_doc: str) -> str:
    name = str(source_doc).lower()
    if any(k in name for k in ["guideline", "nice", "sign", "ilae", "aes"]):
        return "guideline"
    if "case" in name:
        return "case_report"
    return "paper"


def source_reliability_from_type(source_type: str) -> float:
    # guideline > paper > case_report
    if source_type == "guideline":
        return 0.95
    if source_type == "paper":
        return 0.85
    if source_type == "case_report":
        return 0.65
    return 0.80


def throttled_sleep(page_idx, sleep_per_page=3, sleep_every_n=20, long_sleep=60):
    """Throttle to avoid API rate limits"""
    time.sleep(sleep_per_page)
    if (page_idx + 1) % sleep_every_n == 0:
        print(f"Reached {page_idx+1} pages, taking a long break ({long_sleep}s)...")
        time.sleep(long_sleep)


# ----------------------------
# Step 1: Entity Extraction (NER)
# ----------------------------
def extract_entities_with_gpt(text, max_retries=2):
    """Extract biomedical entities using GPT"""
    base_prompt = f"""
You are a professional biomedical NER (Named Entity Recognition) system.

Given the following medical text, extract a comprehensive list of key biomedical entities.
Return your output as a strict JSON object where:
- keys are semantic entity category names (free-form, concise),
- values are arrays of entity strings,
- no extra explanations or markdown.

Text:
\"\"\"{text}\"\"\"
"""
    repair_prompt = "Your previous response was not valid JSON. Return ONLY a valid JSON object."
    messages = [
        {"role": "system", "content": "You are a biomedical entity extraction model."},
        {"role": "user", "content": base_prompt},
    ]
    last_output = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0,
                max_tokens=700,
                response_format={"type": "json_object"},
            )
            output_text = response.choices[0].message.content
            parsed, ok = safe_json_parse(output_text)
            if ok and isinstance(parsed, dict):
                # Normalize: all values must be arrays of strings.
                normalized = {}
                for k, v in parsed.items():
                    key = str(k).strip()
                    if not key:
                        continue
                    if isinstance(v, list):
                        normalized[key] = [str(x).strip() for x in v if str(x).strip()]
                    elif isinstance(v, str) and v.strip():
                        normalized[key] = [v.strip()]
                    else:
                        normalized[key] = []
                return normalized, {"parse_status": "ok" if attempt == 0 else "ok_after_retry", "attempt": attempt}
            last_output = output_text
            messages.append({"role": "assistant", "content": str(output_text)})
            messages.append({"role": "user", "content": repair_prompt})
        except Exception as e:
            last_output = str(e)
            messages.append({"role": "user", "content": repair_prompt})
    return {"raw_output": str(last_output)}, {"parse_status": "failed", "attempt": max_retries}


# ----------------------------
# Step 2: Relation Extraction (RE)
# ----------------------------
def extract_relations_with_gpt(entities, text, max_retries=2):
    """
    Input: Extracted entities + original text
    Output: Triplets [{"head": ..., "relation": ..., "tail": ...}]
    """
    base_prompt = f"""
You are a biomedical relation extraction system.

Given the following text and extracted biomedical entities, infer meaningful relationships among them.
Only output explicit, verifiable relations that are medically relevant, such as:
- Drug treats Disease
- Test diagnoses Disease
- Symptom indicates Disease
- RiskFactor causes Disease
- Surgery treats Condition

Return a JSON list of triplets with "head", "relation", and "tail" keys.
Do NOT include explanations or markdown.

Text:
\"\"\"{text}\"\"\"

Entities:
{json.dumps(entities, ensure_ascii=False)}
"""
    repair_prompt = (
        "Your previous response was not valid JSON list. "
        "Return ONLY a JSON array of objects with keys: head, relation, tail."
    )
    messages = [
        {"role": "system", "content": "You are a biomedical relation extraction assistant."},
        {"role": "user", "content": base_prompt},
    ]
    last_output = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0,
                max_tokens=700,
                response_format={"type": "json_object"},
            )
            output_text = response.choices[0].message.content
            parsed, ok = safe_json_parse(output_text)

            # Allow both direct list or wrapped object forms.
            candidates = None
            if isinstance(parsed, list):
                candidates = parsed
            elif isinstance(parsed, dict):
                if isinstance(parsed.get("triplets"), list):
                    candidates = parsed.get("triplets")
                elif isinstance(parsed.get("relations"), list):
                    candidates = parsed.get("relations")
                elif isinstance(parsed.get("data"), list):
                    candidates = parsed.get("data")

            if ok and isinstance(candidates, list):
                clean_triplets = []
                for t in candidates:
                    if isinstance(t, dict) and all(k in t for k in ["head", "relation", "tail"]):
                        head = str(t["head"]).strip()
                        rel = str(t["relation"]).strip()
                        tail = str(t["tail"]).strip()
                        if head and rel and tail:
                            clean_triplets.append({"head": head, "relation": rel, "tail": tail})
                return clean_triplets, {"parse_status": "ok" if attempt == 0 else "ok_after_retry", "attempt": attempt}

            last_output = output_text
            messages.append({"role": "assistant", "content": str(output_text)})
            messages.append({"role": "user", "content": repair_prompt})
        except Exception as e:
            last_output = str(e)
            messages.append({"role": "user", "content": repair_prompt})
    return [], {"parse_status": "failed", "attempt": max_retries, "raw_output": str(last_output)}


# ----------------------------
# Step 3: Processing Pipeline
# ----------------------------
def process_documents(docs, sleep_per_page=3, sleep_every_n=20, long_sleep=60):
    """Run NER and RE for each page"""
    results = []
    for idx, doc in enumerate(tqdm(docs, desc="Processing documents")):
        if isinstance(doc, dict):
            source_doc = doc.get("source_doc", "unknown")
            page_no = doc.get("page_no", None)
            text = str(doc.get("text", ""))
        else:
            source_doc = "unknown"
            page_no = None
            text = str(doc)

        source_type = infer_source_type(source_doc)
        reliability = source_reliability_from_type(source_type)

        entities, ner_meta = extract_entities_with_gpt(text)
        triplets, re_meta = extract_relations_with_gpt(entities, text)
        enriched_triplets = []
        for t in triplets:
            t_new = dict(t)
            t_new["source"] = source_doc
            t_new["source_type"] = source_type
            t_new["reliability"] = reliability
            t_new["page_no"] = page_no
            enriched_triplets.append(t_new)

        results.append({
            "page_id": idx,
            "source": source_doc,
            "source_type": source_type,
            "reliability": reliability,
            "page_no": page_no,
            "text": text,
            "entities": entities,
            "triplets": enriched_triplets,
            "ner_parse_status": ner_meta.get("parse_status"),
            "re_parse_status": re_meta.get("parse_status"),
            "ner_attempt": ner_meta.get("attempt"),
            "re_attempt": re_meta.get("attempt"),
        })

        throttled_sleep(idx, sleep_per_page, sleep_every_n, long_sleep)
    return results


def save_to_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def flatten_triplets_for_retrieval(page_level_records):
    """
    Convert page-level KG records into flat triplet list for FAISS id alignment.
    """
    flat = []
    for item in page_level_records:
        if not isinstance(item, dict):
            continue
        page_id = item.get("page_id")
        source = item.get("source", "unknown")
        source_type = item.get("source_type", infer_source_type(source))
        reliability = item.get("reliability", source_reliability_from_type(source_type))
        for tri in item.get("triplets", []):
            if not isinstance(tri, dict):
                continue
            if not all(k in tri for k in ["head", "relation", "tail"]):
                continue
            head = str(tri.get("head", "")).strip().lower()
            relation = str(tri.get("relation", "")).strip().lower()
            tail = str(tri.get("tail", "")).strip().lower()
            if not head or not tail:
                continue
            flat.append({
                "head": head,
                "relation": relation,
                "tail": tail,
                "source_page": page_id,
                "source": tri.get("source", source),
                "source_type": tri.get("source_type", source_type),
                "reliability": tri.get("reliability", reliability),
            })
    return flat


if __name__ == "__main__":
    print("Step 1: Extracting raw text from PDFs ...")
    knowledge_file = build_raw_knowledge()

    print("Step 2: Performing entity & relation extraction ...")
    docs = load_json_lines(knowledge_file)
    ner_re_results = process_documents(docs, sleep_per_page=3, sleep_every_n=20, long_sleep=60)

    save_to_json(ner_re_results, "KG_triplets.json")
    print("Knowledge graph triplets saved to KG_triplets.json")

    flat_triplets = flatten_triplets_for_retrieval(ner_re_results)
    save_to_json(flat_triplets, "KG_triplets_flat.json")
    print(f"Flattened triplets saved to KG_triplets_flat.json (count={len(flat_triplets)})")
