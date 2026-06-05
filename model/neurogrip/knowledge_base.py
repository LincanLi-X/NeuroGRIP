import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover - depends on optional local install
    pdfplumber = None


logging.getLogger("pdfminer").setLevel(logging.ERROR)


def pdf_to_pages(pdf_path: str) -> List[str]:
    """Extract text from a PDF, one string per page."""
    if pdfplumber is None:
        raise ImportError("pdfplumber is required for PDF extraction. Install requirements.txt first.")

    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            pages_text.append(text.strip() if text else f"[Page {i} has no extractable text]")
    return pages_text


def build_raw_knowledge(
    pdf_files: Iterable[str],
    output_path: str = "model/neurogrip/knowledge.json",
    page_limit: Optional[int] = None,
) -> str:
    """Serialize guideline PDFs into a page-level text corpus."""
    knowledge_base = []

    for pdf in pdf_files:
        pages = pdf_to_pages(pdf)
        if page_limit is not None:
            pages = pages[:page_limit]
        for page_no, text in enumerate(pages, start=1):
            knowledge_base.append({
                "source_doc": os.path.basename(pdf),
                "page_no": page_no,
                "text": text,
            })
        print(f"Processed {pdf}, extracted {len(pages)} pages")

    save_to_json(knowledge_base, output_path)
    print(f"Generated {output_path}, total {len(knowledge_base)} pages of text")
    return output_path


def _extract_json_substring(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
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
        return json.loads(_extract_json_substring(text)), True
    except Exception:
        return {"raw_output": str(text)}, False


def load_json_lines(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    docs = []
    for idx, item in enumerate(data):
        if isinstance(item, dict):
            docs.append(item)
        else:
            docs.append({"source_doc": "legacy_unknown", "page_no": idx + 1, "text": str(item)})
    return docs


def infer_source_type(source_doc: str) -> str:
    name = str(source_doc).lower()
    if any(k in name for k in ["guideline", "nice", "sign", "ilae", "aes"]):
        return "guideline"
    if "case" in name:
        return "case_report"
    return "paper"


def source_reliability_from_type(source_type: str) -> float:
    if source_type == "guideline":
        return 0.95
    if source_type == "paper":
        return 0.85
    if source_type == "case_report":
        return 0.65
    return 0.80


def throttled_sleep(page_idx: int, sleep_per_page: float, sleep_every_n: int, long_sleep: float):
    if sleep_per_page > 0:
        time.sleep(sleep_per_page)
    if sleep_every_n > 0 and (page_idx + 1) % sleep_every_n == 0:
        print(f"Reached {page_idx + 1} pages, taking a long break ({long_sleep}s)...")
        time.sleep(long_sleep)


def _client_from_env(api_key: Optional[str] = None) -> OpenAI:
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("OPENAI_API_KEY is required for LLM-based KG construction.")
    return OpenAI(api_key=key)


def _chat_json(client: OpenAI, model_name: str, messages: List[dict], max_tokens: int):
    return client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    ).choices[0].message.content


def extract_entities_with_gpt(
    text: str,
    client: OpenAI,
    model_name: str = "gpt-5.1",
    max_retries: int = 2,
) -> Tuple[Dict, Dict]:
    """Extract biomedical entities using an OpenAI model."""
    base_prompt = f"""
You are a professional biomedical NER (Named Entity Recognition) system.

Given the following medical text, extract a comprehensive list of key biomedical entities.
Return a strict JSON object where keys are semantic entity categories and values are arrays of strings.
Do not include explanations or markdown.

Text:
\"\"\"{text}\"\"\"
"""
    messages = [
        {"role": "system", "content": "You extract biomedical entities for epilepsy EEG knowledge graphs."},
        {"role": "user", "content": base_prompt},
    ]
    last_output = None
    for attempt in range(max_retries + 1):
        try:
            output_text = _chat_json(client, model_name, messages, max_tokens=900)
            parsed, ok = safe_json_parse(output_text)
            if ok and isinstance(parsed, dict):
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
        except Exception as exc:
            last_output = str(exc)
        messages.append({"role": "assistant", "content": str(last_output)})
        messages.append({"role": "user", "content": "Return ONLY a valid JSON object."})
    return {"raw_output": str(last_output)}, {"parse_status": "failed", "attempt": max_retries}


def extract_relations_with_gpt(
    entities: Dict,
    text: str,
    client: OpenAI,
    model_name: str = "gpt-5.1",
    max_retries: int = 2,
) -> Tuple[List[Dict], Dict]:
    """Extract clinical triplets from one page of text."""
    base_prompt = f"""
You are a biomedical relation extraction system for epilepsy and EEG diagnosis.

Given the following text and extracted biomedical entities, infer clinically meaningful relationships.
Only output explicit or strongly implied medically relevant relations.
Return a strict JSON object with one key "triplets"; its value is an array of objects with keys:
"head", "relation", and "tail".

Examples of relation semantics:
- test detects biomarker
- disease presents_with symptom
- disease affects brain_region
- risk_factor causes disease
- treatment treats disease

Text:
\"\"\"{text}\"\"\"

Entities:
{json.dumps(entities, ensure_ascii=False)}
"""
    messages = [
        {"role": "system", "content": "You extract clinically valid biomedical relation triplets."},
        {"role": "user", "content": base_prompt},
    ]
    last_output = None
    for attempt in range(max_retries + 1):
        try:
            output_text = _chat_json(client, model_name, messages, max_tokens=1000)
            parsed, ok = safe_json_parse(output_text)
            candidates = None
            if isinstance(parsed, dict):
                candidates = parsed.get("triplets") or parsed.get("relations") or parsed.get("data")
            elif isinstance(parsed, list):
                candidates = parsed

            if ok and isinstance(candidates, list):
                clean_triplets = []
                for item in candidates:
                    if isinstance(item, dict) and all(k in item for k in ["head", "relation", "tail"]):
                        head = str(item["head"]).strip()
                        relation = str(item["relation"]).strip()
                        tail = str(item["tail"]).strip()
                    elif isinstance(item, (list, tuple)) and len(item) >= 3:
                        head, relation, tail = [str(x).strip() for x in item[:3]]
                    else:
                        continue
                    if head and relation and tail:
                        clean_triplets.append({"head": head, "relation": relation, "tail": tail})
                return clean_triplets, {"parse_status": "ok" if attempt == 0 else "ok_after_retry", "attempt": attempt}
            last_output = output_text
        except Exception as exc:
            last_output = str(exc)
        messages.append({"role": "assistant", "content": str(last_output)})
        messages.append({"role": "user", "content": "Return ONLY a JSON object with key triplets."})
    return [], {"parse_status": "failed", "attempt": max_retries, "raw_output": str(last_output)}


def process_documents(
    docs: List[dict],
    client: OpenAI,
    model_name: str = "gpt-5.1",
    sleep_per_page: float = 3,
    sleep_every_n: int = 20,
    long_sleep: float = 60,
    max_pages: Optional[int] = None,
) -> List[dict]:
    """Run NER and RE over page-level documents."""
    if max_pages is not None:
        docs = docs[:max_pages]

    results = []
    for idx, doc in enumerate(tqdm(docs, desc="Processing documents")):
        source_doc = doc.get("source_doc", "unknown")
        page_no = doc.get("page_no", None)
        text = str(doc.get("text", ""))
        source_type = infer_source_type(source_doc)
        reliability = source_reliability_from_type(source_type)

        entities, ner_meta = extract_entities_with_gpt(text, client=client, model_name=model_name)
        triplets, re_meta = extract_relations_with_gpt(entities, text, client=client, model_name=model_name)

        enriched_triplets = []
        for tri in triplets:
            enriched = dict(tri)
            enriched["source"] = source_doc
            enriched["source_type"] = source_type
            enriched["reliability"] = reliability
            enriched["page_no"] = page_no
            enriched_triplets.append(enriched)

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


def save_to_json(data, path: str):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def flatten_triplets_for_retrieval(page_level_records: List[dict]) -> List[dict]:
    """Convert page-level KG records into flat triplets for retriever/index id alignment."""
    flat = []
    for item in page_level_records:
        if not isinstance(item, dict):
            continue
        page_id = item.get("page_id")
        source = item.get("source") or item.get("source_doc") or "unknown"
        source_type = item.get("source_type", infer_source_type(source))
        reliability = item.get("reliability", source_reliability_from_type(source_type))
        for tri in item.get("triplets", []):
            if isinstance(tri, dict):
                head = str(tri.get("head", "")).strip().lower()
                relation = str(tri.get("relation", "")).strip().lower()
                tail = str(tri.get("tail", "")).strip().lower()
                tri_source = tri.get("source", source)
                tri_source_type = tri.get("source_type", source_type)
                tri_reliability = tri.get("reliability", reliability)
            elif isinstance(tri, (list, tuple)) and len(tri) >= 3:
                head, relation, tail = [str(x).strip().lower() for x in tri[:3]]
                tri_source = source
                tri_source_type = source_type
                tri_reliability = reliability
            else:
                continue
            if not head or not tail:
                continue
            flat.append({
                "head": head,
                "relation": relation,
                "tail": tail,
                "source_page": page_id,
                "source": tri_source,
                "source_type": tri_source_type,
                "reliability": tri_reliability,
            })
    return flat


def parse_args():
    parser = argparse.ArgumentParser("Build NeuroGRIP epilepsy EEG knowledge graph.")
    parser.add_argument("--input_pdfs", nargs="*", default=None, help="Guideline/literature PDFs to serialize.")
    parser.add_argument("--knowledge_in", type=str, default=None, help="Existing page-level knowledge JSON.")
    parser.add_argument("--knowledge_out", type=str, default="model/neurogrip/knowledge.json")
    parser.add_argument("--kg_out", type=str, default="model/neurogrip/KG_triplets.json")
    parser.add_argument("--flat_out", type=str, default="model/neurogrip/KG_triplets_flat.json")
    parser.add_argument("--model", type=str, default=os.environ.get("NEUROGRIP_KG_MODEL", "gpt-5.1"))
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--sleep_per_page", type=float, default=3)
    parser.add_argument("--sleep_every_n", type=int, default=20)
    parser.add_argument("--long_sleep", type=float, default=60)
    parser.add_argument("--page_limit", type=int, default=None, help="Limit PDF extraction pages per PDF.")
    parser.add_argument("--max_pages", type=int, default=None, help="Limit LLM extraction pages after loading corpus.")
    parser.add_argument("--flatten_only", action="store_true", help="Only flatten an existing KG JSON.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.flatten_only:
        source = args.kg_out if os.path.exists(args.kg_out) else args.knowledge_in
        if source is None:
            raise ValueError("--flatten_only requires --kg_out or --knowledge_in.")
        records = load_json_lines(source)
        flat_triplets = flatten_triplets_for_retrieval(records)
        save_to_json(flat_triplets, args.flat_out)
        print(f"Flattened triplets saved to {args.flat_out} (count={len(flat_triplets)})")
        return

    if args.knowledge_in:
        knowledge_file = args.knowledge_in
    else:
        if not args.input_pdfs:
            raise ValueError("Provide --input_pdfs or --knowledge_in.")
        knowledge_file = build_raw_knowledge(args.input_pdfs, args.knowledge_out, page_limit=args.page_limit)

    docs = load_json_lines(knowledge_file)
    client = _client_from_env(args.api_key)
    kg_records = process_documents(
        docs,
        client=client,
        model_name=args.model,
        sleep_per_page=args.sleep_per_page,
        sleep_every_n=args.sleep_every_n,
        long_sleep=args.long_sleep,
        max_pages=args.max_pages,
    )
    save_to_json(kg_records, args.kg_out)
    print(f"Knowledge graph triplets saved to {args.kg_out}")

    flat_triplets = flatten_triplets_for_retrieval(kg_records)
    save_to_json(flat_triplets, args.flat_out)
    print(f"Flattened triplets saved to {args.flat_out} (count={len(flat_triplets)})")


if __name__ == "__main__":
    main()
