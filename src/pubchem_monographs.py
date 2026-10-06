"""Fetch PubChem clinical sections and run the same three corpus checks.

PubChem text is not a PDF. It is excluded from the five-PDF minimum and from
the 300-page gate.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from corpus_pipeline import (
    JACCARD_THRESHOLD,
    MIN_LETTERS,
    MIN_PAGES,
    MIN_WORDS,
    REPORTS_DIR,
    ROOT,
    SOURCES_PATH,
    STATS_PATH,
    detect_language,
    jaccard,
    letter_count,
    normalize_page,
    parse_staged_text,
    render_numbered_pages,
    update_checks,
    word_count,
    word_shingles,
)

PUBCHEM_DRUGS = [
    ("amoxicillin", 33613),
    ("doxycycline", 54671203),
    ("vancomycin", 14969),
    ("ceftriaxone", 5479530),
    ("azithromycin", 447043),
    ("metronidazole", 4173),
    ("artesunate", 6917864),
    # The combination record CID 6450800 has almost no clinical text.
    # Artemether and lumefantrine are the two components of that medicine.
    ("artemether", 68911),
    ("lumefantrine", 6437380),
]
CLINICAL_HEADINGS = [
    "Drug Indication",
    "Therapeutic Uses",
    "Drug Warnings",
    "Drug-Drug Interactions",
    "Drug-Food Interactions",
    "EMA Drug Information",
    "Pharmacodynamics",
    "Absorption, Distribution and Excretion",
    "Metabolism/Metabolites",
    "Biological Half-Life",
    "Mechanism of Action",
    "Toxicological Information",
    "Clinical Laboratory Methods",
    # The first 13 headings leave doxycycline at 759 words. These three are
    # still PubChem drug-information sections, and they bring that record
    # over the 1,000-word document rule.
    "Protein Binding",
    "Record Description",
    "GHS Classification",
]
USER_AGENT = "BITS-Mistral-corpus/1.0 (educational assignment)"
RAW_DIR = ROOT / "data" / "pubchem" / "sections"
CORPUS_DIR = ROOT / "domain_corpus"
PUBCHEM_STATS_PATH = REPORTS_DIR / "pubchem_stats.csv"
PUBCHEM_DRUG_COUNTS_PATH = REPORTS_DIR / "pubchem_drug_counts.csv"


def fetch_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=90) as response:
        return json.loads(response.read().decode())


def collect_strings(node, found: list[str]) -> None:
    if isinstance(node, dict):
        value = node.get("String")
        if isinstance(value, str):
            found.append(value)
        for child in node.values():
            collect_strings(child, found)
    elif isinstance(node, list):
        for child in node:
            collect_strings(child, found)


def section_text(payload: dict) -> str:
    found: list[str] = []
    collect_strings(payload.get("Record", payload), found)
    kept = []
    seen = set()
    for value in found:
        cleaned = " ".join(value.split())
        if cleaned in seen or cleaned.startswith("http"):
            continue
        if len(cleaned.split()) < 8:
            continue
        seen.add(cleaned)
        kept.append(cleaned)
    return "\n\n".join(kept)


def fetch_drug_sections(drug_id: str, cid: int) -> list[tuple[str, str]]:
    cache_path = RAW_DIR / f"{drug_id}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(cached, dict) and cached.get("headings") == CLINICAL_HEADINGS:
            return [(item["heading"], item["text"]) for item in cached["sections"]]
    pages: list[tuple[str, str]] = []
    for heading in CLINICAL_HEADINGS:
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/"
            f"{cid}/JSON?heading={quote(heading)}"
        )
        try:
            payload = fetch_json(url)
        except HTTPError:
            time.sleep(0.25)
            continue
        text = section_text(payload)
        if letter_count(text) >= MIN_LETTERS:
            pages.append((heading, text))
        time.sleep(0.25)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"{drug_id}.json").write_text(
        json.dumps(
            {
                "headings": CLINICAL_HEADINGS,
                "sections": [{"heading": heading, "text": text} for heading, text in pages],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return pages


def apply_language(pages: list[tuple[int, str]]) -> tuple[list[tuple[int, str]], int, int]:
    kept = []
    noncontent = 0
    non_english = 0
    for number, text in pages:
        if letter_count(text) < MIN_LETTERS:
            noncontent += 1
            continue
        if detect_language(text) != "en":
            non_english += 1
            continue
        kept.append((number, text))
    return kept, noncontent, non_english


def build_pubchem_corpus() -> dict:
    drugs = []
    for drug_id, cid in PUBCHEM_DRUGS:
        sections = fetch_drug_sections(drug_id, cid)
        numbered = [(index, f"{heading}\n\n{text}") for index, (heading, text) in enumerate(sections, start=1)]
        kept, noncontent, non_english = apply_language(numbered)
        words = sum(word_count(text) for _, text in kept)
        drugs.append(
            {
                "id": drug_id,
                "cid": cid,
                "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
                "pages": kept,
                "page_count": len(kept),
                "word_count": words,
                "noncontent_pages": noncontent,
                "non_english_pages": non_english,
                "own_file": words >= MIN_WORDS and len(kept) >= MIN_PAGES,
            }
        )

    documents = assemble_documents(drugs)
    documents, removed_pages = drop_duplicate_pages(documents)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for stale in CORPUS_DIR.glob("pubchem_*.txt"):
        stale.unlink()
    written = []
    failed_length = []
    for document in documents:
        if not document["pages"]:
            continue
        path = CORPUS_DIR / document["filename"]
        path.write_text(render_numbered_pages(document["pages"]), encoding="utf-8")
        reloaded = parse_staged_text(path.read_text(encoding="utf-8"))
        pages = len(reloaded)
        words = sum(word_count(text) for _, text in reloaded)
        record = {
            "id": document["id"],
            "file": f"domain_corpus/{document['filename']}",
            "pages": reloaded,
            "page_count": pages,
            "word_count": words,
        }
        written.append(record)
        if pages < MIN_PAGES or words < MIN_WORDS:
            failed_length.append({"id": record["id"], "page_count": pages, "word_count": words})

    failed_language = []
    for document in written:
        for number, text in document["pages"]:
            if letter_count(text) >= MIN_LETTERS and detect_language(text) != "en":
                failed_language.append({"id": document["id"], "page_number": number})
    dedup_failures = find_duplicate_pages(written)
    pubchem_pages = sum(document["page_count"] for document in written)
    pubchem_words = sum(document["word_count"] for document in written)
    pdf_pages = pdf_pages_after_dedup()
    record_sources(drugs, written, pubchem_pages, pdf_pages)
    write_drug_counts(drugs)
    write_pubchem_stats(drugs, written, removed_pages, pubchem_pages, pubchem_words)
    checks = update_checks(
        {
            "pubchem_source": "NCBI PubChem clinical headings; not a PDF",
            "pubchem_counts_toward_pdf_page_gate": False,
            "pubchem_counts_toward_minimum_pdf_count": False,
            "pubchem_files": len(written),
            "pubchem_section_pages": pubchem_pages,
            "pubchem_words": pubchem_words,
            "pubchem_length_check_passed": not failed_length,
            "pubchem_language_check_passed": not failed_language,
            "pubchem_dedup_check_passed": not dedup_failures,
            "pdf_pages_after_dedup_excluding_pubchem": pdf_pages,
        }
    )
    if failed_length or failed_language or dedup_failures:
        raise RuntimeError(
            f"PubChem checks failed length={failed_length} language={failed_language[:3]} dedup={dedup_failures[:3]}"
        )
    return {"drugs": drugs, "files": written, "checks": checks, "pdf_pages": pdf_pages}


def assemble_documents(drugs: list[dict]) -> list[dict]:
    documents = []
    merged: list[tuple[int, str]] = []
    for drug in drugs:
        if drug["own_file"]:
            documents.append(
                {
                    "id": f"pubchem_{drug['id']}",
                    "filename": f"pubchem_{drug['id']}.txt",
                    "pages": drug["pages"],
                }
            )
        else:
            merged.extend(drug["pages"])
    if merged:
        documents.append(
            {
                "id": "pubchem_monographs",
                "filename": "pubchem_monographs.txt",
                "pages": [(index, text) for index, (_, text) in enumerate(merged, start=1)],
            }
        )
    return documents


def drop_duplicate_pages(documents: list[dict]) -> tuple[list[dict], list[dict]]:
    removed = []
    seen_hashes: dict[str, tuple[str, int]] = {}
    kept_shingles: list[tuple[str, int, set[str]]] = []
    for document in documents:
        kept_pages = []
        for number, text in document["pages"]:
            normalized = normalize_page(text)
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            earlier = seen_hashes.get(digest)
            shingles = word_shingles(normalized)
            near = None
            if earlier is None:
                for prior in kept_shingles:
                    if jaccard(shingles, prior[2]) >= JACCARD_THRESHOLD:
                        near = prior
                        break
            if earlier is not None or near is not None:
                removed.append(
                    {
                        "file": document["filename"],
                        "page_number": number,
                        "reason": "exact" if earlier is not None else "near",
                    }
                )
                continue
            seen_hashes[digest] = (document["id"], number)
            kept_shingles.append((document["id"], number, shingles))
            kept_pages.append((number, text))
        document["pages"] = [(index, text) for index, (_, text) in enumerate(kept_pages, start=1)]
    return documents, removed


def find_duplicate_pages(documents: list[dict]) -> list[dict]:
    failures = []
    seen_hashes: dict[str, tuple[str, int]] = {}
    pages = []
    for document in documents:
        for number, text in document["pages"]:
            normalized = normalize_page(text)
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            earlier = seen_hashes.get(digest)
            if earlier is not None:
                failures.append({"reason": "exact", "file": document["file"], "page_number": number})
            else:
                seen_hashes[digest] = (document["file"], number)
            pages.append((document["file"], number, word_shingles(normalized)))
    for index, page in enumerate(pages):
        for earlier in pages[:index]:
            if jaccard(page[2], earlier[2]) >= JACCARD_THRESHOLD:
                failures.append(
                    {
                        "reason": "near",
                        "file": page[0],
                        "page_number": page[1],
                        "kept_file": earlier[0],
                        "kept_page_number": earlier[1],
                    }
                )
    return failures


def pdf_pages_after_dedup() -> int:
    import csv

    with STATS_PATH.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["stage"] == "after_dedup":
                return int(row["page_count"])
    raise RuntimeError("after_dedup row is missing from corpus_stats.csv")


def write_drug_counts(drugs: list[dict]) -> None:
    import csv

    PUBCHEM_DRUG_COUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PUBCHEM_DRUG_COUNTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "cid", "page_count", "word_count", "own_file", "noncontent_pages", "non_english_pages"],
        )
        writer.writeheader()
        for drug in drugs:
            writer.writerow(
                {
                    "id": drug["id"],
                    "cid": drug["cid"],
                    "page_count": drug["page_count"],
                    "word_count": drug["word_count"],
                    "own_file": drug["own_file"],
                    "noncontent_pages": drug["noncontent_pages"],
                    "non_english_pages": drug["non_english_pages"],
                }
            )


def write_pubchem_stats(
    drugs: list[dict],
    written: list[dict],
    removed_pages: list[dict],
    pubchem_pages: int,
    pubchem_words: int,
) -> None:
    before_pages = sum(drug["page_count"] for drug in drugs)
    before_words = sum(drug["word_count"] for drug in drugs)
    rows = [
        {
            "stage": "before_pubchem_filters",
            "files": len(drugs),
            "section_pages": before_pages,
            "words": before_words,
            "pages_removed": 0,
            "counts_toward_pdf_page_gate": False,
        },
        {
            "stage": "after_pubchem_filters",
            "files": len(written),
            "section_pages": pubchem_pages,
            "words": pubchem_words,
            "pages_removed": len(removed_pages),
            "counts_toward_pdf_page_gate": False,
        },
    ]
    PUBCHEM_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PUBCHEM_STATS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def record_sources(drugs: list[dict], written: list[dict], pubchem_pages: int, pdf_pages: int) -> None:
    manifest = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    manifest["pdf_page_total_from_pdf_files"] = sum(document["page_count"] for document in manifest["documents"])
    manifest["pdf_pages_after_dedup_excluding_pubchem"] = pdf_pages
    manifest["pubchem"] = {
        "counts_toward_minimum_pdf_count": False,
        "counts_toward_pdf_page_gate": False,
        "license": "NCBI PubChem compound summaries. Public-domain US government data. Linked annotations keep their original source terms.",
        "page_definition": "One clinical heading is one section-page. These section-pages are not PDF pages.",
        "artemether_lumefantrine_note": (
            "Combination CID 6450800 has almost no clinical text. "
            "Artemether CID 68911 and lumefantrine CID 6437380 are the two component records."
        ),
        "extra_headings": ["Protein Binding", "Record Description", "GHS Classification"],
        "extra_headings_reason": (
            "The first 13 clinical headings leave doxycycline at 759 words. "
            "Protein Binding, Record Description, and GHS Classification are English "
            "drug-information sections on the same PubChem record and bring that drug "
            "over the 1,000-word document rule."
        ),
        "section_page_count": pubchem_pages,
        "files": [document["file"] for document in written],
        "drugs": [
            {
                "id": drug["id"],
                "cid": drug["cid"],
                "url": drug["url"],
                "section_pages_after_language": drug["page_count"],
                "words_after_language": drug["word_count"],
                "own_file": drug["own_file"],
            }
            for drug in drugs
        ],
    }
    SOURCES_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
