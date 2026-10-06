"""Domain corpus pipeline for Assignment 1B Part A.

A2 extracts each source PDF page by page. Later filters append to the same reports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import fitz
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = ROOT / "data" / "sources.json"
STAGING_DIR = ROOT / "data" / "extracted"
LANGUAGE_DIR = ROOT / "data" / "staged" / "after_language"
LENGTH_DIR = ROOT / "data" / "staged" / "after_length"
DEDUP_DIR = ROOT / "data" / "staged" / "after_dedup"
REPORTS_DIR = ROOT / "reports"
STATS_PATH = REPORTS_DIR / "corpus_stats.csv"
SHORT_PAGES_PATH = REPORTS_DIR / "extraction_short_pages.csv"
NONCONTENT_PATH = REPORTS_DIR / "language_noncontent_pages.csv"
LANGUAGE_REMOVED_PATH = REPORTS_DIR / "language_removed_pages.csv"
LENGTH_REMOVED_PATH = REPORTS_DIR / "length_removed_documents.csv"
DEDUP_REMOVED_PAGES_PATH = REPORTS_DIR / "dedup_removed_pages.csv"
DEDUP_REMOVED_DOCS_PATH = REPORTS_DIR / "dedup_removed_documents.csv"
CHECKS_PATH = REPORTS_DIR / "filter_checks.json"

MIN_LETTERS = 30
MIN_PAGES = 3
MIN_WORDS = 1000
SHINGLE_SIZE = 5
JACCARD_THRESHOLD = 0.90
LENGTH_JUSTIFICATION = (
    "A document is kept only when the text that survived the language filter still has "
    "at least 3 pages and at least 1,000 words. Three pages is the minimum useful document "
    "length. The 1,000-word rule removes a file whose pages are mostly images, covers, or "
    "contents lists and therefore cannot support instruction pairs."
)
PAGE_MARKER = re.compile(r"^<<<PAGE (\d+)>>>$")
STATS_FIELDS = [
    "stage",
    "pdf_count",
    "page_count",
    "word_count",
    "character_count",
    "documents_removed",
    "pages_removed",
    "short_page_count",
    "noncontent_pages_removed",
    "language_pages_removed",
]
STAGE_ORDER = [
    "extracted",
    "before_language",
    "after_language",
    "before_length",
    "after_length",
    "before_dedup",
    "after_dedup",
]


def letter_count(text: str) -> int:
    return sum(character.isalpha() for character in text)


def word_count(text: str) -> int:
    return len(text.split())


def load_sources() -> list[dict]:
    manifest = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    documents = manifest["documents"]
    if len(documents) < 5:
        raise RuntimeError(f"sources.json has {len(documents)} PDFs; expected at least 5")
    return documents


def extract_pdf_pages(pdf_path: Path) -> list[str]:
    document = fitz.open(pdf_path)
    try:
        return [page.get_text("text") for page in document]
    finally:
        document.close()


def render_numbered_pages(pages: list[tuple[int, str]]) -> str:
    blocks = [f"<<<PAGE {number}>>>\n{text.rstrip()}\n" for number, text in pages]
    return "\n".join(blocks)


def render_staged_text(pages: list[str]) -> str:
    return render_numbered_pages(list(enumerate(pages, start=1)))


def detect_language(text: str) -> str:
    DetectorFactory.seed = 0
    try:
        return detect(text)
    except LangDetectException:
        return "undetected"


def parse_staged_text(text: str) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    current_number: int | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_number is None:
            return
        pages.append((current_number, "\n".join(current_lines).strip("\n")))

    for line in text.splitlines():
        match = PAGE_MARKER.match(line)
        if match:
            flush()
            current_number = int(match.group(1))
            current_lines = []
            continue
        if current_number is None and line == "":
            continue
        current_lines.append(line)
    flush()
    return pages


def short_pages_for(document_file: str, pages: list[str]) -> list[dict]:
    rows = []
    for number, text in enumerate(pages, start=1):
        letters = letter_count(text)
        if letters < MIN_LETTERS:
            rows.append(
                {
                    "file": document_file,
                    "page_number": number,
                    "letter_count": letters,
                    "character_count": len(text),
                }
            )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_stats() -> list[dict]:
    if not STATS_PATH.exists():
        return []
    with STATS_PATH.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_stats_row(row: dict) -> dict:
    normalized = {}
    for field in STATS_FIELDS:
        value = row.get(field, 0)
        if field != "stage" and (value == "" or value is None):
            value = 0
        normalized[field] = value
    return normalized


def upsert_stats(rows: list[dict]) -> None:
    by_stage = {row["stage"]: normalize_stats_row(row) for row in read_stats()}
    for row in rows:
        by_stage[row["stage"]] = normalize_stats_row(row)
    ordered = [by_stage[stage] for stage in STAGE_ORDER if stage in by_stage]
    ordered.extend(row for stage, row in by_stage.items() if stage not in STAGE_ORDER)
    write_csv(STATS_PATH, STATS_FIELDS, ordered)


def load_checks() -> dict:
    if not CHECKS_PATH.exists():
        return {}
    return json.loads(CHECKS_PATH.read_text(encoding="utf-8"))


def update_checks(updates: dict) -> dict:
    checks = load_checks()
    checks.update(updates)
    CHECKS_PATH.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    return checks


def totals(pages: list[tuple[int, str]]) -> tuple[int, int, int]:
    texts = [text for _, text in pages]
    return (
        len(texts),
        sum(word_count(text) for text in texts),
        sum(len(text) for text in texts),
    )


def extract_corpus() -> dict:
    documents = load_sources()
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    short_rows: list[dict] = []
    total_words = 0
    total_characters = 0
    total_pages = 0

    for source in documents:
        pdf_path = ROOT / source["file"]
        if not pdf_path.is_file():
            raise FileNotFoundError(pdf_path)
        pages = extract_pdf_pages(pdf_path)
        if len(pages) != source["page_count"]:
            raise RuntimeError(
                f"{source['id']} page count {len(pages)} does not match sources.json {source['page_count']}"
            )
        staged_path = STAGING_DIR / f"{source['id']}.txt"
        staged_path.write_text(render_staged_text(pages), encoding="utf-8")
        parsed = parse_staged_text(staged_path.read_text(encoding="utf-8"))
        if [number for number, _ in parsed] != list(range(1, len(pages) + 1)):
            raise RuntimeError(f"{source['id']} staged page markers do not match the PDF")
        for (_, staged_text), raw_text in zip(parsed, pages):
            if staged_text != raw_text.rstrip():
                raise RuntimeError(f"{source['id']} staged text does not match the PDF extract")
        total_pages += len(pages)
        total_words += sum(word_count(text) for text in pages)
        total_characters += sum(len(text) for text in pages)
        short_rows.extend(short_pages_for(source["file"], pages))

    stats_row = {
        "stage": "extracted",
        "pdf_count": len(documents),
        "page_count": total_pages,
        "word_count": total_words,
        "character_count": total_characters,
        "documents_removed": 0,
        "pages_removed": 0,
        "short_page_count": len(short_rows),
        "noncontent_pages_removed": 0,
        "language_pages_removed": 0,
    }
    upsert_stats([stats_row])
    write_csv(
        SHORT_PAGES_PATH,
        ["file", "page_number", "letter_count", "character_count"],
        short_rows,
    )
    checks = update_checks(
        {
            "extraction_page_counts_match": True,
            "extraction_staged_text_matches_pdf": True,
            "extraction_short_pages_listed": True,
            "min_letters": MIN_LETTERS,
            "pdf_count": len(documents),
            "page_count": total_pages,
            "short_page_count": len(short_rows),
        }
    )
    return {"stats": stats_row, "short_pages": short_rows, "checks": checks}


def apply_language_filter() -> dict:
    documents = load_sources()
    LANGUAGE_DIR.mkdir(parents=True, exist_ok=True)
    for stale in LANGUAGE_DIR.glob("*.txt"):
        stale.unlink()

    noncontent_rows: list[dict] = []
    removed_rows: list[dict] = []
    kept_documents = 0
    kept_pages: list[tuple[int, str]] = []
    before_pages: list[tuple[int, str]] = []

    for source in documents:
        staged_path = STAGING_DIR / f"{source['id']}.txt"
        if not staged_path.is_file():
            raise FileNotFoundError(f"Missing extract for {source['id']}. Run extract first.")
        pages = parse_staged_text(staged_path.read_text(encoding="utf-8"))
        before_pages.extend(pages)
        kept: list[tuple[int, str]] = []
        for number, text in pages:
            letters = letter_count(text)
            if letters < MIN_LETTERS:
                noncontent_rows.append(
                    {
                        "file": source["file"],
                        "page_number": number,
                        "letter_count": letters,
                    }
                )
                continue
            language = detect_language(text)
            if language != "en":
                removed_rows.append(
                    {
                        "file": source["file"],
                        "page_number": number,
                        "letter_count": letters,
                        "detected_language": language,
                    }
                )
                continue
            kept.append((number, text))
        if kept:
            kept_documents += 1
            (LANGUAGE_DIR / f"{source['id']}.txt").write_text(
                render_numbered_pages(kept),
                encoding="utf-8",
            )
            kept_pages.extend(kept)

    failed_recheck: list[dict] = []
    for source in documents:
        kept_path = LANGUAGE_DIR / f"{source['id']}.txt"
        if not kept_path.is_file():
            continue
        for number, text in parse_staged_text(kept_path.read_text(encoding="utf-8")):
            if letter_count(text) < MIN_LETTERS:
                failed_recheck.append(
                    {"file": source["file"], "page_number": number, "detected_language": "too_short"}
                )
                continue
            language = detect_language(text)
            if language != "en":
                failed_recheck.append(
                    {"file": source["file"], "page_number": number, "detected_language": language}
                )

    before_page_count, before_words, before_characters = totals(before_pages)
    after_page_count, after_words, after_characters = totals(kept_pages)
    before_short = sum(1 for _, text in before_pages if letter_count(text) < MIN_LETTERS)
    documents_removed = len(documents) - kept_documents
    before_row = {
        "stage": "before_language",
        "pdf_count": len(documents),
        "page_count": before_page_count,
        "word_count": before_words,
        "character_count": before_characters,
        "documents_removed": 0,
        "pages_removed": 0,
        "short_page_count": before_short,
        "noncontent_pages_removed": 0,
        "language_pages_removed": 0,
    }
    after_row = {
        "stage": "after_language",
        "pdf_count": kept_documents,
        "page_count": after_page_count,
        "word_count": after_words,
        "character_count": after_characters,
        "documents_removed": documents_removed,
        "pages_removed": len(removed_rows),
        "short_page_count": 0,
        "noncontent_pages_removed": len(noncontent_rows),
        "language_pages_removed": len(removed_rows),
    }
    upsert_stats([before_row, after_row])
    write_csv(NONCONTENT_PATH, ["file", "page_number", "letter_count"], noncontent_rows)
    write_csv(
        LANGUAGE_REMOVED_PATH,
        ["file", "page_number", "letter_count", "detected_language"],
        removed_rows,
    )
    checks = update_checks(
        {
            "language_library": "langdetect",
            "language_seed": 0,
            "language_noncontent_pages": len(noncontent_rows),
            "language_pages_removed": len(removed_rows),
            "language_documents_removed": documents_removed,
            "language_all_kept_pages_english": not failed_recheck,
        }
    )
    if failed_recheck:
        raise RuntimeError(f"Language recheck failed for {len(failed_recheck)} kept pages: {failed_recheck[:5]}")
    return {"before": before_row, "after": after_row, "checks": checks}


def document_measure(pages: list[tuple[int, str]]) -> tuple[int, int, int]:
    page_count, words, characters = totals(pages)
    return page_count, words, characters


def length_failure_reason(page_count: int, words: int) -> str:
    short_pages = page_count < MIN_PAGES
    short_words = words < MIN_WORDS
    if short_pages and short_words:
        return f"fewer than {MIN_PAGES} pages and fewer than {MIN_WORDS} words"
    if short_pages:
        return f"fewer than {MIN_PAGES} pages"
    if short_words:
        return f"fewer than {MIN_WORDS} words"
    return ""


def apply_length_filter() -> dict:
    documents = load_sources()
    if not LANGUAGE_DIR.is_dir():
        raise FileNotFoundError("Missing language output. Run language first.")
    LENGTH_DIR.mkdir(parents=True, exist_ok=True)
    for stale in LENGTH_DIR.glob("*.txt"):
        stale.unlink()

    by_id = {source["id"]: source for source in documents}
    staged_paths = sorted(LANGUAGE_DIR.glob("*.txt"), key=lambda path: path.name)
    if not staged_paths:
        raise RuntimeError("Language output has no documents.")

    before_pages: list[tuple[int, str]] = []
    kept_pages: list[tuple[int, str]] = []
    removed_rows: list[dict] = []
    kept_ids: list[str] = []

    for staged_path in staged_paths:
        source = by_id.get(staged_path.stem)
        if source is None:
            raise RuntimeError(f"Language output {staged_path.name} is not in sources.json")
        pages = parse_staged_text(staged_path.read_text(encoding="utf-8"))
        before_pages.extend(pages)
        page_count, words, characters = document_measure(pages)
        reason = length_failure_reason(page_count, words)
        if reason:
            removed_rows.append(
                {
                    "id": source["id"],
                    "file": source["file"],
                    "page_count": page_count,
                    "word_count": words,
                    "character_count": characters,
                    "reason": reason,
                }
            )
            continue
        kept_ids.append(source["id"])
        (LENGTH_DIR / staged_path.name).write_text(render_numbered_pages(pages), encoding="utf-8")
        kept_pages.extend(pages)

    failed_recheck: list[dict] = []
    for kept_id in kept_ids:
        kept_path = LENGTH_DIR / f"{kept_id}.txt"
        pages = parse_staged_text(kept_path.read_text(encoding="utf-8"))
        page_count, words, _ = document_measure(pages)
        reason = length_failure_reason(page_count, words)
        if reason:
            failed_recheck.append({"id": kept_id, "page_count": page_count, "word_count": words, "reason": reason})

    before_page_count, before_words, before_characters = totals(before_pages)
    after_page_count, after_words, after_characters = totals(kept_pages)
    removed_pages = sum(int(row["page_count"]) for row in removed_rows)
    before_row = {
        "stage": "before_length",
        "pdf_count": len(staged_paths),
        "page_count": before_page_count,
        "word_count": before_words,
        "character_count": before_characters,
        "documents_removed": 0,
        "pages_removed": 0,
        "short_page_count": 0,
        "noncontent_pages_removed": 0,
        "language_pages_removed": 0,
    }
    after_row = {
        "stage": "after_length",
        "pdf_count": len(kept_ids),
        "page_count": after_page_count,
        "word_count": after_words,
        "character_count": after_characters,
        "documents_removed": len(removed_rows),
        "pages_removed": removed_pages,
        "short_page_count": 0,
        "noncontent_pages_removed": 0,
        "language_pages_removed": 0,
    }
    upsert_stats([before_row, after_row])
    write_csv(
        LENGTH_REMOVED_PATH,
        ["id", "file", "page_count", "word_count", "character_count", "reason"],
        removed_rows,
    )
    checks = update_checks(
        {
            "length_min_pages": MIN_PAGES,
            "length_min_words": MIN_WORDS,
            "length_threshold_justification": LENGTH_JUSTIFICATION,
            "length_documents_removed": len(removed_rows),
            "length_pages_removed": removed_pages,
            "length_all_kept_documents_pass": not failed_recheck,
        }
    )
    if failed_recheck:
        raise RuntimeError(f"Length recheck failed: {failed_recheck}")
    return {"before": before_row, "after": after_row, "removed": removed_rows, "checks": checks}


def normalize_page(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def word_shingles(normalized: str) -> set[str]:
    words = normalized.split(" ")
    if len(words) < SHINGLE_SIZE:
        return set()
    return {" ".join(words[index : index + SHINGLE_SIZE]) for index in range(len(words) - SHINGLE_SIZE + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    if len(shorter) / len(longer) < JACCARD_THRESHOLD:
        return 0.0
    intersection = sum(1 for item in shorter if item in longer)
    union = len(shorter) + len(longer) - intersection
    if union == 0:
        return 0.0
    return intersection / union


def load_length_pages(documents: list[dict]) -> list[dict]:
    pages: list[dict] = []
    for source in documents:
        staged_path = LENGTH_DIR / f"{source['id']}.txt"
        if not staged_path.is_file():
            continue
        for number, text in parse_staged_text(staged_path.read_text(encoding="utf-8")):
            normalized = normalize_page(text)
            pages.append(
                {
                    "id": source["id"],
                    "file": source["file"],
                    "page_number": number,
                    "text": text,
                    "normalized": normalized,
                    "digest": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                    "shingles": word_shingles(normalized),
                }
            )
    pages.sort(key=lambda page: (page["file"], page["page_number"]))
    return pages


def apply_dedup_filter() -> dict:
    documents = load_sources()
    if not LENGTH_DIR.is_dir():
        raise FileNotFoundError("Missing length output. Run length first.")
    pages = load_length_pages(documents)
    if not pages:
        raise RuntimeError("Length output has no pages.")

    seen_hashes: dict[str, dict] = {}
    after_exact: list[dict] = []
    removed_pages: list[dict] = []
    for page in pages:
        earlier = seen_hashes.get(page["digest"])
        if earlier is not None:
            removed_pages.append(
                {
                    "file": page["file"],
                    "page_number": page["page_number"],
                    "reason": "exact",
                    "kept_file": earlier["file"],
                    "kept_page_number": earlier["page_number"],
                    "similarity": "1.0000",
                }
            )
            continue
        seen_hashes[page["digest"]] = page
        after_exact.append(page)

    kept_pages: list[dict] = []
    for page in after_exact:
        match = None
        score = 0.0
        for earlier in kept_pages:
            score = jaccard(page["shingles"], earlier["shingles"])
            if score >= JACCARD_THRESHOLD:
                match = earlier
                break
        if match is not None:
            removed_pages.append(
                {
                    "file": page["file"],
                    "page_number": page["page_number"],
                    "reason": "near",
                    "kept_file": match["file"],
                    "kept_page_number": match["page_number"],
                    "similarity": f"{score:.4f}",
                }
            )
            continue
        kept_pages.append(page)

    grouped: dict[str, list[dict]] = {}
    for page in kept_pages:
        grouped.setdefault(page["id"], []).append(page)
    documents_by_file = sorted(
        [source for source in documents if source["id"] in {page["id"] for page in pages}],
        key=lambda source: source["file"],
    )
    doc_rows = []
    for source in documents_by_file:
        doc_pages = sorted(grouped.get(source["id"], []), key=lambda page: page["page_number"])
        if not doc_pages:
            continue
        normalized = " ".join(page["normalized"] for page in doc_pages if page["normalized"])
        doc_rows.append(
            {
                "id": source["id"],
                "file": source["file"],
                "pages": doc_pages,
                "word_count": word_count(normalized),
                "shingles": word_shingles(normalized),
            }
        )

    removed_documents: list[dict] = []
    kept_documents: list[dict] = []
    for document in doc_rows:
        match = None
        score = 0.0
        for earlier in kept_documents:
            score = jaccard(document["shingles"], earlier["shingles"])
            if score >= JACCARD_THRESHOLD:
                match = earlier
                break
        if match is None:
            kept_documents.append(document)
            continue
        drop_current = document["word_count"] < match["word_count"] or (
            document["word_count"] == match["word_count"] and document["file"] > match["file"]
        )
        dropped = document if drop_current else match
        kept = match if drop_current else document
        if not drop_current:
            kept_documents.remove(match)
            kept_documents.append(document)
        removed_documents.append(
            {
                "id": dropped["id"],
                "file": dropped["file"],
                "page_count": len(dropped["pages"]),
                "word_count": dropped["word_count"],
                "kept_id": kept["id"],
                "kept_file": kept["file"],
                "similarity": f"{score:.4f}",
                "reason": "document_near_duplicate",
            }
        )
        for page in dropped["pages"]:
            removed_pages.append(
                {
                    "file": page["file"],
                    "page_number": page["page_number"],
                    "reason": "document_near_duplicate",
                    "kept_file": kept["file"],
                    "kept_page_number": "",
                    "similarity": f"{score:.4f}",
                }
            )

    page_survivor_ids = set(grouped)
    input_ids = {page["id"] for page in pages}
    for source in documents_by_file:
        if source["id"] in input_ids and source["id"] not in page_survivor_ids:
            original_pages = [page for page in pages if page["id"] == source["id"]]
            removed_documents.append(
                {
                    "id": source["id"],
                    "file": source["file"],
                    "page_count": len(original_pages),
                    "word_count": 0,
                    "kept_id": "",
                    "kept_file": "",
                    "similarity": "",
                    "reason": "all pages were duplicates",
                }
            )
    dropped_ids = {row["id"] for row in removed_documents}
    DEDUP_DIR.mkdir(parents=True, exist_ok=True)
    for stale in DEDUP_DIR.glob("*.txt"):
        stale.unlink()
    final_pages: list[dict] = []
    for document in sorted(kept_documents, key=lambda item: item["file"]):
        if document["id"] in dropped_ids:
            continue
        ordered_pages = sorted(document["pages"], key=lambda page: page["page_number"])
        (DEDUP_DIR / f"{document['id']}.txt").write_text(
            render_numbered_pages([(page["page_number"], page["text"]) for page in ordered_pages]),
            encoding="utf-8",
        )
        final_pages.extend(ordered_pages)

    failed = verify_dedup_output(documents)
    before_page_count, before_words, before_characters = totals([(page["page_number"], page["text"]) for page in pages])
    after_page_count, after_words, after_characters = totals(
        [(page["page_number"], page["text"]) for page in final_pages]
    )
    before_ids = {page["id"] for page in pages}
    after_ids = {page["id"] for page in final_pages}
    before_row = {
        "stage": "before_dedup",
        "pdf_count": len(before_ids),
        "page_count": before_page_count,
        "word_count": before_words,
        "character_count": before_characters,
        "documents_removed": 0,
        "pages_removed": 0,
        "short_page_count": 0,
        "noncontent_pages_removed": 0,
        "language_pages_removed": 0,
    }
    after_row = {
        "stage": "after_dedup",
        "pdf_count": len(after_ids),
        "page_count": after_page_count,
        "word_count": after_words,
        "character_count": after_characters,
        "documents_removed": len(before_ids - after_ids),
        "pages_removed": before_page_count - after_page_count,
        "short_page_count": 0,
        "noncontent_pages_removed": 0,
        "language_pages_removed": 0,
    }
    upsert_stats([before_row, after_row])
    write_csv(
        DEDUP_REMOVED_PAGES_PATH,
        ["file", "page_number", "reason", "kept_file", "kept_page_number", "similarity"],
        removed_pages,
    )
    write_csv(
        DEDUP_REMOVED_DOCS_PATH,
        ["id", "file", "page_count", "word_count", "kept_id", "kept_file", "similarity", "reason"],
        removed_documents,
    )
    exact_removed = sum(1 for row in removed_pages if row["reason"] == "exact")
    near_removed = sum(1 for row in removed_pages if row["reason"] == "near")
    checks = update_checks(
        {
            "dedup_method": "SHA-256 exact page hash, then Jaccard >= 0.90 on 5-word shingles for pages and whole documents",
            "dedup_jaccard_threshold": JACCARD_THRESHOLD,
            "dedup_shingle_size": SHINGLE_SIZE,
            "dedup_order": "source file name, then page number",
            "dedup_exact_pages_removed": exact_removed,
            "dedup_near_pages_removed": near_removed,
            "dedup_documents_removed": len(before_ids - after_ids),
            "dedup_pages_removed": before_page_count - after_page_count,
            "dedup_no_repeated_hash": not failed["hashes"],
            "dedup_no_page_pair_at_or_above_threshold": not failed["pages"],
            "dedup_no_document_pair_at_or_above_threshold": not failed["documents"],
        }
    )
    if failed["hashes"] or failed["pages"] or failed["documents"]:
        raise RuntimeError(f"Dedup recheck failed: {failed}")
    return {
        "before": before_row,
        "after": after_row,
        "removed_pages": removed_pages,
        "removed_documents": removed_documents,
        "checks": checks,
    }


def verify_dedup_output(documents: list[dict]) -> dict[str, list]:
    by_id = {source["id"]: source for source in documents}
    pages: list[dict] = []
    for staged_path in sorted(DEDUP_DIR.glob("*.txt")):
        source = by_id[staged_path.stem]
        for number, text in parse_staged_text(staged_path.read_text(encoding="utf-8")):
            normalized = normalize_page(text)
            pages.append(
                {
                    "id": source["id"],
                    "file": source["file"],
                    "page_number": number,
                    "normalized": normalized,
                    "digest": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                    "shingles": word_shingles(normalized),
                }
            )
    hash_failures = []
    seen: dict[str, dict] = {}
    for page in pages:
        earlier = seen.get(page["digest"])
        if earlier is not None:
            hash_failures.append(
                {
                    "file": page["file"],
                    "page_number": page["page_number"],
                    "kept_file": earlier["file"],
                    "kept_page_number": earlier["page_number"],
                }
            )
        else:
            seen[page["digest"]] = page
    page_failures = []
    for index, page in enumerate(pages):
        for earlier in pages[:index]:
            score = jaccard(page["shingles"], earlier["shingles"])
            if score >= JACCARD_THRESHOLD:
                page_failures.append(
                    {
                        "file": page["file"],
                        "page_number": page["page_number"],
                        "kept_file": earlier["file"],
                        "kept_page_number": earlier["page_number"],
                        "similarity": round(score, 4),
                    }
                )
    grouped: dict[str, list[dict]] = {}
    for page in pages:
        grouped.setdefault(page["id"], []).append(page)
    doc_sets = []
    for source_id, doc_pages in grouped.items():
        ordered = sorted(doc_pages, key=lambda page: page["page_number"])
        normalized = " ".join(page["normalized"] for page in ordered if page["normalized"])
        doc_sets.append({"id": source_id, "file": by_id[source_id]["file"], "shingles": word_shingles(normalized)})
    document_failures = []
    for index, document in enumerate(doc_sets):
        for earlier in doc_sets[:index]:
            score = jaccard(document["shingles"], earlier["shingles"])
            if score >= JACCARD_THRESHOLD:
                document_failures.append(
                    {
                        "file": document["file"],
                        "kept_file": earlier["file"],
                        "similarity": round(score, 4),
                    }
                )
    return {"hashes": hash_failures, "pages": page_failures, "documents": document_failures}


def main() -> None:
    parser = argparse.ArgumentParser(description="Assignment 1B domain corpus pipeline")
    parser.add_argument("stage", choices=["extract", "language", "length", "dedup"])
    args = parser.parse_args()
    if args.stage == "extract":
        result = extract_corpus()
        stats = result["stats"]
        print(
            "extracted "
            f"pdfs={stats['pdf_count']} pages={stats['page_count']} "
            f"words={stats['word_count']} characters={stats['character_count']} "
            f"short_pages={stats['short_page_count']}"
        )
    elif args.stage == "language":
        result = apply_language_filter()
        after = result["after"]
        print(
            "language "
            f"pdfs={after['pdf_count']} pages={after['page_count']} "
            f"words={after['word_count']} characters={after['character_count']} "
            f"noncontent_removed={after['noncontent_pages_removed']} "
            f"non_english_removed={after['language_pages_removed']} "
            f"documents_removed={after['documents_removed']}"
        )
    elif args.stage == "length":
        result = apply_length_filter()
        after = result["after"]
        print(
            "length "
            f"pdfs={after['pdf_count']} pages={after['page_count']} "
            f"words={after['word_count']} characters={after['character_count']} "
            f"documents_removed={after['documents_removed']} "
            f"pages_removed={after['pages_removed']}"
        )
        for row in result["removed"]:
            print(f"dropped {row['id']} pages={row['page_count']} words={row['word_count']} reason={row['reason']}")
    elif args.stage == "dedup":
        result = apply_dedup_filter()
        after = result["after"]
        print(
            "dedup "
            f"pdfs={after['pdf_count']} pages={after['page_count']} "
            f"words={after['word_count']} characters={after['character_count']} "
            f"documents_removed={after['documents_removed']} "
            f"pages_removed={after['pages_removed']}"
        )
        reasons: dict[str, int] = {}
        for row in result["removed_pages"]:
            reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
        print(f"page_reasons {reasons}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAILED {error}", file=sys.stderr)
        raise
