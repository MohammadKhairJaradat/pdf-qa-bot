"""Batch PDF ingestion CLI with SHA-256 manifest idempotency and CSV/Parquet export."""

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from qa_bot import InputError, compute_file_sha256, ingest_document

logger = logging.getLogger(__name__)


def collect_pdfs(input_path):
    """Collect a single PDF file or all PDFs inside a directory."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise InputError(f"Only PDF files are supported: {path}")
        return [path]
    return sorted(p for p in path.rglob("*.pdf") if p.is_file())


def load_json(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def load_existing_chunks(csv_path):
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                {
                    "source": row["source"],
                    "file_sha256": row["file_sha256"],
                    "page": int(row["page"]),
                    "chunk_index": int(row["chunk_index"]),
                    "char_count": int(row["char_count"]),
                    "word_count": int(row["word_count"]),
                    "text": row["text"],
                }
            )
        return rows


def save_chunks_catalog(rows, csv_path, parquet_path):
    fieldnames = [
        "source",
        "file_sha256",
        "page",
        "chunk_index",
        "char_count",
        "word_count",
        "text",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    parquet_written = False
    try:
        import pandas as pd

        df = pd.DataFrame(rows, columns=fieldnames)
        df.to_parquet(parquet_path, index=False)
        parquet_written = True
    except Exception as exc:
        logger.debug("Parquet export skipped (%s); CSV catalog written.", exc)
    return parquet_written


def run_batch_ingestion(input_path, output_dir="data/processed", force=False):
    """Ingest a PDF or folder of PDFs with SHA-256 manifest idempotency."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    csv_path = out_dir / "chunks.csv"
    parquet_path = out_dir / "chunks.parquet"
    summary_path = out_dir / "ingestion_summary.json"

    manifest = {} if force else load_json(manifest_path, {})
    existing_rows = [] if force else load_existing_chunks(csv_path)

    pdf_files = collect_pdfs(input_path)
    kept_rows = []
    file_summaries = []
    files_ingested = 0
    files_skipped = 0
    files_failed = 0

    for pdf_path in pdf_files:
        rel_key = pdf_path.name
        file_hash = compute_file_sha256(pdf_path)

        prev_entry = manifest.get(rel_key)
        if (
            not force
            and prev_entry
            and prev_entry.get("file_sha256") == file_hash
        ):
            logger.info("Skipping unchanged file %s (sha256=%s...)", rel_key, file_hash[:12])
            files_skipped += 1
            kept_rows.extend(
                row for row in existing_rows if row["file_sha256"] == file_hash and row["source"] == rel_key
            )
            cached_summary = dict(prev_entry)
            cached_summary["status"] = "skipped_unchanged"
            file_summaries.append(cached_summary)
            continue

        try:
            chunks, doc_summary = ingest_document(pdf_path)
            for idx, chunk in enumerate(chunks, start=1):
                text = chunk.page_content
                kept_rows.append(
                    {
                        "source": doc_summary["source"],
                        "file_sha256": doc_summary["file_sha256"],
                        "page": int(chunk.metadata["page"]),
                        "chunk_index": idx,
                        "char_count": len(text),
                        "word_count": len(text.split()),
                        "text": text,
                    }
                )
            manifest[rel_key] = {
                **doc_summary,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
            entry_out = dict(doc_summary)
            entry_out["status"] = "ingested"
            file_summaries.append(entry_out)
            files_ingested += 1
        except InputError as err:
            logger.warning("Validation failed for %s: %s", rel_key, err)
            file_summaries.append(
                {
                    "source": rel_key,
                    "file_sha256": file_hash,
                    "status": "failed",
                    "error": str(err),
                }
            )
            files_failed += 1

    parquet_written = save_chunks_catalog(kept_rows, csv_path, parquet_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_dir": str(out_dir),
        "files_discovered": len(pdf_files),
        "files_ingested": files_ingested,
        "files_skipped_unchanged": files_skipped,
        "files_failed": files_failed,
        "total_chunks_in_catalog": len(kept_rows),
        "parquet_exported": parquet_written,
        "files": file_summaries,
    }
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "Batch complete: %d ingested, %d skipped, %d failed (%d total chunks)",
        files_ingested,
        files_skipped,
        files_failed,
        len(kept_rows),
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ingest a PDF or folder of PDFs into a validated chunk catalog."
    )
    parser.add_argument(
        "input_path",
        help="Path to a PDF file or directory of PDFs (e.g. sample_data/)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for chunks.csv, chunks.parquet, manifest.json, and ingestion_summary.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest files even if their SHA-256 matches the manifest",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run_batch_ingestion(args.input_path, args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
