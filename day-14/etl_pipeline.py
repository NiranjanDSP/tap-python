"""Reusable validation and ETL pipeline for the Olist CSV datasets."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


class SchemaError(Exception):
    """Raised when a CSV does not contain all required columns."""


class FileFormatError(Exception):
    """Raised when an input CSV is missing, empty, or unreadable."""


class RejectionThresholdError(Exception):
    """Raised when a dataset exceeds the configured rejection rate."""


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Read ETL paths, thresholds, dataset rules, and mappings from YAML."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError("config.yaml must contain a YAML mapping.")
    return config


def setup_logger(log_dir: str) -> tuple[logging.Logger, str]:
    """Create a console logger and a unique timestamped log file."""
    folder = Path(log_dir)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    log_path = folder / f"etl_run_{stamp}.log"

    logger = logging.getLogger("olist_etl")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    for handler in (
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger, str(log_path)


def file_hash(path: Path) -> str:
    """Calculate a SHA-256 hash so unchanged source files can be skipped."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[str, Any]:
    """Read the manifest, or return an empty manifest on the first run."""
    if not path.exists():
        return {"processed": {}}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return manifest if isinstance(manifest, dict) else {"processed": {}}
    except (json.JSONDecodeError, OSError):
        return {"processed": {}}


def write_json(path: Path, data: Any) -> None:
    """Write a JSON file, creating its parent folder when required."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def validate_rows(
    frame: pd.DataFrame,
    dataset_name: str,
    rules: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return clean rows and rejected rows with a reason for each rejection."""
    mapping = rules.get("column_mapping", {})
    if mapping:
        frame = frame.rename(columns=mapping)

    required = rules.get("required_columns", [])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SchemaError(f"{dataset_name}: missing columns {missing}")

    data = frame.copy().fillna("")
    row_errors: list[list[str]] = [[] for _ in range(len(data))]

    def add_error(mask: pd.Series, message: str) -> None:
        for position, failed in enumerate(mask.tolist()):
            if failed:
                row_errors[position].append(message)

    for column in required:
        add_error(
            data[column].astype(str).str.strip().eq(""),
            f"Empty required value: {column}",
        )

    id_pattern = re.compile(r"^[0-9a-fA-F]{32}$")
    for column in rules.get("id_columns", []):
        add_error(
            ~data[column].astype(str).str.fullmatch(id_pattern),
            f"Invalid 32-character hexadecimal ID: {column}",
        )

    zip_pattern = re.compile(r"^[0-9]{5}$")
    for column in rules.get("zip_columns", []):
        add_error(
            ~data[column].astype(str).str.fullmatch(zip_pattern),
            f"ZIP prefix must contain 5 digits: {column}",
        )

    for column in rules.get("numeric_columns", []):
        numeric = pd.to_numeric(data[column], errors="coerce")
        add_error(numeric.isna(), f"Must be numeric: {column}")

    for column in rules.get("positive_columns", []):
        numeric = pd.to_numeric(data[column], errors="coerce")
        add_error(
            numeric.isna() | numeric.le(0),
            f"Must be greater than zero: {column}",
        )

    for column in rules.get("nonnegative_columns", []):
        numeric = pd.to_numeric(data[column], errors="coerce")
        add_error(
            numeric.isna() | numeric.lt(0),
            f"Must be zero or greater: {column}",
        )

    for column in rules.get("date_columns", []):
        parsed = pd.to_datetime(data[column], errors="coerce")
        add_error(parsed.isna(), f"Invalid date: {column}")

    if "allowed_statuses" in rules:
        status = data["order_status"].astype(str).str.strip().str.lower()
        allowed = {value.lower() for value in rules["allowed_statuses"]}
        add_error(~status.isin(allowed), "Invalid order_status")

    rejected_mask = pd.Series(
        [bool(errors) for errors in row_errors],
        index=data.index,
    )
    clean = data.loc[~rejected_mask].copy()
    rejected = data.loc[rejected_mask].copy()

    if not rejected.empty:
        rejected.insert(
            0,
            "source_row_number",
            [int(position) + 2 for position in range(len(data)) if row_errors[position]],
        )
        rejected["reason"] = [
            "; ".join(errors) for errors in row_errors if errors
        ]

    return clean, rejected


def process_dataset(
    name: str,
    source: Path,
    rules: dict[str, Any],
    output_dir: Path,
    threshold: float,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Load, validate, save clean/rejected outputs, and enforce the threshold."""
    if not source.exists():
        raise FileFormatError(f"Input file not found: {source}")
    if source.stat().st_size == 0:
        raise FileFormatError(f"Input file is empty: {source}")

    try:
        frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise FileFormatError(f"Cannot read {source}: {error}") from error

    if frame.empty:
        raise FileFormatError(f"CSV contains headers but no data rows: {source}")

    clean, rejected = validate_rows(frame, name, rules)
    total = len(frame)
    rejected_count = len(rejected)
    rejection_rate = rejected_count / total * 100

    output_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_dir / f"{name}_clean.csv", index=False)
    clean.to_json(
        output_dir / f"{name}_clean.json",
        orient="records",
        indent=2,
        force_ascii=False,
    )
    bad_path = output_dir / f"{name}_bad_records.csv"
    if rejected.empty:
        bad_path.unlink(missing_ok=True)
    else:
        rejected.to_csv(bad_path, index=False)

    result = {
        "dataset": name,
        "total_rows": total,
        "valid_rows": len(clean),
        "rejected_rows": rejected_count,
        "rejection_percent": round(rejection_rate, 2),
    }
    logger.info(
        "%s | total=%d valid=%d rejected=%d rejection=%.2f%%",
        name,
        total,
        len(clean),
        rejected_count,
        rejection_rate,
    )

    if rejection_rate > threshold:
        raise RejectionThresholdError(
            f"{name}: rejection rate {rejection_rate:.2f}% "
            f"exceeds limit {threshold:.2f}%"
        )
    return result


def run_pipeline(config_path: str = "config.yaml") -> dict[str, Any]:
    """Run all configured datasets and skip unchanged successful inputs."""
    start = time.perf_counter()
    config = load_config(config_path)
    logger, log_path = setup_logger(config.get("log_dir", "logs"))

    input_dir = Path(config.get("input_dir", "olist_mini"))
    output_dir = Path(config.get("output_dir", "output"))
    summary_dir = Path(config.get("summary_dir", "output/summaries"))
    manifest_path = Path(config.get("manifest_path", "output/manifest.json"))
    manifest = read_manifest(manifest_path)
    processed_manifest = manifest.setdefault("processed", {})

    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    processed: list[dict[str, Any]] = []
    skipped: list[str] = []
    failed: list[dict[str, str]] = []

    logger.info("Starting ETL run %s", run_id)

    for name, rules in config.get("datasets", {}).items():
        source = input_dir / rules.get("file", f"{name}.csv")
        if not source.exists():
            logger.warning("Skipping missing optional file: %s", source)
            skipped.append(name)
            continue

        try:
            digest = file_hash(source)
            old_entry = processed_manifest.get(name, {})
            if (
                old_entry.get("file_hash") == digest
                and old_entry.get("config_hash") == config_hash
                and old_entry.get("status") == "success"
            ):
                logger.info("Skipping unchanged file: %s", name)
                skipped.append(name)
                continue

            result = process_dataset(
                name,
                source,
                rules,
                output_dir,
                float(config.get("max_rejection_percent", 20.0)),
                logger,
            )
            processed.append(result)
            processed_manifest[name] = {
                "file": str(source),
                "file_hash": digest,
                "config_hash": config_hash,
                "status": "success",
                "last_run_id": run_id,
            }
        except (
            SchemaError,
            FileFormatError,
            RejectionThresholdError,
            OSError,
            ValueError,
        ) as error:
            logger.error("%s failed: %s", name, error)
            failed.append({"dataset": name, "error": str(error)})

    duration = round(time.perf_counter() - start, 3)
    summary = {
        "run_id": run_id,
        "duration_seconds": duration,
        "processed_count": len(processed),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "log_file": log_path,
    }
    write_json(
        summary_dir / f"run_summary_{run_id}.json",
        summary,
    )
    write_json(manifest_path, manifest)
    logger.info(
        "Finished run: processed=%d skipped=%d failed=%d duration=%.3fs",
        len(processed),
        len(skipped),
        len(failed),
        duration,
    )

    for handler in logger.handlers[:]:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)

    return summary
