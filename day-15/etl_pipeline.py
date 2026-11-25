from __future__ import annotations
import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import pandas as pd
import yaml


class PipelineError(Exception):
    """Base exception for ETL pipeline errors."""


class ConfigError(PipelineError):
    """Invalid or unavailable configuration."""


class MissingInputFileError(PipelineError):
    """A configured input CSV is missing."""


class SchemaError(PipelineError):
    """Required source columns are missing."""


class RejectionThresholdError(PipelineError):
    """Rejected-row percentage exceeds the configured limit."""


def load_config(config_path: str | Path = "config.yaml") -> tuple[dict[str, Any], Path]:
    path = Path(config_path).resolve()
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError("config.yaml must contain a YAML mapping.")
    if not isinstance(config.get("datasets"), dict):
        raise ConfigError("config.yaml must contain a 'datasets' mapping.")
    return config, path.parent


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def make_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    logger = logging.getLogger(f"olist_etl_{stamp}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_dir / f"etl_{stamp}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"processed": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise PipelineError(f"Cannot read manifest {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("processed", {}), dict):
        raise ConfigError("Manifest must contain a 'processed' mapping.")
    data.setdefault("processed", {})
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def validate_and_map(frame: pd.DataFrame, dataset_name: str, rules: dict[str, Any]):
    required = rules.get("required_columns", [])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SchemaError(f"{dataset_name}: missing required source columns: {missing}")

    mapped = frame.rename(columns=rules.get("column_mapping", {})).copy()
    invalid = pd.Series(False, index=mapped.index)
    reasons = {idx: [] for idx in mapped.index}

    def mark(mask: pd.Series, message: str):
        for idx in mapped.index[mask.fillna(True)]:
            invalid.loc[idx] = True
            reasons[idx].append(message)

    mapping = rules.get("column_mapping", {})
    def output_column(source_column):
        return mapping.get(source_column, source_column)

    for source_column in rules.get("id_columns", []):
        column = output_column(source_column)
        if column in mapped:
            mark(mapped[column].astype("string").str.strip().eq(""), f"{source_column} is empty")
    for source_column in rules.get("numeric_columns", []):
        column = output_column(source_column)
        if column in mapped:
            mark(pd.to_numeric(mapped[column], errors="coerce").isna(), f"{source_column} must be numeric")
    for source_column in rules.get("positive_columns", []):
        column = output_column(source_column)
        if column in mapped:
            values = pd.to_numeric(mapped[column], errors="coerce")
            mark(values.le(0) | values.isna(), f"{source_column} must be greater than zero")
    for source_column, allowed in rules.get("allowed_values", {}).items():
        column = output_column(source_column)
        if column in mapped:
            mark(~mapped[column].isin(allowed), f"{source_column} has an unsupported value")
    for source_column, bounds in rules.get("allowed_ranges", {}).items():
        column = output_column(source_column)
        if column in mapped:
            values = pd.to_numeric(mapped[column], errors="coerce")
            mark(values.lt(bounds[0]) | values.gt(bounds[1]) | values.isna(),
                 f"{source_column} must be between {bounds[0]} and {bounds[1]}")
    for source_column in rules.get("zip_columns", []):
        column = output_column(source_column)
        if column in mapped:
            values = mapped[column].astype("string").str.strip()
            mark(~values.str.fullmatch(r"\d{5}"), f"{source_column} must contain five digits")

    bad = mapped.loc[invalid].copy()
    if not bad.empty:
        bad["rejection_reasons"] = ["; ".join(reasons[idx]) for idx in bad.index]
    return mapped.loc[~invalid].reset_index(drop=True), bad.reset_index(drop=True)


def process_dataset(name, rules, input_dir, output_dir, bad_dir, threshold, logger):
    source = input_dir / rules["file"]
    if not source.is_file():
        raise MissingInputFileError(f"Input file not found: {source}")
    logger.info("Reading %s", source)
    frame = pd.read_csv(source, dtype="string", keep_default_na=False)
    clean, bad = validate_and_map(frame, name, rules)

    target = output_dir / name
    target.mkdir(parents=True, exist_ok=True)
    clean.to_csv(target / f"{name}_clean.csv", index=False)
    clean.to_json(target / f"{name}_clean.json", orient="records", indent=2, force_ascii=False)
    if not bad.empty:
        bad_dir.mkdir(parents=True, exist_ok=True)
        bad.to_csv(bad_dir / f"{name}_bad_records.csv", index=False)

    total = len(frame)
    rejected = len(bad)
    pct = rejected / total * 100 if total else 0.0
    result = {"dataset": name, "source_file": source.name, "total_rows": total,
              "valid_rows": len(clean), "rejected_rows": rejected,
              "rejection_percent": round(pct, 2), "status": "success"}
    if pct > threshold:
        raise RejectionThresholdError(
            f"{name}: {pct:.2f}% rejected; configured limit is {threshold:.2f}%"
        )
    logger.info("%s complete: %s valid, %s rejected", name, len(clean), rejected)
    return result


def run_pipeline(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    started = time.perf_counter()
    config, root = load_config(config_path)
    input_dir = resolve_path(root, config.get("input_dir", "."))
    output_dir = resolve_path(root, config.get("output_dir", "output"))
    bad_dir = resolve_path(root, config.get("bad_records_dir", "bad_records"))
    log_dir = resolve_path(root, config.get("log_dir", "logs"))
    manifest_path = resolve_path(root, config.get("manifest_path", "output/manifest.json"))
    logger = make_logger(log_dir)
    manifest = read_manifest(manifest_path)
    threshold = float(config.get("max_rejection_percent", 20.0))
    processed, skipped, failed = [], [], []

    for name, rules in config["datasets"].items():
        try:
            source = input_dir / rules["file"]
            if not source.is_file():
                raise MissingInputFileError(f"Input file not found: {source}")
            source_hash = sha256_file(source)
            config_hash = hashlib.sha256(json.dumps(rules, sort_keys=True).encode()).hexdigest()
            old = manifest["processed"].get(name, {})
            if old.get("status") == "success" and old.get("source_hash") == source_hash and old.get("config_hash") == config_hash:
                skipped.append(name)
                logger.info("Skipping unchanged dataset %s", name)
                continue
            result = process_dataset(name, rules, input_dir, output_dir, bad_dir, threshold, logger)
            processed.append(result)
            manifest["processed"][name] = {
                "source_hash": source_hash, "config_hash": config_hash,
                "status": "success", "processed_at": datetime.now(UTC).isoformat()
            }
        except (PipelineError, OSError, pd.errors.ParserError, ValueError, KeyError) as exc:
            logger.exception("Failed dataset %s: %s", name, exc)
            failed.append({"dataset": name, "error": str(exc)})
            manifest["processed"].pop(name, None)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    summary = {"run_id": run_id, "processed_count": len(processed),
               "skipped_count": len(skipped), "failed_count": len(failed),
               "processed": processed, "skipped": skipped, "failed": failed,
               "duration_seconds": round(time.perf_counter() - started, 3)}
    write_json(manifest_path, manifest)
    write_json(output_dir / "summaries" / f"run_summary_{run_id}.json", summary)
    logger.info("Pipeline finished: %s", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run_pipeline(), indent=2))
