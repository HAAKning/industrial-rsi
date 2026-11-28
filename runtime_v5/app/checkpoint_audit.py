"""Verify the current host parent and fingerprint the complete resumed state."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def snapshot(model):
    def arrays(group):
        return canonical_digest({k: {"shape": list(v.shape), "dtype": str(v.dtype),
            "sha256": hashlib.sha256(v.tobytes()).hexdigest()} for k, v in group.items()})
    return {"metadata": dict(model.meta), "weights_sha256": arrays(model.params),
            "adam_first_sha256": arrays(model.first), "adam_second_sha256": arrays(model.second),
            "normalization_sha256": arrays({"mean": model.mean, "scale": model.scale})}


def verified_parent(rnd, destination):
    parent = rnd.parent
    paths = parent.descriptor.get("checkpoint_paths")
    if not isinstance(paths, list) or len(paths) != 1:
        raise ValueError("PARENT_CHECKPOINT_AMBIGUOUS: this official MLP requires one declared checkpoint")
    relative = paths[0]
    # Always use the official SDK. PARENT_CHECKPOINT is never an authority.
    source = parent.path(relative)
    files = parent.receipt.get("files", [])
    matches = [f for f in files if f.get("path") == relative]
    if len(matches) != 1:
        raise ValueError("PARENT_MANIFEST_INVALID: checkpoint must appear exactly once")
    item = matches[0]
    declared = "sha256:" + item["sha256"].removeprefix("sha256:")
    if source.stat().st_size != item.get("size_bytes"):
        raise ValueError("PARENT_SIZE_MISMATCH")
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError("PARENT_COPY_EXISTS")
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    actual = digest(destination)
    if actual != declared:
        raise ValueError("PARENT_COPY_DIGEST_MISMATCH")
    destination.chmod(0o400)
    return {"parent_artifact_id": parent.artifact_id,
            "platform_parent_checkpoint_sha256": parent.checkpoint_sha256,
            "parent_artifact_sha256": parent.receipt.get("artifact_sha256"),
            "checkpoint_relative_path": relative, "declared_file_sha256": declared,
            "read_checkpoint_sha256": actual, "size_bytes": item["size_bytes"]}


def validate_model(model, config, task_contract, train, validation):
    key = config["profile"]
    m = model.meta
    expected_dim = {"A": 128, "B": 34, "C": 24}[key]
    expected_task = "classification" if key == "A" else "regression"
    if (task_contract.get("dataset_key", key) != key
            or task_contract.get("task") != expected_task
            or config["task"] != expected_task or m.get("profile") != key
            or m["task"] != expected_task or m["classes"] != config.get("classes", [])
            or len(m["paths"]) != expected_dim or m["paths"] != train["paths"]
            or m["paths"] != validation["paths"] or m["hidden"] != config["hidden"]
            or m["seed"] != config["seed"]):
        raise ValueError("PARENT_SCHEMA_MISMATCH: task/profile/features/classes/architecture/seed")
    if key == "B" and (m.get("target_unit") != "minutes" or config.get("target_unit") != "minutes"):
        raise ValueError("PARENT_UNIT_MISMATCH: B requires minutes")
    # Official C checkpoints do not store target_unit; C profile identifies cycles.
    unit = {"A": None, "B": "minutes", "C": "cycles"}[key]
    if m.get("target_unit", unit) != unit or task_contract.get("target_unit", unit) != unit:
        raise ValueError("PARENT_UNIT_MISMATCH")
    if set(train["ids"]) & set(validation["ids"]):
        raise ValueError("PUBLIC_SAMPLE_OVERLAP")
    prediction = model.predict(validation["x"][:8])
    return {"state": snapshot(model), "initial_prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(),
            "initial_predictions": prediction.tolist(), "target_unit": unit}
