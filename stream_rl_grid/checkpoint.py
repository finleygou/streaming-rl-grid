"""Atomic, versioned checkpoints for exact continuation of a training stream."""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, Union


CHECKPOINT_FORMAT = "stream-rl-grid"
CHECKPOINT_VERSION = 1


def save_checkpoint(path: Union[str, Path], payload: Dict[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    wrapped = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "payload": payload,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(wrapped, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(destination))
    return destination


def load_checkpoint(path: Union[str, Path]) -> Dict[str, Any]:
    source = Path(path).expanduser().resolve()
    with source.open("rb") as handle:
        wrapped = pickle.load(handle)
    if not isinstance(wrapped, dict) or wrapped.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("This is not a stream-rl-grid checkpoint.")
    if int(wrapped.get("version", -1)) != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version: %r" % wrapped.get("version"))
    payload = wrapped.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload is malformed.")
    return payload
