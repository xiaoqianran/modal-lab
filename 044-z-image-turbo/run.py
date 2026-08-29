# /// script
# requires-python = ">=3.12"
# dependencies = ["modal==1.5.4"]
# ///
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

import modal

MODEL = 'z-image-turbo'
PROMPT = "a red fox sitting in a snowy forest, cinematic photo, no text"
SEED = 42
RESULTS = Path(__file__).parent / "results"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='044-z-image-turbo Z-Image-Turbo production smoke')
    parser.add_argument("--check-env", action="store_true")
    return parser.parse_args()


def verify_png(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[:8] != PNG_SIGNATURE or data[12:16] != b"IHDR":
        raise AssertionError("artifact is not a valid PNG")
    width, height = struct.unpack(">II", data[16:24])
    if (width, height) != (1024, 1024):
        raise AssertionError(f"unexpected PNG dimensions: {width}x{height}")
    return width, height


def main() -> int:
    args = parse_args()
    if args.check_env:
        print(json.dumps({"ok": True, "model": MODEL, "modal": getattr(modal, "__version__", "unknown")}))
        return 0

    client = modal.Client.from_env()
    client.hello()
    capability = modal.Function.from_name("modal-2d", "capabilities", client=client).remote()
    model = next((item for item in capability["models"] if item["id"] == MODEL), None)
    if model is None:
        raise AssertionError(f"{MODEL} is not advertised by modal-2D")
    route = model["generation_entrypoint"]
    worker = modal.Cls.from_name(route["app"], route["class_name"], client=client)(model_id=MODEL)
    method = getattr(worker, route["generate_method"])

    started = time.perf_counter()
    result = method.remote({"prompt": PROMPT, "model": MODEL, "seed": SEED})
    elapsed = round(time.perf_counter() - started, 3)
    artifact = result["artifact"]
    data = modal.Function.from_name("modal-2d", "read_artifact", client=client).remote(artifact["id"])
    width, height = verify_png(data)
    sha256 = hashlib.sha256(data).hexdigest()
    if artifact["sha256"] != sha256 or artifact["digest"] != f"sha256:{sha256}":
        raise AssertionError("artifact hash mismatch")
    if artifact["bytes"] != len(data):
        raise AssertionError("artifact byte count mismatch")

    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / f"{MODEL}-seed-{SEED}.png"
    output.write_bytes(data)
    payload = {
        "status": "passed",
        "model": MODEL,
        "seed": SEED,
        "worker": route,
        "elapsedSeconds": elapsed,
        "artifactId": artifact["id"],
        "bytes": len(data),
        "sha256": sha256,
        "dimensions": [width, height],
    }
    (RESULTS / "latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
