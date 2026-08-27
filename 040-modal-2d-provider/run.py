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

PROMPT = (
    "a single realistic red apple, centered isolated object, clean neutral studio "
    "background, no text, no extra objects"
)
MODELS = ("sana-sprint-0.6b", "sana-sprint-1.6b")
SEEDS = (42, 73)
RESULTS = Path(__file__).parent / "results"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="040 modal-2D provider verification")
    parser.add_argument("--check-env", action="store_true", help="只验证本地 Python 依赖，不调用远端")
    return parser.parse_args()


def check_env() -> dict[str, object]:
    return {
        "ok": True,
        "modal_version": getattr(modal, "__version__", "unknown"),
        "models": list(MODELS),
    }


def verify_png(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[:8] != PNG_SIGNATURE or data[12:16] != b"IHDR":
        raise AssertionError("artifact is not a valid PNG header")
    width, height = struct.unpack(">II", data[16:24])
    if (width, height) != (1024, 1024):
        raise AssertionError(f"unexpected PNG dimensions: {width}x{height}")
    return width, height


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    client = modal.Client.from_env()
    client.hello()
    capabilities = modal.Function.from_name("modal-2d", "capabilities", client=client).remote()
    assert capabilities["provider"] == "modal-2d"
    assert capabilities["kind"] == "image.generate"
    assert capabilities["operation"] == "modal-2d.image.text_to_image.v1"
    assert capabilities["outputs"] == [{"role": "primary-image", "mediaType": "image/png"}]
    advertised = {item["id"] for item in capabilities["models"]}
    assert set(MODELS) <= advertised

    submit = modal.Function.from_name("modal-2d", "submit", client=client)
    volume = modal.Volume.from_name("modal-2d-artifacts", client=client)
    rows: list[dict[str, object]] = []
    digests: dict[str, list[str]] = {model: [] for model in MODELS}

    for model in MODELS:
        for seed in SEEDS:
            started = time.perf_counter()
            result = submit.remote({"prompt": PROMPT, "model": model, "seed": seed})
            elapsed = round(time.perf_counter() - started, 3)
            artifact = result["artifact"]
            data = b"".join(volume.read_file(artifact["remote_path"]))
            width, height = verify_png(data)
            sha256 = hashlib.sha256(data).hexdigest()
            assert artifact["mediaType"] == "image/png"
            assert artifact["mime"] == "image/png"
            assert artifact["bytes"] == len(data)
            assert artifact["sha256"] == sha256
            assert artifact["digest"] == f"sha256:{sha256}"
            assert artifact["producer"]["provider"] == "modal-2d"
            assert artifact["producer"]["operation"] == "modal-2d.image.text_to_image.v1"
            output = RESULTS / f"{model}-seed-{seed}.png"
            output.write_bytes(data)
            digests[model].append(sha256)
            row = {
                "model": model,
                "seed": seed,
                "elapsedSeconds": elapsed,
                "artifactId": artifact["id"],
                "bytes": len(data),
                "sha256": sha256,
                "dimensions": [width, height],
                "file": str(output.resolve()),
            }
            rows.append(row)
            print(json.dumps({"phase": "candidate", **row}, separators=(",", ":")), flush=True)

    for model, values in digests.items():
        assert len(set(values)) == len(SEEDS), f"{model}: distinct seeds produced identical bytes"

    payload = {
        "status": "passed",
        "prompt": PROMPT,
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "candidates": rows,
    }
    (RESULTS / "latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "candidates": len(rows)}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    args = parse_args()
    if args.check_env:
        print(json.dumps(check_env(), separators=(",", ":")))
        raise SystemExit(0)
    raise SystemExit(main())
