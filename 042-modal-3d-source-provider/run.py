# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "modal==1.5.4",
#   "pillow>=11,<13",
# ]
# ///
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import struct
import time
from pathlib import Path

import modal
from PIL import Image

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
RESULTS = HERE / "results"
SOURCE_RESULT = LAB / "040-modal-2d-provider" / "results" / "latest.json"
BASELINE_RESULT = LAB / "041-modal-3d-provider" / "results" / "latest.json"
MODELS = (
    "fastsam3d-plus-plus",
    "hermit-trellis2-plus-plus",
    "hunyuan2.1-plus-plus",
    "pixal3d",
)


def check_env() -> dict[str, object]:
    return {
        "ok": True,
        "modal_version": getattr(modal, "__version__", "unknown"),
        "pillow_version": getattr(Image, "__version__", "unknown"),
        "source_result_exists": SOURCE_RESULT.is_file(),
        "baseline_result_exists": BASELINE_RESULT.is_file(),
        "models": list(MODELS),
    }


def default_source() -> Path:
    if not SOURCE_RESULT.is_file():
        raise FileNotFoundError("040 result missing; run experiment 040 or pass --source")
    payload = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    for row in payload.get("candidates", []):
        if row.get("model") == "sana-sprint-1.6b" and row.get("seed") == 42:
            path = Path(str(row["file"]))
            if path.is_file():
                return path
    raise FileNotFoundError("040 sana-sprint-1.6b seed 42 artifact is unavailable")


def inspect_source(data: bytes) -> tuple[str, str, tuple[int, int], str]:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        formats = {
            "PNG": ("image/png", ".png"),
            "JPEG": ("image/jpeg", ".jpg"),
            "WEBP": ("image/webp", ".webp"),
        }
        if image.format not in formats:
            raise AssertionError(f"unsupported source format: {image.format}")
        media_type, extension = formats[image.format]
        return media_type, extension, image.size, image.mode


def verify_glb(
    volume: modal.Volume, artifact: dict[str, object], model: str
) -> dict[str, object]:
    destination = RESULTS / f"{model}.glb"
    digest = hashlib.sha256()
    header = bytearray()
    total = 0
    with destination.open("wb") as output:
        for chunk in volume.read_file(str(artifact["path"])):
            if len(header) < 12:
                header.extend(chunk[: 12 - len(header)])
            digest.update(chunk)
            total += len(chunk)
            output.write(chunk)
    if len(header) != 12:
        raise AssertionError("GLB header is truncated")
    magic, version, declared = struct.unpack("<4sII", bytes(header))
    sha256 = digest.hexdigest()
    if magic != b"glTF" or version != 2 or declared != total:
        raise AssertionError(f"invalid GLB for {model}")
    if artifact.get("bytes") != total or artifact.get("sha256") != sha256:
        raise AssertionError(f"artifact descriptor mismatch for {model}")
    return {
        "bytes": total,
        "sha256": sha256,
        "glbVersion": version,
        "file": str(destination.resolve()),
    }


def baseline() -> dict[str, dict[str, object]]:
    if not BASELINE_RESULT.is_file():
        return {}
    payload = json.loads(BASELINE_RESULT.read_text(encoding="utf-8"))
    return {str(row["model"]): row for row in payload.get("models", [])}


def wait_model(
    client: modal.Client,
    volume: modal.Volume,
    model: str,
    call_id: str,
    started: float,
    source_sha256: str,
    baseline_rows: dict[str, dict[str, object]],
) -> dict[str, object]:
    try:
        value = modal.FunctionCall.from_id(call_id, client=client).get(timeout=2400)
        if not isinstance(value, dict) or value.get("model") != model:
            raise AssertionError("worker returned invalid result envelope")
        conditioning = value.get("conditioning")
        if not isinstance(conditioning, dict):
            raise TypeError("conditioning evidence is missing")
        if conditioning.get("strategy") != "birefnet":
            raise AssertionError("unexpected conditioning strategy")
        if conditioning.get("engine") != "birefnet-general-lite":
            raise AssertionError("unexpected conditioning engine")
        if conditioning.get("source_sha256") != source_sha256:
            raise AssertionError("conditioning source digest mismatch")
        canonical_sha = conditioning.get("canonical_sha256")
        if not isinstance(canonical_sha, str) or len(canonical_sha) != 64:
            raise AssertionError("conditioning canonical digest is invalid")
        artifact = value.get("artifact")
        if not isinstance(artifact, dict):
            raise TypeError("worker artifact is missing")
        verified = verify_glb(volume, artifact, model)
        elapsed = round(time.perf_counter() - started, 3)
        old = baseline_rows.get(model, {})
        old_artifact = old.get("artifact") if isinstance(old, dict) else None
        old_bytes = old_artifact.get("bytes") if isinstance(old_artifact, dict) else None
        old_elapsed = old.get("elapsedSeconds") if isinstance(old, dict) else None
        return {
            "model": model,
            "status": "passed",
            "callId": call_id,
            "elapsedSeconds": elapsed,
            "conditioning": {
                "strategy": conditioning.get("strategy"),
                "engine": conditioning.get("engine"),
                "sourceSha256": conditioning.get("source_sha256"),
                "canonicalSha256": canonical_sha,
                "foregroundRatio": conditioning.get("foreground_ratio"),
                "maskElapsedMs": conditioning.get("mask_elapsed_ms"),
            },
            "artifact": verified,
            "baseline041": {
                "bytes": old_bytes,
                "bytesDelta": verified["bytes"] - old_bytes
                if isinstance(old_bytes, int)
                else None,
                "elapsedSeconds": old_elapsed,
                "elapsedDeltaSeconds": round(elapsed - old_elapsed, 3)
                if isinstance(old_elapsed, (int, float))
                else None,
            },
        }
    except (AssertionError, TypeError, ValueError, OSError, RuntimeError, TimeoutError) as exc:
        return {
            "model": model,
            "status": "failed",
            "callId": call_id,
            "elapsedSeconds": round(time.perf_counter() - started, 3),
            "errorType": type(exc).__name__,
            "error": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="042 modal-3D source-image provider parity verification")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--check-env", action="store_true", help="只验证本地依赖/输入前置，不调用远端")
    args = parser.parse_args()
    if args.check_env:
        print(json.dumps(check_env(), separators=(",", ":")))
        return 0

    RESULTS.mkdir(parents=True, exist_ok=True)
    source_path = args.source or default_source()
    source = source_path.read_bytes()
    source_sha = hashlib.sha256(source).hexdigest()
    media_type, extension, dimensions, mode = inspect_source(source)

    client = modal.Client.from_env()
    client.hello()
    gateway = modal.Function.from_name("modal-3d-gateway", "submit", client=client)
    capabilities = modal.Function.from_name(
        "modal-3d-gateway", "capabilities", client=client
    ).remote()
    public_input = capabilities["generation"]["public_input_contract"]
    assert public_input["role"] == "source_image"
    assert public_input["conditioning"] == "provider"
    assert media_type in public_input["mediaTypes"]
    assert len(source) <= public_input["maxBytes"]

    enabled = {
        row["id"]: row
        for row in capabilities["models"]
        if row.get("status") == "enabled"
    }
    assert set(MODELS) <= set(enabled)

    volume = modal.Volume.from_name("modal-3d-artifacts", client=client)
    input_path = f"source-inputs/{source_sha}{extension}"
    with volume.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(source), input_path)

    submissions: dict[str, tuple[str, float]] = {}
    for model in MODELS:
        profile = enabled[model]["profiles"][0]
        options = dict(profile["options"])
        options["seed"] = 42
        started = time.perf_counter()
        first = gateway.remote(model, input_path, options)
        second = gateway.remote(model, input_path, options)
        if first["call_id"] != second["call_id"]:
            raise AssertionError(f"{model}: duplicate submit did not reuse callId")
        call_id = str(first["call_id"])
        submissions[model] = (call_id, started)
        print(
            json.dumps(
                {
                    "phase": "submitted",
                    "model": model,
                    "callId": call_id,
                    "idempotent": True,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    baseline_rows = baseline()
    rows: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        futures = {
            pool.submit(
                wait_model,
                client,
                volume,
                model,
                call_id,
                started,
                source_sha,
                baseline_rows,
            ): model
            for model, (call_id, started) in submissions.items()
        }
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                json.dumps(
                    {"phase": "model-result", **row}, separators=(",", ":")
                ),
                flush=True,
            )

    rows.sort(key=lambda row: MODELS.index(str(row["model"])))
    passed = [row for row in rows if row["status"] == "passed"]
    canonical = {row["conditioning"]["canonicalSha256"] for row in passed}
    if len(passed) == len(MODELS) and len(canonical) != 1:
        raise AssertionError(
            f"same source produced multiple canonical digests: {sorted(canonical)}"
        )

    payload = {
        "status": "passed" if len(passed) == len(MODELS) else "failed",
        "source": {
            "path": str(source_path.resolve()),
            "mediaType": media_type,
            "bytes": len(source),
            "sha256": source_sha,
            "dimensions": list(dimensions),
            "mode": mode,
            "inputPath": input_path,
        },
        "publicInputContract": public_input,
        "canonicalSha256": next(iter(canonical)) if len(canonical) == 1 else None,
        "models": rows,
    }
    (RESULTS / "latest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": payload["status"], "passed": len(passed), "models": len(MODELS)},
            separators=(",", ":"),
        )
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
