from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import os
import struct
import sys
import time
from pathlib import Path

import modal
from PIL import Image

LAB = Path(__file__).resolve().parent
SOURCE_RESULT = LAB.parent / "040-modal-2d-provider" / "results" / "latest.json"
RESULTS = LAB / "results"
MODELS = (
    "fastsam3d-plus-plus",
    "hermit-trellis2-plus-plus",
    "hunyuan2.1-plus-plus",
    "pixal3d",
)


def wait_for_source(timeout_seconds: int = 900) -> Path:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if SOURCE_RESULT.is_file():
            payload = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
            for row in payload.get("candidates", []):
                if row.get("model") == "sana-sprint-1.6b" and row.get("seed") == 42:
                    path = Path(row["file"])
                    if path.is_file():
                        return path
        time.sleep(2)
    raise TimeoutError("040 modal-2D result did not become available")


def verify_canonical(data: bytes) -> tuple[int, int, tuple[int, int]]:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGBA" or image.size != (1024, 1024):
            raise AssertionError(f"invalid canonical image: {image.format}/{image.mode}/{image.size}")
        alpha = image.getchannel("A")
        extrema = alpha.getextrema()
        if extrema == (255, 255) or extrema == (0, 0):
            raise AssertionError(f"canonical alpha is degenerate: {extrema}")
        return image.width, image.height, extrema


def upload(volume: modal.Volume, data: bytes, sha256: str) -> str:
    path = f"client-inputs/{sha256}.png"
    with volume.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(data), path)
    return path


def verify_glb(volume: modal.Volume, artifact: dict[str, object], model: str) -> dict[str, object]:
    path = str(artifact["path"])
    destination = RESULTS / f"{model}.glb"
    digest = hashlib.sha256()
    total = 0
    header = bytearray()
    with destination.open("wb") as stream:
        for chunk in volume.read_file(path):
            if len(header) < 12:
                header.extend(chunk[: 12 - len(header)])
            digest.update(chunk)
            total += len(chunk)
            stream.write(chunk)
    if len(header) != 12:
        raise AssertionError("GLB header is truncated")
    magic, version, declared = struct.unpack("<4sII", bytes(header))
    sha256 = digest.hexdigest()
    if magic != b"glTF" or version != 2 or declared != total:
        raise AssertionError(f"invalid GLB: magic={magic!r} version={version} declared={declared} actual={total}")
    if total != artifact["bytes"] or sha256 != artifact["sha256"]:
        raise AssertionError("GLB descriptor does not match content")
    return {
        "bytes": total,
        "sha256": sha256,
        "glbVersion": version,
        "file": str(destination.resolve()),
    }


def wait_model(client: modal.Client, volume: modal.Volume, model: str, call_id: str, started: float) -> dict[str, object]:
    try:
        value = modal.FunctionCall.from_id(call_id, client=client).get(timeout=2400)
        if not isinstance(value, dict) or value.get("model") != model:
            raise AssertionError("worker returned invalid result envelope")
        artifact = value.get("artifact")
        if not isinstance(artifact, dict):
            raise AssertionError("worker did not return an artifact")
        verified = verify_glb(volume, artifact, model)
        return {
            "model": model,
            "status": "passed",
            "callId": call_id,
            "elapsedSeconds": round(time.perf_counter() - started, 3),
            "artifact": verified,
        }
    except Exception as exc:
        return {
            "model": model,
            "status": "failed",
            "callId": call_id,
            "elapsedSeconds": round(time.perf_counter() - started, 3),
            "errorType": type(exc).__name__,
            "error": str(exc),
        }


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    source = wait_for_source()
    source_bytes = source.read_bytes()

    workspace = Path(os.environ.get("AGENTSCAPE_ROOT", "/workspace/wk/AgentScape"))
    hub = workspace / "providers" / "modal" / "inference-hub"
    sys.path.insert(0, str(hub))
    from agent import modal_client, rembg_preprocess  # noqa: PLC0415

    client = modal.Client.from_env()
    client.hello()
    modal_client._client = client

    preprocess_started = time.perf_counter()
    preprocess = rembg_preprocess._cloud_process(source_bytes)
    canonical = preprocess["canonical_bytes"]
    width, height, alpha_extrema = verify_canonical(canonical)
    canonical_sha = hashlib.sha256(canonical).hexdigest()
    canonical_file = RESULTS / "canonical.png"
    canonical_file.write_bytes(canonical)
    preprocess_evidence = {
        "elapsedSeconds": round(time.perf_counter() - preprocess_started, 3),
        "execution": preprocess["execution"],
        "engine": preprocess["engine"],
        "provider": preprocess["provider"],
        "foregroundRatio": preprocess["foreground_ratio"],
        "componentCount": preprocess["component_count"],
        "canonicalSha256": canonical_sha,
        "dimensions": [width, height],
        "alphaExtrema": list(alpha_extrema),
    }
    print(json.dumps({"phase": "preprocess", **preprocess_evidence}, separators=(",", ":")), flush=True)

    volume = modal.Volume.from_name("modal-3d-artifacts", client=client)
    input_path = upload(volume, canonical, canonical_sha)
    capabilities = modal.Function.from_name("modal-3d-gateway", "capabilities", client=client).remote()
    enabled = {item["id"]: item for item in capabilities["models"] if item.get("status") == "enabled"}
    assert set(MODELS) <= set(enabled)
    gateway = modal.Function.from_name("modal-3d-gateway", "submit", client=client)

    submissions: dict[str, tuple[str, float]] = {}
    idempotency: dict[str, bool] = {}
    for model in MODELS:
        profile = enabled[model]["profiles"][0]
        options = dict(profile["options"])
        options["seed"] = 42
        started = time.perf_counter()
        first = gateway.remote(model, input_path, options)
        second = gateway.remote(model, input_path, options)
        if first["call_id"] != second["call_id"]:
            raise AssertionError(f"{model}: gateway idempotency failed")
        call_id = str(first["call_id"])
        submissions[model] = (call_id, started)
        idempotency[model] = True
        print(json.dumps({"phase": "submitted", "model": model, "callId": call_id, "idempotent": True}, separators=(",", ":")), flush=True)

    rows: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        futures = {
            pool.submit(wait_model, client, volume, model, call_id, started): model
            for model, (call_id, started) in submissions.items()
        }
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps({"phase": "model-result", **row}, separators=(",", ":")), flush=True)

    rows.sort(key=lambda item: MODELS.index(str(item["model"])))
    payload = {
        "status": "passed" if all(row["status"] == "passed" for row in rows) else "failed",
        "source": str(source.resolve()),
        "preprocess": preprocess_evidence,
        "inputPath": input_path,
        "idempotency": idempotency,
        "models": rows,
    }
    (RESULTS / "latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "models": len(rows)}, separators=(",", ":")))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
