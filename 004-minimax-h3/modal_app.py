# -*- coding: utf-8 -*-
"""
004-minimax-h3 — MiniMax H3 文生视频（Modal Volume 为唯一权威输出）。

【输出只写远程 Volume，不依赖本机目录】
  Volume 名: modal-lab-minimax-h3-outputs
  容器挂载: /outputs  →  Volume 根
  视频路径:
    /outputs/videos/<name>.mp4
    /outputs/videos/latest.mp4
  基准:
    /outputs/benchmarks/<name>.json

写完必须 outputs_vol.commit()，否则远端看不到。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import modal

APP_NAME = "modal-lab-minimax-h3"
DEFAULT_GPU = "RTX-PRO-6000"
HF_REPO = "Comfy-Org/MiniMax-H3"
COMFY_DIR = Path("/opt/ComfyUI")
WEIGHTS_MOUNT = "/weights"
OUTPUTS_MOUNT = "/outputs"
VIDEOS_DIR = Path(OUTPUTS_MOUNT) / "videos"
BENCH_DIR = Path(OUTPUTS_MOUNT) / "benchmarks"
COMFY_PORT = 8188
VOLUME_OUTPUTS_NAME = "modal-lab-minimax-h3-outputs"
VOLUME_WEIGHTS_NAME = "modal-lab-minimax-h3-weights"

GPU_PRICE_PER_SEC = {
    "RTX-PRO-6000": 0.000842,
    "A100-80GB": 0.000694,
    "A100-40GB": 0.000583,
    "L40S": 0.000542,
    "H100": 0.001097,
    "H100!": 0.001097,
}

WEIGHT_FILES = {
    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors": (
        "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    ),
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": (
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    ),
    "vae/minimax_h3_video_vae_fp16.safetensors": "vae/minimax_h3_video_vae_fp16.safetensors",
    "vae/minimax_h3_audio_vae_fp32.safetensors": "vae/minimax_h3_audio_vae_fp32.safetensors",
}

DOWNLOAD_TIMEOUT = 4 * 60 * 60
INFER_TIMEOUT = 2 * 60 * 60
SMOKE_TIMEOUT = 30 * 60

EXP_DIR = Path(__file__).resolve().parent

weights_vol = modal.Volume.from_name(VOLUME_WEIGHTS_NAME, create_if_missing=True)
outputs_vol = modal.Volume.from_name(VOLUME_OUTPUTS_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "wget",
        "curl",
        "ca-certificates",
    )
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "torchaudio==2.7.1",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .run_commands(
        "git clone --depth 1 --branch v0.30.0 "
        "https://github.com/Comfy-Org/ComfyUI.git /opt/ComfyUI",
        "python -m pip install --no-cache-dir -r /opt/ComfyUI/requirements.txt",
        "python -c \"import torch; print('torch', torch.__version__)\"",
    )
    .pip_install(
        "huggingface_hub[hf_transfer]>=0.26.0",
        "safetensors",
        "requests",
        "Pillow",
        "pyyaml",
        "websocket-client",
        "fastapi",
    )
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
            "COMFYUI_PATH": str(COMFY_DIR),
        }
    )
    .add_local_dir(
        str(EXP_DIR / "workflows"),
        remote_path="/root/workflows",
    )
)

serve_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi", "python-multipart")
    .env({"PYTHONUNBUFFERED": "1"})
)

app = modal.App(APP_NAME)


def _list_dir_sizes(root: str | Path) -> dict[str, Any]:
    p = Path(root)
    if not p.exists():
        return {"exists": False, "path": str(root), "files": 0, "size_gb": 0.0}
    total = 0
    files = 0
    for f in p.rglob("*"):
        if f.is_file():
            files += 1
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return {
        "exists": True,
        "path": str(root),
        "files": files,
        "size_gb": round(total / 1e9, 2),
    }


def _nvidia_smi_query() -> dict[str, Any] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    line = out.splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        return {"raw": out}
    return {
        "name": parts[0],
        "mem_used_mib": float(parts[1]),
        "mem_total_mib": float(parts[2]),
        "util_gpu_pct": float(parts[3]),
    }


class VramSampler:
    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict[str, Any]] = []
        self.peak_used_mib = 0.0
        self.peak_util = 0.0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self.summary()

    def _loop(self) -> None:
        while not self._stop.is_set():
            q = _nvidia_smi_query()
            if q and "mem_used_mib" in q:
                self.samples.append({"t": time.time(), **q})
                self.peak_used_mib = max(self.peak_used_mib, q["mem_used_mib"])
                self.peak_util = max(self.peak_util, q.get("util_gpu_pct") or 0.0)
            self._stop.wait(self.interval_s)

    def summary(self) -> dict[str, Any]:
        total = name = None
        if self.samples:
            total = self.samples[-1].get("mem_total_mib")
            name = self.samples[-1].get("name")
        return {
            "gpu_name_smi": name,
            "peak_mem_used_mib": round(self.peak_used_mib, 1),
            "peak_mem_used_gb": round(self.peak_used_mib / 1024.0, 2)
            if self.peak_used_mib
            else None,
            "mem_total_mib": total,
            "mem_total_gb": round(total / 1024.0, 2) if total else None,
            "peak_util_gpu_pct": round(self.peak_util, 1),
            "n_samples": len(self.samples),
            "samples_sparse": _sparse_samples(self.samples),
        }


def _sparse_samples(samples: list[dict[str, Any]], k: int = 12) -> list[dict[str, Any]]:
    if not samples:
        return []
    if len(samples) <= k:
        return [
            {"mem_used_gb": round(s["mem_used_mib"] / 1024, 2), "util": s.get("util_gpu_pct")}
            for s in samples
        ]
    idxs = sorted(set(int(i * (len(samples) - 1) / (k - 1)) for i in range(k)))
    return [
        {
            "mem_used_gb": round(samples[i]["mem_used_mib"] / 1024, 2),
            "util": samples[i].get("util_gpu_pct"),
        }
        for i in idxs
    ]


def _link_models_from_volume() -> None:
    models = COMFY_DIR / "models"
    for sub in ("diffusion_models", "text_encoders", "vae", "checkpoints", "clip"):
        (models / sub).mkdir(parents=True, exist_ok=True)
    vol = Path(WEIGHTS_MOUNT)
    for src_sub, dst_sub in (
        ("diffusion_models", "diffusion_models"),
        ("text_encoders", "text_encoders"),
        ("vae", "vae"),
    ):
        src_dir = vol / src_sub
        dst_dir = models / dst_sub
        if not src_dir.is_dir():
            continue
        for f in src_dir.iterdir():
            if not f.is_file():
                continue
            dest = dst_dir / f.name
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.symlink_to(f)


def _http_json(method: str, url: str, body: dict | None = None, timeout: float = 60.0) -> Any:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        if not raw:
            return None
        return json.loads(raw.decode("utf-8"))


def _wait_comfy(base: str, timeout_s: float = 180.0) -> None:
    t0 = time.time()
    last_err = None
    while time.time() - t0 < timeout_s:
        try:
            _http_json("GET", f"{base}/system_stats", timeout=5.0)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5)
    raise RuntimeError(f"ComfyUI 未在 {timeout_s}s 内就绪: {last_err!r}")


def _start_comfy() -> subprocess.Popen:
    _link_models_from_volume()
    (COMFY_DIR / "output").mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        str(COMFY_DIR / "main.py"),
        "--listen",
        "127.0.0.1",
        "--port",
        str(COMFY_PORT),
        "--gpu-only",
        "--disable-auto-launch",
        "--disable-metadata",
    ]
    log_path = Path("/tmp/comfyui.log")
    log_f = open(log_path, "w")  # noqa: SIM115
    proc = subprocess.Popen(
        cmd,
        cwd=str(COMFY_DIR),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    try:
        _wait_comfy(f"http://127.0.0.1:{COMFY_PORT}", timeout_s=240.0)
    except Exception:
        proc.terminate()
        try:
            print(log_path.read_text()[-8000:], flush=True)
        except OSError:
            pass
        raise
    return proc


def _stop_comfy(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def _queue_and_wait(workflow: dict[str, Any], timeout_s: float = 5400.0) -> dict[str, Any]:
    base = f"http://127.0.0.1:{COMFY_PORT}"
    payload = {"prompt": workflow, "client_id": "modal-lab-minimax-h3"}
    res = _http_json("POST", f"{base}/prompt", payload, timeout=120.0)
    prompt_id = res["prompt_id"]
    print(f"[comfy] queued prompt_id={prompt_id}", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            hist = _http_json("GET", f"{base}/history/{prompt_id}", timeout=30.0)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                time.sleep(2)
                continue
            raise
        if prompt_id in (hist or {}):
            entry = hist[prompt_id]
            status = entry.get("status") or {}
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI 执行失败: {status.get('messages') or entry}")
            if entry.get("outputs") is not None or status.get("completed"):
                return entry
        try:
            q = _http_json("GET", f"{base}/queue", timeout=10.0)
            running = q.get("queue_running") or []
            pending = q.get("queue_pending") or []
            still = any(prompt_id in str(x) for x in running + pending)
            if not still and prompt_id in (hist or {}):
                return hist[prompt_id]
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    raise TimeoutError(f"生成超时 ({timeout_s}s) prompt_id={prompt_id}")


def _find_newest_video(comfy_output: Path) -> Path | None:
    candidates = sorted(
        list(comfy_output.rglob("*.mp4"))
        + list(comfy_output.rglob("*.webm"))
        + list(comfy_output.rglob("*.mkv")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _publish_video_to_volume(src: Path, name: str, meta: dict[str, Any]) -> dict[str, Any]:
    """
    唯一出口：把成片写入远程 Modal Volume。

    Volume: modal-lab-minimax-h3-outputs
      videos/<name>.mp4
      videos/latest.mp4
      videos/<name>_meta.json
      videos/latest_meta.json
      benchmarks/<name>.json
    """
    # 确保看到最新 volume 状态再写
    try:
        outputs_vol.reload()
    except Exception as e:  # noqa: BLE001
        print(f"[volume] reload warn: {e!r}", flush=True)

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    named = VIDEOS_DIR / f"{name}.mp4"
    latest = VIDEOS_DIR / "latest.mp4"
    meta_named = VIDEOS_DIR / f"{name}_meta.json"
    meta_latest = VIDEOS_DIR / "latest_meta.json"
    bench_path = BENCH_DIR / f"{name}.json"

    if not src.is_file() or src.stat().st_size < 1000:
        raise RuntimeError(f"源视频无效: {src} size={src.stat().st_size if src.exists() else 0}")

    shutil.copy2(src, named)
    shutil.copy2(src, latest)
    size = named.stat().st_size
    if size < 1000:
        raise RuntimeError(f"写入后文件过小: {named} size={size}")

    payload = {
        **meta,
        "volume_name": VOLUME_OUTPUTS_NAME,
        "volume_paths": {
            "named": f"videos/{name}.mp4",
            "latest": "videos/latest.mp4",
            "meta": f"videos/{name}_meta.json",
            "latest_meta": "videos/latest_meta.json",
            "benchmark": f"benchmarks/{name}.json",
        },
        "container_paths": {
            "named": str(named),
            "latest": str(latest),
        },
        "bytes": size,
        "download_url_hint": (
            f"https://seachenxyt--modal-lab-minimax-h3-download.modal.run?name={name}"
        ),
        "cli_get": f"modal volume get {VOLUME_OUTPUTS_NAME} videos/{name}.mp4 ./",
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    meta_named.write_text(text, encoding="utf-8")
    meta_latest.write_text(text, encoding="utf-8")
    bench_path.write_text(text, encoding="utf-8")

    # 必须 commit，否则 Volume 对外部不可见
    outputs_vol.commit()
    print("=" * 60, flush=True)
    print("[VOLUME COMMITTED] 视频已写入远程 Modal Volume", flush=True)
    print(f"  volume : {VOLUME_OUTPUTS_NAME}", flush=True)
    print(f"  path   : videos/{name}.mp4  ({size} bytes)", flush=True)
    print("  also   : videos/latest.mp4", flush=True)
    print(f"  meta   : videos/{name}_meta.json", flush=True)
    print(f"  bench  : benchmarks/{name}.json", flush=True)
    print(f"  browser: {payload['download_url_hint']}", flush=True)
    print(f"  cli    : {payload['cli_get']}", flush=True)
    print("=" * 60, flush=True)

    # 自检：reload 后文件还在
    try:
        outputs_vol.reload()
        if not named.is_file() or named.stat().st_size != size:
            print(
                f"[volume] WARN post-reload size mismatch: "
                f"exists={named.is_file()} size={named.stat().st_size if named.exists() else 0}",
                flush=True,
            )
        else:
            print(f"[volume] self-check OK: {named} == {size} bytes", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[volume] self-check skip: {e!r}", flush=True)

    return payload


@app.function(
    image=image,
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=DOWNLOAD_TIMEOUT,
    cpu=4,
    memory=16384,
)
def download_weights(force: bool = False) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    root = Path(WEIGHTS_MOUNT)
    root.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    results = []
    for rel in WEIGHT_FILES:
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_file() and dest.stat().st_size > 1_000_000 and not force:
            results.append({"file": rel, "skipped": True, "size_gb": round(dest.stat().st_size / 1e9, 2)})
            print(f"[download] skip existing {rel}", flush=True)
            continue
        print(f"[download] {HF_REPO} :: {rel}", flush=True)
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename=rel,
            local_dir=str(root),
            token=token,
        )
        results.append(
            {
                "file": rel,
                "skipped": False,
                "path": path,
                "size_gb": round(Path(path).stat().st_size / 1e9, 2),
            }
        )
    weights_vol.commit()
    info = {"files": results, "volume": _list_dir_sizes(root)}
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
    return info


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    volumes={WEIGHTS_MOUNT: weights_vol},
    timeout=SMOKE_TIMEOUT,
    memory=65536,
    cpu=8,
)
def smoke() -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {
        "cuda": bool(torch.cuda.is_available()),
        "torch": str(torch.__version__),
        "gpu_name": None,
        "vram_gb": None,
        "smi": _nvidia_smi_query(),
        "weights": _list_dir_sizes(WEIGHTS_MOUNT),
    }
    if out["cuda"]:
        out["gpu_name"] = str(torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        out["vram_gb"] = round(int(props.total_memory) / 1024**3, 1)
    missing = [rel for rel in WEIGHT_FILES if not (Path(WEIGHTS_MOUNT) / rel).is_file()]
    out["missing_weights"] = missing
    out["weights_ready"] = not missing
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str), flush=True)
    return out


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    volumes={WEIGHTS_MOUNT: weights_vol, OUTPUTS_MOUNT: outputs_vol},
    timeout=INFER_TIMEOUT,
    memory=65536,
    cpu=8,
)
def t2v(
    prompt: str,
    width: int = 864,
    height: int = 480,
    seconds: float = 5.0,
    steps: int = 20,
    seed: int = 42,
    output_name: str = "t2v",
    gpu_label: str = DEFAULT_GPU,
    unet: str = "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    clip: str = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    video_vae: str = "minimax_h3_video_vae_fp16.safetensors",
    audio_vae: str = "minimax_h3_audio_vae_fp32.safetensors",
) -> dict[str, Any]:
    import importlib.util
    import sys

    import torch

    builder_path = Path("/root/workflows/build_t2v_api.py")
    spec = importlib.util.spec_from_file_location("build_t2v_api", builder_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_t2v_api"] = mod
    spec.loader.exec_module(mod)

    for rel in WEIGHT_FILES:
        p = Path(WEIGHTS_MOUNT) / rel
        if not p.is_file():
            raise FileNotFoundError(f"缺少权重 {p}；请先 download")

    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in output_name).strip("_") or "t2v"

    workflow = mod.build_t2v_prompt(
        prompt=prompt,
        width=width,
        height=height,
        seconds=seconds,
        steps=steps,
        seed=seed,
        unet=unet,
        clip=clip,
        video_vae=video_vae,
        audio_vae=audio_vae,
        filename_prefix=f"minimax_h3/{safe}",
    )

    actual_gpu = str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    vram_total_gb = round(int(props.total_memory) / 1024**3, 1) if props else None
    sm = None
    if props:
        major, minor = torch.cuda.get_device_capability(0)
        sm = f"sm_{major}{minor}"
    price = GPU_PRICE_PER_SEC.get(gpu_label)

    meta: dict[str, Any] = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "seconds": seconds,
        "steps": steps,
        "seed": seed,
        "frame_length": mod.frame_length_from_seconds(seconds),
        "gpu_request": gpu_label,
        "gpu_actual": actual_gpu,
        "vram_total_gb": vram_total_gb,
        "sm": sm,
        "unet": unet,
        "clip": clip,
        "output_name": safe,
        "comfy_flags": ["--gpu-only"],
        "price_per_sec_usd": price,
    }
    print("[t2v] config:", json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    print(
        f"[t2v] 完成后只写入远程 Volume: {VOLUME_OUTPUTS_NAME}/videos/{safe}.mp4",
        flush=True,
    )

    sampler = VramSampler(interval_s=1.0)
    proc = None
    t0 = time.time()
    try:
        meta["vram_idle"] = _nvidia_smi_query()
        sampler.start()
        proc = _start_comfy()
        t_ready = time.time()
        entry = _queue_and_wait(workflow, timeout_s=INFER_TIMEOUT - 300)
        t_done = time.time()
        vram = sampler.stop()
        src = _find_newest_video(COMFY_DIR / "output")
        if src is None:
            try:
                print(Path("/tmp/comfyui.log").read_text()[-6000:], flush=True)
            except OSError:
                pass
            raise RuntimeError("未找到 Comfy 输出视频")

        gen_s = round(t_done - t_ready, 1)
        total_s = round(time.time() - t0, 1)
        cost = round(total_s * price, 4) if price else None

        meta.update(
            {
                "seconds_comfy_ready": round(t_ready - t0, 1),
                "seconds_generate": gen_s,
                "seconds_total": total_s,
                "est_cost_usd_total": cost,
                "est_cost_usd_generate_only": round(gen_s * price, 4) if price else None,
                "vram": vram,
                "history_status": entry.get("status"),
            }
        )

        # ★ 只提交远程 Volume（本机目录与此无关）
        published = _publish_video_to_volume(src, safe, meta)

        result = {
            "ok": True,
            "where": "REMOTE Modal Volume ONLY (not local disk)",
            "volume_name": VOLUME_OUTPUTS_NAME,
            "volume_file": f"videos/{safe}.mp4",
            "volume_latest": "videos/latest.mp4",
            "bytes": published["bytes"],
            "gpu_request": gpu_label,
            "gpu_actual": actual_gpu,
            "seconds_total": total_s,
            "seconds_generate": gen_s,
            "peak_vram_gb": vram.get("peak_mem_used_gb"),
            "est_cost_usd_total": cost,
            "download_url": published["download_url_hint"],
            "cli_get": published["cli_get"],
            "paths": published["volume_paths"],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    except Exception:
        sampler.stop()
        raise
    finally:
        _stop_comfy(proc)


@app.function(
    image=serve_image,
    volumes={OUTPUTS_MOUNT: outputs_vol},
    timeout=120,
    cpu=1,
    memory=2048,
)
def list_outputs() -> dict[str, Any]:
    outputs_vol.reload()
    items = []
    if VIDEOS_DIR.is_dir():
        for f in sorted(VIDEOS_DIR.glob("*.mp4")):
            items.append(
                {
                    "name": f.name,
                    "volume_path": f"videos/{f.name}",
                    "bytes": f.stat().st_size,
                    "download_url": (
                        f"https://seachenxyt--modal-lab-minimax-h3-download.modal.run"
                        f"?name={f.stem}"
                    ),
                }
            )
    result = {
        "volume_name": VOLUME_OUTPUTS_NAME,
        "count": len(items),
        "videos": items,
        "note": "这些文件在远程 Modal Volume，不在仓库本地文件夹",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


@app.function(
    image=serve_image,
    volumes={OUTPUTS_MOUNT: outputs_vol},
    timeout=300,
    cpu=1,
    memory=4096,
)
@modal.fastapi_endpoint(method="GET")
def download(name: str = "latest"):
    from fastapi.responses import FileResponse, JSONResponse

    outputs_vol.reload()
    safe = "".join(c if c.isalnum() or c in "-_." else "" for c in name) or "latest"
    path = VIDEOS_DIR / (safe if safe.endswith(".mp4") else f"{safe}.mp4")
    if not path.is_file():
        alts = sorted(VIDEOS_DIR.glob("*.mp4")) if VIDEOS_DIR.is_dir() else []
        return JSONResponse(
            {
                "error": f"not found on volume: {path.name}",
                "volume": VOLUME_OUTPUTS_NAME,
                "available": [p.name for p in alts],
            },
            status_code=404,
        )
    return FileResponse(
        path=str(path),
        media_type="video/mp4",
        filename=path.name,
        headers={"Content-Disposition": f'inline; filename="{path.name}"'},
    )


@app.function(
    image=serve_image,
    volumes={OUTPUTS_MOUNT: outputs_vol},
    timeout=120,
    cpu=1,
    memory=2048,
)
@modal.fastapi_endpoint(method="GET")
def index():
    from fastapi.responses import HTMLResponse

    outputs_vol.reload()
    rows = []
    if VIDEOS_DIR.is_dir():
        for f in sorted(VIDEOS_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
            mb = f.stat().st_size / 1e6
            rows.append(
                f'<li><a href="download?name={f.stem}"><b>{f.name}</b></a> '
                f"({mb:.2f} MB) — volume path: videos/{f.name}</li>"
            )
    body = "\n".join(rows) or "<li>(volume empty)</li>"
    return HTMLResponse(
        f"""<!doctype html><html><head><meta charset="utf-8">
<title>Volume {VOLUME_OUTPUTS_NAME}</title>
<style>body{{font-family:system-ui;max-width:800px;margin:2rem auto;padding:0 1rem}}
code{{background:#f4f4f4;padding:2px 6px}}</style></head>
<body>
<h1>远程 Volume 视频</h1>
<p>Volume: <code>{VOLUME_OUTPUTS_NAME}</code></p>
<p>路径前缀: <code>videos/</code></p>
<ul>{body}</ul>
<p><a href="download?name=latest"><b>⬇ 下载 latest.mp4</b></a></p>
</body></html>"""
    )


@app.local_entrypoint()
def main(
    action: str = "status",
    prompt: str = "",
    width: int = 864,
    height: int = 480,
    seconds: float = 5.0,
    steps: int = 20,
    seed: int = 42,
    output_name: str = "t2v",
    gpu: str = DEFAULT_GPU,
    force_download: bool = False,
) -> None:
    if action == "status":
        print(
            json.dumps(
                {
                    "app": APP_NAME,
                    "default_gpu": DEFAULT_GPU,
                    "output_volume": VOLUME_OUTPUTS_NAME,
                    "output_layout": {
                        "videos": "videos/<name>.mp4 + videos/latest.mp4",
                        "benchmarks": "benchmarks/<name>.json",
                    },
                    "note": "成片只写远程 Volume；本机 outputs/ 不会自动出现文件",
                    "see_videos": [
                        f"modal volume ls {VOLUME_OUTPUTS_NAME} videos",
                        "https://seachenxyt--modal-lab-minimax-h3-index.modal.run",
                        "https://seachenxyt--modal-lab-minimax-h3-download.modal.run?name=latest",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if action == "download":
        print(download_weights.remote(force=force_download))
        return
    if action == "smoke":
        print(smoke.with_options(gpu=gpu).remote())
        return
    if action == "list-outputs":
        print(list_outputs.remote())
        return
    if action == "t2v":
        if not prompt.strip():
            raise SystemExit("t2v 需要 --prompt")
        print(
            t2v.with_options(gpu=gpu).remote(
                prompt=prompt,
                width=width,
                height=height,
                seconds=seconds,
                steps=steps,
                seed=seed,
                output_name=output_name,
                gpu_label=gpu,
            )
        )
        return
    raise SystemExit(f"unknown action={action!r}")
