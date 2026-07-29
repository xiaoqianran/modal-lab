# -*- coding: utf-8 -*-
"""MinerU 3.4.4 在 Modal 上的可复现解析与基准入口。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "modal-lab-mineru"
MINERU_VERSION = "3.4.4"
MINERU_COMMIT = "0dfc9460cd9ab693b9af60ae3fbffd7bc111b062"
DATA_DIR = Path("/data")
MODEL_DIR = Path("/models")
DEFAULT_REMOTE_PDF = "/books/EN-算法导论4.pdf"
DEFAULT_BACKEND = "hybrid-engine"
DEFAULT_EFFORT = "medium"
GPU_TYPE = os.environ.get("MODAL_LAB_GPU_TYPE", "H100!")

EXP_DIR = Path(__file__).resolve().parent
DEFAULT_LOCAL_PDF = EXP_DIR.parent / "books" / "EN-算法导论4.pdf"

model_volume = modal.Volume.from_name(
    "modal-lab-mineru-models", create_if_missing=True
)
data_volume = modal.Volume.from_name(
    "modal-lab-mineru-data", create_if_missing=True
)

model_env = {
    "HF_HOME": "/models/huggingface",
    "MODELSCOPE_CACHE": "/models/modelscope",
    "MINERU_TOOLS_CONFIG_JSON": "/models/mineru.json",
    "MINERU_MODEL_SOURCE": "local",
    "HF_XET_HIGH_PERFORMANCE": "1",
    "PYTHONUNBUFFERED": "1",
}

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "fonts-noto-core",
        "fonts-noto-cjk",
        "fontconfig",
    )
    .uv_pip_install(f"mineru=={MINERU_VERSION}")
    .env(model_env)
)

# 与 MinerU 官方 CUDA 12.9 Docker 路径一致。Modal 注入独立 Python，
# 因此仍需显式安装 vLLM extra，确保 MinerU 不会静默回退到 Transformers。
vllm_base_image = modal.Image.from_registry(
    "vllm/vllm-openai:v0.21.0-cu129",
    add_python="3.12",
)

inference_image = (
    vllm_base_image
    .entrypoint([])
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "fonts-noto-core",
        "fonts-noto-cjk",
        "fontconfig",
    )
    .run_commands(
        # 先让 vLLM 锁定其二进制兼容的 PyTorch 版本；反过来安装会让
        # MinerU core 拉取更新的 PyTorch，导致 vllm/_C 符号不匹配。
        "python3 -m pip install --no-cache-dir "
        "'vllm==0.21.0' --break-system-packages",
        "python3 -m pip install --no-cache-dir "
        f"'mineru[core]=={MINERU_VERSION}' --break-system-packages",
        "python3 -c \"import importlib.metadata, mineru, vllm; "
        "print('MinerU/vLLM ready:', "
        "importlib.metadata.version('mineru'), vllm.__version__, "
        "importlib.metadata.version('torch'))\"",
    )
    .env(
        {
            **model_env,
            "MINERU_PROCESSING_WINDOW_SIZE": "64",
            "MINERU_PDF_RENDER_THREADS": "8",
            "MINERU_API_MAX_CONCURRENT_REQUESTS": "3",
        }
    )
)

app = modal.App(APP_NAME)


def _dir_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path), "files": 0, "size_gb": 0}
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "exists": True,
        "path": str(path),
        "files": len(files),
        "size_gb": round(sum(item.stat().st_size for item in files) / 1e9, 3),
    }


@app.function(
    image=download_image,
    volumes={"/models": model_volume},
    timeout=2 * 60 * 60,
    cpu=4,
    memory=16384,
)
def download_models() -> dict[str, Any]:
    """下载 Hybrid 所需的 pipeline 与 VLM 模型并持久化配置。"""
    config_path = MODEL_DIR / "mineru.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            model_dirs = config.get("models-dir") or {}
            if all(
                Path(model_dirs.get(kind, "")).is_dir()
                for kind in ("pipeline", "vlm")
            ):
                result = _dir_info(MODEL_DIR)
                result.update({"skipped": True, "version": MINERU_VERSION})
                print(json.dumps(result, ensure_ascii=False), flush=True)
                return result
        except (OSError, ValueError, TypeError):
            pass

    env = {
        "HF_HOME": str(MODEL_DIR / "huggingface"),
        "MODELSCOPE_CACHE": str(MODEL_DIR / "modelscope"),
        "MINERU_TOOLS_CONFIG_JSON": str(config_path),
    }
    command = [
        "mineru-models-download",
        "--source",
        "huggingface",
        "--model_type",
        "all",
    ]
    subprocess.run(command, check=True, env={**os.environ, **env})
    model_volume.commit()
    result = _dir_info(MODEL_DIR)
    result.update({"skipped": False, "version": MINERU_VERSION})
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def _summarize_gpu(samples: list[dict[str, float]]) -> dict[str, Any]:
    if not samples:
        return {"sample_count": 0}

    def stats(key: str) -> dict[str, float]:
        values = [row[key] for row in samples]
        return {
            "avg": round(statistics.fmean(values), 2),
            "p50": round(statistics.median(values), 2),
            "p95": round(_percentile(values, 0.95), 2),
            "max": round(max(values), 2),
        }

    return {
        "sample_count": len(samples),
        "gpu_util_percent": stats("gpu_util_percent"),
        "memory_controller_util_percent": stats(
            "memory_controller_util_percent"
        ),
        "memory_used_mib": stats("memory_used_mib"),
        "memory_total_mib": round(samples[0]["memory_total_mib"], 2),
        "power_draw_w": stats("power_draw_w"),
        "power_limit_w": round(samples[0]["power_limit_w"], 2),
    }


def _sample_gpu(
    stop: threading.Event, samples: list[dict[str, float]]
) -> None:
    query = (
        "utilization.gpu,utilization.memory,memory.used,memory.total,"
        "power.draw,power.limit"
    )
    while not stop.is_set():
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            ).strip()
            values = [float(value.strip()) for value in output.split(",")]
            samples.append(
                {
                    "gpu_util_percent": values[0],
                    "memory_controller_util_percent": values[1],
                    "memory_used_mib": values[2],
                    "memory_total_mib": values[3],
                    "power_draw_w": values[4],
                    "power_limit_w": values[5],
                }
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        stop.wait(1)


def _output_slug(backend: str, effort: str) -> str:
    base = backend.removesuffix("-engine").replace("-", "_")
    return f"{base}-{effort}" if base == "hybrid" else base


@app.function(
    image=inference_image,
    gpu=GPU_TYPE,
    volumes={"/models": model_volume, "/data": data_volume},
    timeout=4 * 60 * 60,
    cpu=16,
    memory=65536,
    min_containers=0,
    max_containers=1,
    scaledown_window=60,
)
def parse_pdf(
    remote_pdf: str = DEFAULT_REMOTE_PDF,
    backend: str = DEFAULT_BACKEND,
    effort: str = DEFAULT_EFFORT,
    start_page: int = 1,
    end_page: int = 0,
    mode: str = "full",
    resume: bool = True,
) -> dict[str, Any]:
    """使用 MinerU Python API 解析整本或指定页段。"""
    import mineru
    import torch
    from mineru.cli.common import do_parse
    from mineru.utils.engine_utils import get_vlm_engine
    from pypdf import PdfReader

    if backend not in {"pipeline", "vlm-engine", "hybrid-engine"}:
        raise ValueError("backend 必须是 pipeline、vlm-engine 或 hybrid-engine")
    if effort not in {"medium", "high"}:
        raise ValueError("effort 必须是 medium 或 high")
    inference_engine = (
        "pipeline"
        if backend == "pipeline"
        else get_vlm_engine("auto", is_async=False)
    )

    pdf_path = DATA_DIR / remote_pdf.lstrip("/")
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")
    total_pages = len(PdfReader(str(pdf_path)).pages)
    final_page = total_pages if end_page <= 0 else min(end_page, total_pages)
    if not 1 <= start_page <= final_page:
        raise ValueError(f"页码范围无效: {start_page}..{final_page}")

    slug = _output_slug(backend, effort)
    if mode == "full":
        output_dir = DATA_DIR / "outputs" / pdf_path.stem / slug
    elif mode == "benchmark":
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = (
            DATA_DIR
            / "benchmarks"
            / pdf_path.stem
            / f"{slug}-{run_id}-p{start_page}-{final_page}"
        )
    else:
        raise ValueError("mode 必须是 full 或 benchmark")

    summary_path = output_dir / "summary.json"
    if resume and summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            previous.get("status") == "completed"
            and previous.get("start_page") == start_page
            and previous.get("end_page") == final_page
        ):
            previous["skipped"] = True
            print(json.dumps(previous, ensure_ascii=False), flush=True)
            return previous

    output_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, float]] = []
    sample_stop = threading.Event()
    sampler = threading.Thread(
        target=_sample_gpu,
        args=(sample_stop, samples),
        daemon=True,
    )
    sampler.start()
    started = time.perf_counter()
    try:
        do_parse(
            output_dir=str(output_dir),
            pdf_file_names=[pdf_path.stem],
            pdf_bytes_list=[pdf_path.read_bytes()],
            p_lang_list=["en"],
            backend=backend,
            parse_method="auto",
            formula_enable=True,
            table_enable=True,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=True,
            f_dump_middle_json=True,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=True,
            start_page_id=start_page - 1,
            end_page_id=final_page - 1,
            image_analysis=effort == "high",
            effort=effort,
        )
        status = "completed"
        error = None
    except Exception as exc:
        status = "failed"
        error = repr(exc)
    finally:
        elapsed_seconds = time.perf_counter() - started
        sample_stop.set()
        sampler.join(timeout=5)

    markdown_files = sorted(output_dir.rglob("*.md"))
    json_files = sorted(output_dir.rglob("*.json"))
    image_files = sorted(
        item
        for item in output_dir.rglob("*")
        if item.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    main_markdown = (
        max(markdown_files, key=lambda item: item.stat().st_size)
        if markdown_files
        else None
    )
    parsed_pages = final_page - start_page + 1
    summary = {
        "status": status,
        "error": error,
        "app": APP_NAME,
        "mineru_version": getattr(mineru, "__version__", MINERU_VERSION),
        "mineru_commit": MINERU_COMMIT,
        "backend": backend,
        "inference_engine": inference_engine,
        "effort": effort,
        "gpu": torch.cuda.get_device_name(0),
        "pdf": str(pdf_path),
        "pdf_pages": total_pages,
        "start_page": start_page,
        "end_page": final_page,
        "parsed_pages": parsed_pages,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "pages_per_minute": round(
            parsed_pages / elapsed_seconds * 60, 3
        ),
        "markdown_files": len(markdown_files),
        "json_files": len(json_files),
        "image_files": len(image_files),
        "main_markdown": str(main_markdown) if main_markdown else None,
        "main_markdown_bytes": (
            main_markdown.stat().st_size if main_markdown else 0
        ),
        "gpu_metrics": _summarize_gpu(samples),
        "output_dir": str(output_dir),
        "skipped": False,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    data_volume.commit()
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if status == "failed":
        raise RuntimeError(error)
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def _upload_pdf(local_pdf: Path, remote_pdf: str) -> dict[str, Any]:
    if not local_pdf.is_file():
        raise FileNotFoundError(f"本地 PDF 不存在: {local_pdf}")
    skipped = False
    try:
        async with data_volume.batch_upload() as batch:
            batch.put_file(str(local_pdf), remote_pdf)
    except FileExistsError:
        skipped = True
    result = {
        "local_pdf": str(local_pdf),
        "remote_pdf": remote_pdf,
        "size_mb": round(local_pdf.stat().st_size / 1e6, 3),
        "sha256": _sha256(local_pdf),
        "skipped": skipped,
    }
    print(json.dumps({"event": "pdf_uploaded", **result}, ensure_ascii=False))
    return result


@app.local_entrypoint()
async def main(
    action: str = "info",
    local_pdf: str = str(DEFAULT_LOCAL_PDF),
    remote_pdf: str = DEFAULT_REMOTE_PDF,
    backend: str = DEFAULT_BACKEND,
    effort: str = DEFAULT_EFFORT,
    start_page: int = 1,
    end_page: int = 0,
    benchmark_pages: int = 100,
    resume: bool = True,
) -> None:
    action = action.strip().lower()
    if action == "info":
        print(
            json.dumps(
                {
                    "app": APP_NAME,
                    "mineru_version": MINERU_VERSION,
                    "mineru_commit": MINERU_COMMIT,
                    "default_backend": DEFAULT_BACKEND,
                    "default_effort": DEFAULT_EFFORT,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if action == "upload":
        result = await _upload_pdf(Path(local_pdf), remote_pdf)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if action == "download":
        result = await download_models.remote.aio()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if action not in {"parse", "benchmark"}:
        raise SystemExit(
            "action 必须是 info、upload、download、parse 或 benchmark"
        )

    upload = await _upload_pdf(Path(local_pdf), remote_pdf)
    models = await download_models.remote.aio()
    if action == "benchmark":
        end_page = start_page + benchmark_pages - 1
    result = await parse_pdf.remote.aio(
        remote_pdf=remote_pdf,
        backend=backend,
        effort=effort,
        start_page=start_page,
        end_page=end_page,
        mode="benchmark" if action == "benchmark" else "full",
        resume=resume,
    )
    print(
        json.dumps(
            {"upload": upload, "models": models, "result": result},
            ensure_ascii=False,
            indent=2,
        )
    )
