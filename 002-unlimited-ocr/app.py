# -*- coding: utf-8 -*-
"""Unlimited-OCR 在 Modal 上的 SGLang 批量解析与基准入口。"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import queue
import re
import statistics
import sys
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "modal-lab-unlimited-ocr"
MODEL_ID = "baidu/Unlimited-OCR"
MODEL_REVISION = "3f2e9c956588f5560efcfb7c62240f5d67b63e60"
MODEL_DIR = Path("/models/Unlimited-OCR")
DATA_DIR = Path("/data")
DEFAULT_REMOTE_PDF = "/books/EN-算法导论4.pdf"
DEFAULT_CONCURRENCY = 24
DEFAULT_GPU = "H100!"
SERVER_URL = "http://127.0.0.1:10000"
SERVED_MODEL_NAME = "Unlimited-OCR"
SGLANG_WHEEL = (
    "https://raw.githubusercontent.com/baidu/Unlimited-OCR/main/wheel/"
    "sglang-0.0.0.dev11416%2Bg92e8bb79e-py3-none-any.whl"
)

EXP_DIR = Path(__file__).resolve().parent
DEFAULT_LOCAL_PDF = EXP_DIR.parent / "books" / "EN-算法导论4.pdf"

model_volume = modal.Volume.from_name(
    "modal-lab-unlimited-ocr-weights", create_if_missing=True
)
data_volume = modal.Volume.from_name(
    "modal-lab-unlimited-ocr-data", create_if_missing=True
)

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface_hub[hf_xet]==0.36.2")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1"})
)

sglang_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("libgl1", "libglib2.0-0", "libnuma1")
    .uv_pip_install(
        SGLANG_WHEEL,
        "kernels==0.11.7",
        "PyMuPDF==1.27.2.2",
    )
    .env(
        {
            "HF_HOME": "/models/hf-cache",
            "HF_HUB_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
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
    timeout=60 * 60,
    cpu=4,
    memory=8192,
)
def download_model() -> dict[str, Any]:
    """把固定 revision 的模型缓存到独立 Volume。"""
    from huggingface_hub import snapshot_download

    revision_file = MODEL_DIR / ".modal-lab-revision"
    if (
        (MODEL_DIR / "config.json").is_file()
        and revision_file.is_file()
        and revision_file.read_text(encoding="utf-8").strip() == MODEL_REVISION
    ):
        result = _dir_info(MODEL_DIR)
        result["skipped"] = True
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=str(MODEL_DIR),
        ignore_patterns=["*.pdf", "*.gif", "assets/*", "wheel/*"],
    )
    revision_file.write_text(MODEL_REVISION + "\n", encoding="utf-8")
    model_volume.commit()
    result = _dir_info(MODEL_DIR)
    result["skipped"] = False
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def _clean_grounding(raw: str) -> str:
    """去除版面坐标，保留内容、标题语义和图片占位符。"""

    def semantic_prefix(label: str) -> str:
        kind = label.strip().lower().replace("-", "_")
        if kind == "image":
            return "\n[IMAGE]\n"
        if kind in {"title", "page_title"}:
            return "\n# "
        if kind in {"sub_title", "subtitle", "section_title"}:
            return "\n## "
        return ""

    ref_det_pattern = re.compile(
        r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>.*?<\|/det\|>",
        re.DOTALL,
    )
    det_only_pattern = re.compile(
        r"<\|det\|>\s*([A-Za-z_][\w-]*)\s*\[[^\]]+\]\s*<\|/det\|>",
        re.DOTALL,
    )
    text = ref_det_pattern.sub(
        lambda match: semantic_prefix(match.group(1)), raw
    )
    text = det_only_pattern.sub(
        lambda match: semantic_prefix(match.group(1)), text
    )
    text = re.sub(r"<\|/?(?:ref|det)\|>", "", text)
    text = text.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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


@app.cls(
    image=sglang_image,
    gpu=DEFAULT_GPU,
    volumes={"/models": model_volume, "/data": data_volume},
    timeout=45 * 60,
    cpu=8,
    memory=32768,
    min_containers=0,
    max_containers=3,
    scaledown_window=30,
)
class UnlimitedOCR:
    @modal.enter()
    def start_server(self) -> None:
        import requests

        if not (MODEL_DIR / "config.json").is_file():
            raise FileNotFoundError(f"模型不存在: {MODEL_DIR}")

        gpu_line = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,power.limit",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        self.gpu_name, total_memory, power_limit = [
            value.strip() for value in gpu_line.split(",")
        ]
        self.gpu_total_memory_mib = float(total_memory)
        self.gpu_power_limit_w = float(power_limit)
        self.attention_backend = (
            "flashinfer"
            if "RTX PRO 6000" in self.gpu_name.upper()
            else "fa3"
        )

        self.server_log = Path("/tmp/sglang-server.log")
        log_handle = self.server_log.open("w", encoding="utf-8")
        command = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model",
            str(MODEL_DIR),
            "--served-model-name",
            SERVED_MODEL_NAME,
            "--attention-backend",
            self.attention_backend,
            "--page-size",
            "1",
            "--mem-fraction-static",
            "0.8",
            "--context-length",
            "32768",
            "--enable-custom-logit-processor",
            "--disable-overlap-schedule",
            "--skip-server-warmup",
            "--cuda-graph-max-bs",
            "32",
            "--host",
            "127.0.0.1",
            "--port",
            "10000",
        ]
        started = time.perf_counter()
        self.server_process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        self.server_log_handle = log_handle
        deadline = time.monotonic() + 10 * 60
        last_error = ""
        while time.monotonic() < deadline:
            if self.server_process.poll() is not None:
                log_handle.flush()
                tail = self.server_log.read_text(
                    encoding="utf-8", errors="replace"
                )[-12000:]
                raise RuntimeError(f"SGLang 启动失败:\n{tail}")
            try:
                response = requests.get(f"{SERVER_URL}/health", timeout=2)
                if response.status_code == 200:
                    break
                last_error = f"HTTP {response.status_code}"
            except requests.RequestException as exc:
                last_error = str(exc)
            time.sleep(2)
        else:
            raise TimeoutError(f"SGLang 启动超时: {last_error}")

        self.server_load_seconds = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "event": "sglang_ready",
                    "gpu": self.gpu_name,
                    "attention_backend": self.attention_backend,
                    "memory_total_mib": self.gpu_total_memory_mib,
                    "power_limit_w": self.gpu_power_limit_w,
                    "seconds": round(self.server_load_seconds, 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    @modal.exit()
    def stop_server(self) -> None:
        if getattr(self, "server_process", None) is not None:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait(timeout=10)
        if getattr(self, "server_log_handle", None) is not None:
            self.server_log_handle.close()

    @staticmethod
    def _encode_image(image_path: Path) -> dict[str, Any]:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded}"},
        }

    def _infer_one(
        self,
        image_path: Path,
        max_tokens: int,
        image_mode: str,
        processor: str,
    ) -> dict[str, Any]:
        import requests

        started = time.perf_counter()
        payload = {
            "model": SERVED_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "document parsing."},
                        self._encode_image(image_path),
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "skip_special_tokens": False,
            "images_config": {"image_mode": image_mode},
            "custom_logit_processor": processor,
            "custom_params": {"ngram_size": 35, "window_size": 128},
        }
        response = requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=20 * 60,
        )
        response.raise_for_status()
        body = response.json()
        raw = body["choices"][0]["message"]["content"]
        usage = body.get("usage") or {}
        return {
            "raw": raw,
            "output_tokens": int(usage.get("completion_tokens") or 0),
            "request_seconds": round(time.perf_counter() - started, 3),
            "finish_reason": body["choices"][0].get("finish_reason"),
        }

    @staticmethod
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

    @staticmethod
    def _validate(
        concurrency: int,
        remote_pdf: str,
        start_page: int,
    ) -> tuple[Path, int]:
        if concurrency not in {1, 2, 4, 8, 16, 24, 32}:
            raise ValueError("concurrency 必须是 1/2/4/8/16/24/32")
        pdf_path = DATA_DIR / remote_pdf.lstrip("/")
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF 不存在: {pdf_path}")
        import fitz

        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
        if not 1 <= start_page <= total_pages:
            raise ValueError(f"start_page 必须在 1..{total_pages} 内")
        return pdf_path, total_pages

    def _write_server_log(self, output_dir: Path) -> None:
        self.server_log_handle.flush()
        (output_dir / "sglang-server.log").write_text(
            self.server_log.read_text(encoding="utf-8", errors="replace"),
            encoding="utf-8",
        )

    @modal.method()
    def benchmark(
        self,
        concurrency: int = DEFAULT_CONCURRENCY,
        runtime_seconds: int = 300,
        remote_pdf: str = DEFAULT_REMOTE_PDF,
        start_page: int = 1,
        dpi: int = 200,
        max_tokens: int = 4096,
        image_mode: str = "gundam",
    ) -> dict[str, Any]:
        """预渲染后测试纯 OCR 吞吐，便于复现实验表。"""
        import fitz
        from sglang.srt.sampling.custom_logit_processor import (
            DeepseekOCRNoRepeatNGramLogitProcessor,
        )

        if runtime_seconds < 30:
            raise ValueError("runtime_seconds 至少为 30")
        pdf_path, total_pages = self._validate(
            concurrency, remote_pdf, start_page
        )
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = (
            DATA_DIR
            / "outputs"
            / pdf_path.stem
            / f"benchmark-{run_id}-c{concurrency}-s{runtime_seconds}"
        )
        pages_dir = output_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)

        render_count = min(1024, total_pages - start_page + 1)
        image_paths: list[tuple[int, Path]] = []
        render_started = time.perf_counter()
        with fitz.open(pdf_path) as doc:
            for page_number in range(start_page, start_page + render_count):
                pix = doc[page_number - 1].get_pixmap(
                    matrix=fitz.Matrix(dpi / 72, dpi / 72),
                    alpha=False,
                )
                image_path = Path(f"/tmp/page-{page_number:04d}.png")
                pix.save(image_path)
                image_paths.append((page_number, image_path))
        render_seconds = time.perf_counter() - render_started

        processor = DeepseekOCRNoRepeatNGramLogitProcessor.to_str()
        work_lock = threading.Lock()
        next_index = 0
        rows: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        samples: list[dict[str, float]] = []
        sample_stop = threading.Event()
        benchmark_started = time.perf_counter()
        deadline = benchmark_started + runtime_seconds
        sampler = threading.Thread(
            target=self._sample_gpu,
            args=(sample_stop, samples),
            daemon=True,
        )
        sampler.start()

        def worker() -> None:
            nonlocal next_index
            while time.perf_counter() < deadline:
                with work_lock:
                    if next_index >= len(image_paths):
                        return
                    page_number, image_path = image_paths[next_index]
                    next_index += 1
                try:
                    result = self._infer_one(
                        image_path, max_tokens, image_mode, processor
                    )
                    raw = str(result.pop("raw") or "")
                    (pages_dir / f"{page_number:04d}.raw.md").write_text(
                        raw, encoding="utf-8"
                    )
                    row = {"page": page_number, **result}
                    with work_lock:
                        rows.append(row)
                        completed = len(rows)
                    if completed <= concurrency or completed % 25 == 0:
                        print(
                            json.dumps(
                                {
                                    "event": "progress",
                                    "completed": completed,
                                    **row,
                                }
                            ),
                            flush=True,
                        )
                except Exception as exc:
                    error = {"page": page_number, "error": repr(exc)}
                    with work_lock:
                        errors.append(error)
                    print(
                        json.dumps({"event": "page_error", **error}),
                        flush=True,
                    )

        try:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(worker) for _ in range(concurrency)
                ]
                for future in as_completed(futures):
                    future.result()
        finally:
            for _, image_path in image_paths:
                image_path.unlink(missing_ok=True)
            sample_stop.set()
            sampler.join(timeout=5)

        benchmark_seconds = time.perf_counter() - benchmark_started
        rows.sort(key=lambda row: row["page"])
        errors.sort(key=lambda row: row["page"])
        total_output_tokens = sum(row["output_tokens"] for row in rows)
        summary = {
            "app": APP_NAME,
            "backend": "sglang",
            "attention_backend": self.attention_backend,
            "gpu": self.gpu_name,
            "server_load_seconds": round(self.server_load_seconds, 3),
            "pdf": str(pdf_path),
            "pdf_pages": total_pages,
            "start_page": start_page,
            "last_completed_page": rows[-1]["page"] if rows else None,
            "runtime_budget_seconds": runtime_seconds,
            "benchmark_seconds": round(benchmark_seconds, 3),
            "concurrency": concurrency,
            "image_mode": image_mode,
            "dpi": dpi,
            "max_tokens": max_tokens,
            "rendered_pages": render_count,
            "render_seconds_before_benchmark": round(render_seconds, 3),
            "pages_completed": len(rows),
            "pages_per_minute": round(
                len(rows) / benchmark_seconds * 60, 3
            ),
            "total_output_tokens": total_output_tokens,
            "output_tokens_per_second": round(
                total_output_tokens / benchmark_seconds, 3
            ),
            "errors": errors,
            "gpu_metrics": _summarize_gpu(samples),
            "pages": rows,
            "output_dir": str(output_dir),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._write_server_log(output_dir)
        data_volume.commit()
        print(
            json.dumps(
                {
                    "event": "benchmark_complete",
                    **{
                        key: value
                        for key, value in summary.items()
                        if key != "pages"
                    },
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return summary

    @modal.method()
    def parse_pdf(
        self,
        concurrency: int = DEFAULT_CONCURRENCY,
        remote_pdf: str = DEFAULT_REMOTE_PDF,
        start_page: int = 1,
        end_page: int = 0,
        dpi: int = 200,
        max_tokens: int = 4096,
        image_mode: str = "gundam",
        retries: int = 3,
        resume: bool = True,
    ) -> dict[str, Any]:
        """边渲染边识别整本 PDF，并生成可续跑的逐页和合并结果。"""
        import fitz
        from sglang.srt.sampling.custom_logit_processor import (
            DeepseekOCRNoRepeatNGramLogitProcessor,
        )

        pdf_path, total_pages = self._validate(
            concurrency, remote_pdf, start_page
        )
        final_page = total_pages if end_page <= 0 else min(end_page, total_pages)
        if final_page < start_page:
            raise ValueError("end_page 必须不小于 start_page")
        if retries < 1:
            raise ValueError("retries 至少为 1")

        output_dir = (
            DATA_DIR
            / "outputs"
            / pdf_path.stem
            / f"full-c{concurrency}-{image_mode}-dpi{dpi}"
        )
        pages_dir = output_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)

        def page_complete(page_number: int) -> bool:
            return all(
                (pages_dir / f"{page_number:04d}{suffix}").is_file()
                for suffix in (".raw.md", ".md", ".json")
            )

        requested_pages = list(range(start_page, final_page + 1))
        skipped_pages = [
            page_number
            for page_number in requested_pages
            if resume and page_complete(page_number)
        ]
        skipped_set = set(skipped_pages)
        pending_pages = [
            page_number
            for page_number in requested_pages
            if page_number not in skipped_set
        ]

        processor = DeepseekOCRNoRepeatNGramLogitProcessor.to_str()
        work_queue: queue.Queue[tuple[int, Path, float] | None] = queue.Queue(
            maxsize=max(64, concurrency * 3)
        )
        rows: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        producer_errors: list[dict[str, Any]] = []
        samples: list[dict[str, float]] = []
        state_lock = threading.Lock()
        commit_lock = threading.Lock()
        completed_since_commit = 0
        started = time.perf_counter()
        sample_stop = threading.Event()
        sampler = threading.Thread(
            target=self._sample_gpu,
            args=(sample_stop, samples),
            daemon=True,
        )
        sampler.start()

        def producer() -> None:
            try:
                with fitz.open(pdf_path) as doc:
                    for page_number in pending_pages:
                        render_started = time.perf_counter()
                        pix = doc[page_number - 1].get_pixmap(
                            matrix=fitz.Matrix(dpi / 72, dpi / 72),
                            alpha=False,
                        )
                        image_path = Path(
                            f"/tmp/unlimited-ocr-{page_number:04d}.png"
                        )
                        pix.save(image_path)
                        work_queue.put(
                            (
                                page_number,
                                image_path,
                                time.perf_counter() - render_started,
                            )
                        )
            except Exception as exc:
                producer_errors.append({"error": repr(exc)})
            finally:
                for _ in range(concurrency):
                    work_queue.put(None)

        def worker() -> None:
            nonlocal completed_since_commit
            while True:
                item = work_queue.get()
                if item is None:
                    work_queue.task_done()
                    return
                page_number, image_path, render_seconds = item
                page_started = time.perf_counter()
                try:
                    last_error: Exception | None = None
                    result: dict[str, Any] | None = None
                    attempts = 0
                    for attempts in range(1, retries + 1):
                        try:
                            result = self._infer_one(
                                image_path,
                                max_tokens,
                                image_mode,
                                processor,
                            )
                            break
                        except Exception as exc:
                            last_error = exc
                            if attempts < retries:
                                time.sleep(2**attempts)
                    if result is None:
                        raise RuntimeError(
                            f"{retries} 次请求均失败: {last_error!r}"
                        )

                    raw = str(result.pop("raw") or "")
                    clean = _clean_grounding(raw)
                    row = {
                        "page": page_number,
                        "render_seconds": round(render_seconds, 3),
                        **result,
                        "attempts": attempts,
                        "output_chars": len(clean),
                        "wall_seconds": round(
                            time.perf_counter() - page_started, 3
                        ),
                    }
                    (pages_dir / f"{page_number:04d}.raw.md").write_text(
                        raw, encoding="utf-8"
                    )
                    (pages_dir / f"{page_number:04d}.md").write_text(
                        clean, encoding="utf-8"
                    )
                    (pages_dir / f"{page_number:04d}.json").write_text(
                        json.dumps(row, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    should_commit = False
                    with state_lock:
                        rows.append(row)
                        completed = len(rows)
                        completed_since_commit += 1
                        if completed_since_commit >= 100:
                            completed_since_commit = 0
                            should_commit = True
                    if should_commit:
                        with commit_lock:
                            data_volume.commit()
                    if completed <= concurrency or completed % 50 == 0:
                        elapsed = time.perf_counter() - started
                        print(
                            json.dumps(
                                {
                                    "event": "progress",
                                    "completed_this_run": completed,
                                    "skipped": len(skipped_pages),
                                    "target": len(requested_pages),
                                    "pages_per_minute": round(
                                        completed / elapsed * 60, 3
                                    ),
                                    **row,
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                except Exception as exc:
                    error = {"page": page_number, "error": repr(exc)}
                    with state_lock:
                        errors.append(error)
                    print(
                        json.dumps(
                            {"event": "page_error", **error},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                finally:
                    image_path.unlink(missing_ok=True)
                    work_queue.task_done()

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()
        try:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(worker) for _ in range(concurrency)
                ]
                for future in as_completed(futures):
                    future.result()
            producer_thread.join()
        finally:
            sample_stop.set()
            sampler.join(timeout=5)

        elapsed_seconds = time.perf_counter() - started
        rows.sort(key=lambda row: row["page"])
        errors.sort(key=lambda row: row["page"])

        completed_pages = [
            page_number
            for page_number in requested_pages
            if page_complete(page_number)
        ]
        merged_path = output_dir / "book.md"
        with merged_path.open("w", encoding="utf-8") as merged:
            for page_number in completed_pages:
                clean = (pages_dir / f"{page_number:04d}.md").read_text(
                    encoding="utf-8"
                )
                merged.write(f"<!-- page {page_number} -->\n\n")
                merged.write(clean)
                merged.write("\n\n")

        total_output_tokens = sum(
            json.loads(
                (pages_dir / f"{page_number:04d}.json").read_text(
                    encoding="utf-8"
                )
            ).get("output_tokens", 0)
            for page_number in completed_pages
        )
        summary = {
            "app": APP_NAME,
            "backend": "sglang",
            "attention_backend": self.attention_backend,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "gpu": self.gpu_name,
            "server_load_seconds": round(self.server_load_seconds, 3),
            "pdf": str(pdf_path),
            "pdf_pages": total_pages,
            "start_page": start_page,
            "end_page": final_page,
            "requested_pages": len(requested_pages),
            "resumed_pages": len(skipped_pages),
            "pages_completed_this_run": len(rows),
            "pages_completed_total": len(completed_pages),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "pages_per_minute_this_run": round(
                len(rows) / elapsed_seconds * 60, 3
            )
            if elapsed_seconds
            else 0,
            "concurrency": concurrency,
            "image_mode": image_mode,
            "dpi": dpi,
            "max_tokens": max_tokens,
            "retries": retries,
            "total_output_tokens": total_output_tokens,
            "output_tokens_per_second_this_run": round(
                sum(row["output_tokens"] for row in rows) / elapsed_seconds, 3
            )
            if elapsed_seconds
            else 0,
            "errors": errors,
            "producer_errors": producer_errors,
            "gpu_metrics": _summarize_gpu(samples),
            "merged_markdown": str(merged_path),
            "output_dir": str(output_dir),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._write_server_log(output_dir)
        data_volume.commit()
        print(
            json.dumps(
                {"event": "parse_complete", **summary},
                ensure_ascii=False,
            ),
            flush=True,
        )
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="002 Unlimited-OCR on Modal")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="打印实验固定信息；纯本地")

    download = sub.add_parser("download", help="下载固定 revision 模型")
    download.add_argument("--dry-run", action="store_true")

    upload = sub.add_parser("upload", help="上传本地 PDF 到 data Volume，并记录 hash")
    upload.add_argument("--pdf", type=Path, default=DEFAULT_LOCAL_PDF)
    upload.add_argument("--remote-pdf", default=DEFAULT_REMOTE_PDF)
    upload.add_argument("--dry-run", action="store_true")

    benchmark = sub.add_parser("benchmark", help="固定时间窗口吞吐 benchmark")
    benchmark.add_argument("--pdf", type=Path, default=DEFAULT_LOCAL_PDF)
    benchmark.add_argument("--remote-pdf", default=DEFAULT_REMOTE_PDF)
    benchmark.add_argument("--seconds", type=int, default=300)
    benchmark.add_argument("--start-page", type=int, default=1)
    benchmark.add_argument("--dpi", type=int, default=200)
    benchmark.add_argument("--max-tokens", type=int, default=4096)
    benchmark.add_argument("--concurrencies", default=str(DEFAULT_CONCURRENCY))
    benchmark.add_argument("--image-mode", default="gundam")
    benchmark.add_argument("--gpu", default=DEFAULT_GPU)
    benchmark.add_argument("--dry-run", action="store_true")

    parse = sub.add_parser("parse", help="完整/区间解析")
    parse.add_argument("--pdf", type=Path, default=DEFAULT_LOCAL_PDF)
    parse.add_argument("--remote-pdf", default=DEFAULT_REMOTE_PDF)
    parse.add_argument("--start-page", type=int, default=1)
    parse.add_argument("--end-page", type=int, default=0)
    parse.add_argument("--dpi", type=int, default=200)
    parse.add_argument("--max-tokens", type=int, default=4096)
    parse.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parse.add_argument("--image-mode", default="gundam")
    parse.add_argument("--retries", type=int, default=3)
    parse.add_argument("--no-resume", action="store_true")
    parse.add_argument("--gpu", default=DEFAULT_GPU)
    parse.add_argument("--dry-run", action="store_true")
    return parser


def parse_cli(argv: list[str] | tuple[str, ...]) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


def local_status() -> dict[str, Any]:
    return {
        "experiment": "002-unlimited-ocr",
        "app": APP_NAME,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "default_gpu": DEFAULT_GPU,
        "default_concurrency": DEFAULT_CONCURRENCY,
        "default_pdf": str(DEFAULT_LOCAL_PDF),
        "default_pdf_exists": DEFAULT_LOCAL_PDF.is_file(),
        "model_volume": "modal-lab-unlimited-ocr-weights",
        "data_volume": "modal-lab-unlimited-ocr-data",
        "gpu_selection": "explicit Modal Cls.with_options; no environment-variable protocol",
    }


def _parse_concurrencies(raw: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError:
        raise ValueError("--concurrencies 必须是逗号分隔整数") from None
    if not values or any(value <= 0 for value in values):
        raise ValueError("concurrency 必须 > 0")
    return values


def benchmark_plan(args: argparse.Namespace) -> dict[str, Any]:
    values = _parse_concurrencies(args.concurrencies)
    if args.seconds <= 0:
        raise ValueError("--seconds 必须 > 0")
    if args.start_page < 1:
        raise ValueError("--start-page 必须 >= 1")
    if args.dpi <= 0 or args.max_tokens <= 0:
        raise ValueError("--dpi / --max-tokens 必须 > 0")
    return {
        "action": "benchmark",
        "local_pdf": str(args.pdf),
        "remote_pdf": args.remote_pdf,
        "gpu": args.gpu,
        "runtime_seconds": args.seconds,
        "start_page": args.start_page,
        "dpi": args.dpi,
        "max_tokens": args.max_tokens,
        "concurrencies": values,
        "image_mode": args.image_mode,
    }


def parse_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.start_page < 1:
        raise ValueError("--start-page 必须 >= 1")
    if args.end_page > 0 and args.end_page < args.start_page:
        raise ValueError("--end-page 不能小于 --start-page")
    if args.dpi <= 0 or args.max_tokens <= 0 or args.concurrency <= 0:
        raise ValueError("--dpi / --max-tokens / --concurrency 必须 > 0")
    if args.retries < 0:
        raise ValueError("--retries 必须 >= 0")
    return {
        "action": "parse",
        "local_pdf": str(args.pdf),
        "remote_pdf": args.remote_pdf,
        "gpu": args.gpu,
        "start_page": args.start_page,
        "end_page": args.end_page,
        "dpi": args.dpi,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "image_mode": args.image_mode,
        "retries": args.retries,
        "resume": not args.no_resume,
    }


async def cli(argv: list[str] | tuple[str, ...]) -> None:
    args = parse_cli(argv)
    if args.command == "status":
        print(json.dumps(local_status(), ensure_ascii=False, indent=2))
        return
    if args.command == "download":
        if args.dry_run:
            print(json.dumps({"action": "download"}, ensure_ascii=False, indent=2))
            return
        print(json.dumps(await download_model.remote.aio(), ensure_ascii=False, indent=2))
        return
    if args.command == "upload":
        plan = {"action": "upload", "local_pdf": str(args.pdf), "remote_pdf": args.remote_pdf}
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return
        print(json.dumps(await _upload_pdf(args.pdf, args.remote_pdf), ensure_ascii=False, indent=2))
        return

    try:
        plan = benchmark_plan(args) if args.command == "benchmark" else parse_plan(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    upload = await _upload_pdf(args.pdf, args.remote_pdf)
    model = await download_model.remote.aio()
    service = UnlimitedOCR.with_options(gpu=plan["gpu"])()
    if args.command == "benchmark":
        results = await asyncio.gather(
            *[
                service.benchmark.remote.aio(
                    concurrency=value,
                    runtime_seconds=plan["runtime_seconds"],
                    remote_pdf=plan["remote_pdf"],
                    start_page=plan["start_page"],
                    dpi=plan["dpi"],
                    max_tokens=plan["max_tokens"],
                    image_mode=plan["image_mode"],
                )
                for value in plan["concurrencies"]
            ]
        )
        compact = [{key: value for key, value in result.items() if key != "pages"} for result in results]
        print(json.dumps({"upload": upload, "model": model, "benchmarks": compact}, ensure_ascii=False, indent=2))
        return

    result = await service.parse_pdf.remote.aio(
        concurrency=plan["concurrency"],
        remote_pdf=plan["remote_pdf"],
        start_page=plan["start_page"],
        end_page=plan["end_page"],
        dpi=plan["dpi"],
        max_tokens=plan["max_tokens"],
        image_mode=plan["image_mode"],
        retries=plan["retries"],
        resume=plan["resume"],
    )
    print(json.dumps({"upload": upload, "model": model, "parse": result}, ensure_ascii=False, indent=2))


@app.local_entrypoint()
async def main(*argv: str) -> None:
    await cli(argv)


if __name__ == "__main__":
    asyncio.run(cli(sys.argv[1:]))
