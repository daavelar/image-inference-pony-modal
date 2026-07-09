from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

# Modal list prices (USD/s) — https://modal.com/pricing
GPU_COST_PER_SECOND: dict[str, float] = {
    "B300": 0.001972,
    "B200": 0.001736,
    "H200": 0.001261,
    "H100": 0.001097,
    "RTX-PRO-6000": 0.000842,
    "A100-80GB": 0.000694,
    "A100-40GB": 0.000583,
    "L40S": 0.000542,
    "A10G": 0.000306,
    "A10": 0.000306,
    "L4": 0.000222,
    "T4": 0.000164,
}

CPU_COST_PER_CORE_SECOND = 0.0000131
MEMORY_COST_PER_GIB_SECOND = 0.00000222
MIN_CPU_CORES = 0.125


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _query_gpu_name() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            text=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""


def resolve_gpu_cost_per_second(gpu_name: str) -> tuple[str, float]:
    normalized = gpu_name.upper()
    if not normalized:
        return "H200", GPU_COST_PER_SECOND["H200"]
    if "B300" in normalized:
        return "B300", GPU_COST_PER_SECOND["B300"]
    if "B200" in normalized:
        return "B200", GPU_COST_PER_SECOND["B200"]
    if "H200" in normalized:
        return "H200", GPU_COST_PER_SECOND["H200"]
    if "H100" in normalized:
        return "H100", GPU_COST_PER_SECOND["H100"]
    if "RTX PRO 6000" in normalized or "RTX 6000" in normalized:
        return "RTX-PRO-6000", GPU_COST_PER_SECOND["RTX-PRO-6000"]
    if "A100" in normalized and "80" in normalized:
        return "A100-80GB", GPU_COST_PER_SECOND["A100-80GB"]
    if "A100" in normalized:
        return "A100-40GB", GPU_COST_PER_SECOND["A100-40GB"]
    if "L40S" in normalized:
        return "L40S", GPU_COST_PER_SECOND["L40S"]
    if "A10G" in normalized or "A10" in normalized:
        return "A10G", GPU_COST_PER_SECOND["A10G"]
    if "L4" in normalized:
        return "L4", GPU_COST_PER_SECOND["L4"]
    if "T4" in normalized:
        return "T4", GPU_COST_PER_SECOND["T4"]
    return "H200", GPU_COST_PER_SECOND["H200"]


def get_gpu_name() -> str:
    return _query_gpu_name()


def detect_gpu_cost_per_second() -> float:
    return resolve_gpu_cost_per_second(_query_gpu_name())[1]


@dataclass
class _JobContext:
    started_at: float
    idle_before_seconds: float = 0.0
    bootstrap_seconds: float = 0.0


@dataclass
class ContainerCostTracker:
    """Approximate Modal per-job billing from container wall time."""

    scaledown_window: float = 300.0
    tail_idle_seconds: float | None = None
    cpu_cores: float = 2.0
    memory_gib: float = 16.0
    region_multiplier: float = 1.0

    _bootstrap_seconds: float = field(default=0.0, init=False, repr=False)
    _last_job_finished_at: float | None = field(default=None, init=False, repr=False)
    _job_count: int = field(default=0, init=False, repr=False)

    def on_boot_start(self) -> float:
        return time.perf_counter()

    def on_boot_complete(self, boot_started_at: float) -> None:
        self._bootstrap_seconds = max(0.0, time.perf_counter() - boot_started_at)

    def begin_job(self) -> _JobContext:
        now = time.perf_counter()
        idle_before = 0.0
        if self._last_job_finished_at is not None:
            idle_before = max(0.0, now - self._last_job_finished_at)
            if idle_before > self.scaledown_window:
                idle_before = 0.0

        bootstrap = self._bootstrap_seconds if self._job_count == 0 else 0.0
        self._job_count += 1
        return _JobContext(
            started_at=now,
            idle_before_seconds=idle_before,
            bootstrap_seconds=bootstrap,
        )

    def finish_job(self, ctx: _JobContext) -> dict[str, Any]:
        finished_at = time.perf_counter()
        job_wall = max(0.0, finished_at - ctx.started_at)
        tail_idle = (
            self.tail_idle_seconds
            if self.tail_idle_seconds is not None
            else _env_float("MODAL_TAIL_IDLE_SECONDS", self.scaledown_window * 0.4)
        )

        billed_seconds = (
            ctx.bootstrap_seconds + ctx.idle_before_seconds + job_wall + tail_idle
        )

        gpu_name = _query_gpu_name()
        gpu_label, gpu_rate = resolve_gpu_cost_per_second(gpu_name)
        region = _env_float("MODAL_REGION_MULTIPLIER", self.region_multiplier)
        cpu_cores = max(MIN_CPU_CORES, _env_float("MODAL_CPU_CORES", self.cpu_cores))
        memory_gib = _env_float("MODAL_MEMORY_GIB", self.memory_gib)

        gpu_cost = billed_seconds * gpu_rate
        cpu_cost = billed_seconds * cpu_cores * CPU_COST_PER_CORE_SECOND
        memory_cost = billed_seconds * memory_gib * MEMORY_COST_PER_GIB_SECOND
        total = (gpu_cost + cpu_cost + memory_cost) * region

        self._last_job_finished_at = finished_at

        return {
            "cost": round(total, 6),
            "billed_seconds": round(billed_seconds, 2),
            "job_wall_seconds": round(job_wall, 2),
            "idle_before_seconds": round(ctx.idle_before_seconds, 2),
            "bootstrap_seconds": round(ctx.bootstrap_seconds, 2),
            "tail_idle_seconds": round(tail_idle, 2),
            "gpu": gpu_label,
            "gpu_cost_usd": round(gpu_cost * region, 6),
            "cpu_memory_cost_usd": round((cpu_cost + memory_cost) * region, 6),
        }

    @classmethod
    def from_env(cls, scaledown_window: float = 300.0) -> ContainerCostTracker:
        tail_raw = os.environ.get("MODAL_TAIL_IDLE_SECONDS", "").strip()
        tail_val = float(tail_raw) if tail_raw else None
        return cls(
            scaledown_window=scaledown_window,
            tail_idle_seconds=tail_val,
            cpu_cores=_env_float("MODAL_CPU_CORES", 2.0),
            memory_gib=_env_float("MODAL_MEMORY_GIB", 16.0),
            region_multiplier=_env_float("MODAL_REGION_MULTIPLIER", 1.0),
        )


def apply_job_cost(result: dict[str, Any], billing: dict[str, Any]) -> dict[str, Any]:
    if result.get("error"):
        return result
    result["cost"] = billing["cost"]
    result["billed_seconds"] = billing["billed_seconds"]
    result["cost_breakdown"] = {
        key: billing[key]
        for key in (
            "job_wall_seconds",
            "idle_before_seconds",
            "bootstrap_seconds",
            "tail_idle_seconds",
            "gpu",
            "gpu_cost_usd",
            "cpu_memory_cost_usd",
        )
    }
    return result


def run_with_billing(
    tracker: ContainerCostTracker,
    fn: Any,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    ctx = tracker.begin_job()
    try:
        result = fn(*args, **kwargs)
    except Exception:
        tracker._job_count = max(0, tracker._job_count - 1)
        raise
    if not isinstance(result, dict):
        return {"error": f"unexpected job result type: {type(result).__name__}"}
    return apply_job_cost(result, tracker.finish_job(ctx))
