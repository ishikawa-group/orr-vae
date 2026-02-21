"""Configuration for Pt-Ni example workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class GenerationSettings:
    elements: tuple[str, str] = ("Pt", "Ni")
    size: tuple[int, int, int] = (4, 4, 6)
    vacuum: float | None = None
    initial_num_structures: int = 128
    generated_num_structures: int = 128
    min_fraction_secondary: float = 1.0 / 64.0
    max_fraction_secondary: float = 63.0 / 64.0


@dataclass(frozen=True)
class WorkflowSettings:
    seed: int
    label_threshold: float
    batch_size: int
    max_epoch: int
    latent_size: int
    beta: float
    max_iter: int
    calculator: str
    with_visualization: bool
    with_analysis: bool
    job_num: str
    root_dir: Path
    example_dir: Path
    output_dir: Path
    data_dir: Path
    result_dir: Path
    log_dir: Path
    solvent_correction_yaml: Path


@dataclass(frozen=True)
class RuntimeSettings:
    generation: GenerationSettings
    workflow: WorkflowSettings


def load_settings() -> RuntimeSettings:
    root_dir = Path(__file__).resolve().parents[3]
    example_dir = Path(__file__).resolve().parents[1]

    seed = _env_int("SEED", 0)
    job_num = os.getenv("JOB_NUM", "manual")

    output_dir = Path(os.getenv("OUTPUT_DIR", str(example_dir / "results" / f"seed_{seed}" / job_num))).resolve()
    data_dir = Path(os.getenv("DATA_DIR", str(output_dir / "data"))).resolve()
    result_dir = Path(os.getenv("RESULT_DIR", str(output_dir / "result"))).resolve()
    log_dir = Path(os.getenv("LOG_DIR", str(result_dir / "log"))).resolve()

    generation = GenerationSettings(
        size=(
            _env_int("GRID_X", 4),
            _env_int("GRID_Y", 4),
            _env_int("GRID_Z", 6),
        ),
        vacuum=float(os.getenv("VACUUM", "0")) if os.getenv("VACUUM") not in (None, "none", "None") else None,
        initial_num_structures=_env_int("INITIAL_NUM_STRUCTURES", 128),
        generated_num_structures=_env_int("GENERATED_NUM_STRUCTURES", 128),
        min_fraction_secondary=_env_float("MIN_SECONDARY_FRACTION", 1.0 / 64.0),
        max_fraction_secondary=_env_float("MAX_SECONDARY_FRACTION", 63.0 / 64.0),
    )

    workflow = WorkflowSettings(
        seed=seed,
        label_threshold=_env_float("LABEL_THRESHOLD", 0.3),
        batch_size=_env_int("BATCH_SIZE", 16),
        max_epoch=_env_int("MAX_EPOCH", 200),
        latent_size=_env_int("LATENT_SIZE", 32),
        beta=_env_float("BETA", 1.0),
        max_iter=_env_int("MAX_ITER", 5),
        calculator=os.getenv("CALCULATOR", "fairchem"),
        with_visualization=_env_bool("WITH_VISUALIZATION", True),
        with_analysis=_env_bool("WITH_ANALYSIS", True),
        job_num=job_num,
        root_dir=root_dir,
        example_dir=example_dir,
        output_dir=output_dir,
        data_dir=data_dir,
        result_dir=result_dir,
        log_dir=log_dir,
        solvent_correction_yaml=Path(
            os.getenv("SOLVENT_CORRECTION_YAML", str(Path(__file__).resolve().parent / "solvent_correction.yaml"))
        ).resolve(),
    )

    return RuntimeSettings(generation=generation, workflow=workflow)
