import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Track(Enum):

    TRACK1 = "track1"


class PerturbationType(Enum):

    POSITION = "position"
    INSTRUCTION = "instruction"
    OBJECT = "object"
    COMPOSITION = "composition"


@dataclass
class PerturbationConfig:


    robot_init_pos_noise: float = 0.0
    object_pos_noise: float = 0.0
    camera_view_shift: float = 0.0


    action_noise: float = 0.0
    observation_noise: float = 0.0


    enable_composition: bool = False
    composition_length: int = 1


TRACK_PERTURBATIONS: dict[Track, PerturbationConfig] = {
    Track.TRACK1: PerturbationConfig(

    ),
}


@dataclass
class EvalConfig:


    camera_height: int = int(os.environ.get("LIBERO_EVAL_CAMERA", "128"))
    camera_width: int = int(os.environ.get("LIBERO_EVAL_CAMERA", "128"))
    max_steps_per_episode: int = 600
    n_eval_episodes: int = 20


    max_tasks: int | None = None


    task_ids: list[str] | None = None


    n_parallel_envs: int = 20
    use_subprocess: bool = True


    gpu_time_limit_sec: int = 3600
    episode_timeout_sec: int = 120


    benchmark_name: str = "libero_spatial"
    task_order_index: int = 0


    libero_root: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "LIBERO_ROOT",
                _PROJECT_ROOT.parent
                / "Isaac-GR00T"
                / "external_dependencies"
                / "LIBERO",
            )
        )
    )
    output_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("EVAL_OUTPUT_DIR", _PROJECT_ROOT / "results")
        )
    )


    device: str = "cpu"
    seed: int = 42
    
    # 動画保存設定
    save_video: bool = False
    video_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "videos"
    )


@dataclass
class SubmissionConstraints:


    max_zip_size_mb: int = 20 * 1024

    max_extracted_size_mb: int = 40 * 1024

    max_total_size_mb: int = 20 * 1024
    max_model_size_mb: int = 20 * 1024

    required_files: list[str] = field(
        default_factory=lambda: ["policy_server.py", "requirements.txt"]
    )

    required_endpoints: list[str] = field(
        default_factory=lambda: ["/health", "/reset", "/act"]
    )

    required_interface: str = "get_action"


TRACK_BENCHMARKS: dict[Track, list[str]] = {
    Track.TRACK1: [
        "libero_t1",
    ],
}
