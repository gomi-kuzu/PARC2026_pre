import json
import logging
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import (
    EvalConfig,
    PerturbationConfig,
    SubmissionConstraints,
    Track,
    TRACK_BENCHMARKS,
    TRACK_PERTURBATIONS,
)
from .environment import EnvironmentManager
from .rollout import PolicyInterface, RolloutExecutor, TaskResult
from .scorer import Scorer, TrackScore
from .validator import SubmissionValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:

    submission_id: str
    validation: ValidationResult
    track_scores: list[TrackScore]
    total_elapsed_sec: float
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "validation": {
                "is_valid": self.validation.is_valid,
                "errors": self.validation.errors,
                "warnings": self.validation.warnings,
            },
            "tracks": [ts.to_dict() for ts in self.track_scores],
            "total_elapsed_sec": self.total_elapsed_sec,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class EvaluationPipeline:

    def __init__(
        self,
        eval_config: EvalConfig | None = None,
        constraints: SubmissionConstraints | None = None,
    ):
        self.config = eval_config or EvalConfig()
        self.validator = SubmissionValidator(constraints)
        self.env_manager = EnvironmentManager(self.config)
        self.scorer = Scorer()


        self.executor = RolloutExecutor(
            self.env_manager, self.config, scoring_config=self.scorer.scoring_config
        )

    def run(
        self,
        submission_dir: Path,
        policy: PolicyInterface,
        tracks: list[Track] | None = None,
        submission_id: str | None = None,
        benchmark_override: str | None = None,
        skip_validation: bool = False,
    ) -> PipelineResult:
        start_time = time.time()
        timestamp = datetime.now(timezone.utc).isoformat()
        submission_id = submission_id or f"sub_{int(time.time())}"

        if tracks is None:
            tracks = [Track.TRACK1]

        logger.info(
            "=" * 60 + "\n評価パイプライン開始: %s\nトラック: %s\n" + "=" * 60,
            submission_id,
            [t.value for t in tracks],
        )


        logger.info("--- ステップ 1: バリデーション ---")
        if skip_validation:
            from .validator import ValidationResult

            validation = ValidationResult(is_valid=True, errors=[], warnings=[])
            logger.info("バリデーションをスキップ（サーバーモード / ドライラン）")
        else:
            validation = self.validator.validate(submission_dir)
        logger.info(str(validation))

        if not validation.is_valid:
            logger.error("バリデーション失敗。パイプラインを中断。")
            return PipelineResult(
                submission_id=submission_id,
                validation=validation,
                track_scores=[],
                total_elapsed_sec=time.time() - start_time,
                timestamp=timestamp,
            )


        track_scores: list[TrackScore] = []

        for track in tracks:
            logger.info("--- ステップ 2-4: Track %s 評価 ---", track.value)
            try:
                track_score = self._evaluate_track(policy, track, benchmark_override)
            except Exception as exc:


                logger.error(
                    "Track %s の評価に失敗しました: %s",
                    track.value,
                    exc,
                    exc_info=True,
                )
                track_score = self._errored_track_score(track, exc)
            track_scores.append(track_score)


        elapsed = time.time() - start_time
        result = PipelineResult(
            submission_id=submission_id,
            validation=validation,
            track_scores=track_scores,
            total_elapsed_sec=elapsed,
            timestamp=timestamp,
            metadata={
                "eval_config": {
                    "n_eval_episodes": self.config.n_eval_episodes,
                    "max_steps_per_episode": self.config.max_steps_per_episode,
                    "seed": self.config.seed,
                },
                "gpu_device": self.config.device,
            },
        )


        self._save_result(result)

        logger.info(
            "=" * 60 + "\n評価完了: %s\n合計時間: %.1f秒\n" + "=" * 60,
            submission_id,
            elapsed,
        )

        return result

    def _evaluate_track(
        self,
        policy: PolicyInterface,
        track: Track,
        benchmark_override: str | None = None,
    ) -> TrackScore:
        perturbation = TRACK_PERTURBATIONS[track]
        benchmarks = TRACK_BENCHMARKS[track]

        all_task_results: list[TaskResult] = []
        eval_benchmark = benchmark_override or benchmarks[-1]

        task_infos = self.env_manager.get_task_infos(eval_benchmark)
        task_infos = self._filter_by_task_ids(task_infos, eval_benchmark)
        if self.config.max_tasks is not None:
            task_infos = task_infos[:self.config.max_tasks]

        task_results = self.executor.evaluate_tasks(policy, task_infos, perturbation)
        all_task_results.extend(task_results)

        # save_trajectory 用に生の TaskResult を保持しておく
        self._last_task_results = getattr(self, "_last_task_results", [])
        self._last_task_results.extend(all_task_results)

        track_score = self.scorer.score_track(track, eval_benchmark, all_task_results)

        return track_score

    def _filter_by_task_ids(self, task_infos: list, benchmark: str) -> list:
        requested = self.config.task_ids
        if not requested:
            return task_infos

        available = {t.name for t in task_infos}
        unknown = [name for name in requested if name not in available]
        if unknown:
            raise RuntimeError(
                f"--tasks に {benchmark} に存在しないタスクが指定されました: {unknown}。"
                f" 利用可能なタスク: {sorted(available)}"
            )

        wanted = set(requested)
        filtered = [t for t in task_infos if t.name in wanted]
        logger.info(
            "--tasks で絞り込み: %d → %d タスク", len(task_infos), len(filtered)
        )
        return filtered

    def _errored_track_score(self, track: Track, exc: Exception) -> TrackScore:
        benchmarks = TRACK_BENCHMARKS.get(track) or ["?"]
        return TrackScore(
            track=track,
            benchmark_name=benchmarks[-1],
            task_scores=[],
            overall_score=0.0,
            overall_metrics={"error": 1.0, "n_tasks": 0.0},
        )

    def _save_result(self, result: PipelineResult) -> None:
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{result.submission_id}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info("結果を保存: %s", output_path)

        # 軌跡データ保存 (--save-trajectory 指定時)
        if self.config.save_trajectory:
            self._save_trajectories(result)

    def _save_trajectories(self, result: PipelineResult) -> None:
        """各エピソードの state / action 軌跡を npz に保存する。

        npz 内のキー:
          ee_positions   : (T, 3)  — robot0_eef_pos の時系列
          ee_orientations: (T, 4)  — robot0_eef_quat (xyzw) の時系列
          gripper_qpos   : (T, 2)  — robot0_gripper_qpos の時系列
          actions        : (T, 7)  — policy が返したアクション [dx,dy,dz,droll,dpitch,dyaw,gripper]
          joint_positions: (T, 7)  — robot0_joint_pos の時系列
          rewards        : (T,)    — 報酬の時系列
          success        : scalar bool
          task_name      : str (in meta.json)
        """
        import numpy as np

        traj_dir = self.config.output_dir / "trajectories" / result.submission_id
        traj_dir.mkdir(parents=True, exist_ok=True)

        n_saved = 0
        for ts in result.track_scores:
            for task_result in ts.task_scores:
                # task_result は TaskScore (scorer.py) で TaskResult は executor 内に留まるため
                # pipeline.run() 内で task_result を直接保持していない。
                # ここでは track_score が保持している生データを使う。
                pass  # 後述の _evaluate_track を wrap する形で実装

        # track_results が result に含まれていないため、
        # _evaluate_track をオーバーライドして EpisodeResult を横取りする。
        # ここでは self._last_task_results に格納しておく（下記 _evaluate_track 参照）。
        task_results = getattr(self, "_last_task_results", [])
        for task_result in task_results:
            for ep in task_result.episodes:
                if not ep.actions:
                    continue
                fname = f"{ep.task_name}_ep{ep.episode_id:03d}.npz"
                fpath = traj_dir / fname
                np.savez(
                    fpath,
                    ee_positions=np.array(ep.ee_positions, dtype=np.float32),
                    ee_orientations=np.array(ep.ee_orientations, dtype=np.float32),
                    gripper_qpos=np.array(ep.gripper_qpos, dtype=np.float32),
                    actions=np.array(ep.actions, dtype=np.float32),
                    joint_positions=np.array(ep.joint_positions, dtype=np.float32),
                    rewards=np.array(ep.rewards, dtype=np.float32),
                    success=np.bool_(ep.success),
                )
                n_saved += 1
        logger.info("軌跡データを %d エピソード保存: %s", n_saved, traj_dir)

    def run_single_track(
        self,
        policy: PolicyInterface,
        track: Track,
        benchmark_name: str | None = None,
    ) -> TrackScore:
        if benchmark_name:
            perturbation = TRACK_PERTURBATIONS[track]
            task_infos = self.env_manager.get_task_infos(benchmark_name)
            task_results = self.executor.evaluate_tasks(
                policy, task_infos, perturbation
            )
            return self.scorer.score_track(track, benchmark_name, task_results)

        return self._evaluate_track(policy, track)
