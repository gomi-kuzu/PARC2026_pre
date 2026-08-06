import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .config import EvalConfig, PerturbationConfig
from .environment import EnvironmentManager, TaskInfo
from .total_score import load_scoring_config

logger = logging.getLogger(__name__)


class PolicyInterface(Protocol):

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        ...

    def reset(self, instruction: str = "", seed: int | None = None) -> None:
        ...


@dataclass
class EpisodeResult:
    task_name: str
    episode_id: int
    success: bool
    total_steps: int
    elapsed_time_sec: float

    joint_positions: list[np.ndarray] = field(default_factory=list)
    ee_positions: list[np.ndarray] = field(default_factory=list)
    ee_orientations: list[np.ndarray] = field(default_factory=list)
    gripper_qpos: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    
    # 動画保存用
    render_frames: list[np.ndarray] = field(default_factory=list)
    camera_frames: dict[str, list[np.ndarray]] = field(default_factory=dict)

    collided: bool = False

    @property
    def trajectory(self) -> list[np.ndarray]:
        return self.joint_positions


@dataclass
class TaskResult:
    task_info: TaskInfo
    episodes: list[EpisodeResult]

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.success) / len(self.episodes)

    @property
    def avg_steps(self) -> float:
        successful = [e for e in self.episodes if e.success]
        if not successful:
            return 0.0
        return sum(e.total_steps for e in successful) / len(successful)

    @property
    def avg_time(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.elapsed_time_sec for e in self.episodes) / len(self.episodes)


class RolloutExecutor:

    def __init__(
        self,
        env_manager: EnvironmentManager,
        eval_config: EvalConfig,
        scoring_config: dict | None = None,
    ):
        self.env_manager = env_manager
        self.config = eval_config


        self.scoring_config = scoring_config or load_scoring_config()

    def evaluate_task(
        self,
        policy: PolicyInterface,
        task_info: TaskInfo,
        perturbation: PerturbationConfig,
    ) -> TaskResult:
        logger.info(
            "タスク評価開始: %s (%d エピソード)",
            task_info.name, self.config.n_eval_episodes,
        )

        env = self.env_manager.create_env(task_info)


        init_states = self.env_manager.get_perturbed_init_states(
            task_info, perturbation, self.config.n_eval_episodes
        )


        collision_enabled = bool(self.scoring_config.get("collision", {}).get("enabled", True))
        obj_of_interest = (
            self.env_manager.get_obj_of_interest(task_info) if collision_enabled else set()
        )

        episodes: list[EpisodeResult] = []

        try:
            for ep_id in range(self.config.n_eval_episodes):
                result = self._run_episode(
                    env=env,
                    policy=policy,
                    task_info=task_info,
                    init_state=init_states[ep_id],
                    episode_id=ep_id,
                    perturbation=perturbation,
                    obj_of_interest=obj_of_interest,
                )
                episodes.append(result)

                if result.success:
                    logger.debug(
                        "  Episode %d: 成功 (%d steps)", ep_id, result.total_steps
                    )
                else:
                    logger.debug(
                        "  Episode %d: 失敗 (%d steps)", ep_id, result.total_steps
                    )
        finally:
            env.close()

        task_result = TaskResult(task_info=task_info, episodes=episodes)
        logger.info(
            "タスク評価完了: %s — 成功率 %.1f%% (平均 %.1f steps)",
            task_info.name, task_result.success_rate * 100, task_result.avg_steps,
        )
        return task_result

    def _run_episode(
        self,
        env: Any,
        policy: PolicyInterface,
        task_info: TaskInfo,
        init_state: np.ndarray,
        episode_id: int,
        perturbation: PerturbationConfig,
        obj_of_interest: set[str],
    ) -> EpisodeResult:
        start_time = time.time()
        joint_positions: list[np.ndarray] = []
        ee_positions: list[np.ndarray] = []
        ee_orientations: list[np.ndarray] = []
        gripper_qpos_log: list[np.ndarray] = []
        actions_log: list[np.ndarray] = []
        rewards_log: list[float] = []

        cc = self.scoring_config.get("collision", {})
        collision_enabled = bool(cc.get("enabled", True))
        collision_threshold = float(cc.get("threshold_m", 0.001))


        env.reset()
        env.sim.set_state_from_flattened(init_state)
        env.sim.forward()


        action_dim = env.robots[0].action_dim
        dummy_action = np.zeros(action_dim)
        for _ in range(10):
            obs, _, _, _ = env.step(dummy_action)


        object_init_pos: dict[str, np.ndarray] = {}
        if collision_enabled:
            object_init_pos = {
                k[:-4]: np.asarray(obs[k]).copy()
                for k in obs
                if k.endswith("_pos")
                and not k.startswith("robot0")
                and not k.endswith("_to_robot0_eef_pos")
                and k[:-4] not in obj_of_interest
            }
        object_max_disp: dict[str, float] = {}


        episode_seed = self.config.seed + episode_id
        policy.reset(instruction=task_info.language, seed=episode_seed)
        done = False
        total_steps = 0
        
        # 動画保存用のフレームバッファ
        render_frames: list[np.ndarray] = []
        camera_frames: dict[str, list[np.ndarray]] = {}
        if self.config.save_video:
            camera_frames["agentview"] = []
            camera_frames["wrist"] = []

        for step in range(self.config.max_steps_per_episode):

            obs_for_policy = self.env_manager.apply_observation_noise(
                obs, perturbation
            )


            action = policy.get_action(obs_for_policy)


            action = self.env_manager.apply_action_noise(action, perturbation)


            obs, reward, done, info = env.step(action)


            joint_positions.append(obs.get("robot0_joint_pos", np.zeros(7)).copy())
            ee_positions.append(obs.get("robot0_eef_pos", np.zeros(3)).copy())
            ee_orientations.append(obs.get("robot0_eef_quat", np.array([1, 0, 0, 0], dtype=np.float64)).copy())
            gripper_qpos_log.append(obs.get("robot0_gripper_qpos", np.zeros(2)).copy())
            actions_log.append(action.copy())
            rewards_log.append(float(reward))
            
            # 動画保存: レンダリング画像と観測カメラ画像を収集
            if self.config.save_video:
                try:
                    # ロボットの様子（レンダリング画像）
                    # OffScreenRenderEnv には gym 形式の render() が存在しないため、
                    # env.sim（MuJoCo simulation）から直接オフスクリーンレンダリングする
                    render_img = env.sim.render(
                        camera_name="frontview",
                        height=480,
                        width=480,
                    )[::-1]
                    render_frames.append(render_img)
                    
                    # 観測カメラ画像
                    if "agentview_image" in obs:
                        camera_frames["agentview"].append(obs["agentview_image"].copy())
                    if "robot0_eye_in_hand_image" in obs:
                        camera_frames["wrist"].append(obs["robot0_eye_in_hand_image"].copy())
                except Exception as e:
                    logger.warning("動画フレーム取得エラー: %s", e)


            for name, p0 in object_init_pos.items():
                cur = obs.get(name + "_pos")
                if cur is not None:
                    d = float(np.sum(np.abs(np.asarray(cur) - p0)))
                    if d > object_max_disp.get(name, 0.0):
                        object_max_disp[name] = d

            total_steps = step + 1

            if total_steps % 50 == 0:
                logger.info(
                    "  [進捗] %s: %d/%d steps (%.1fs)",
                    task_info.name, total_steps, self.config.max_steps_per_episode,
                    time.time() - start_time,
                )

            if done:
                break

        elapsed = time.time() - start_time


        collided = any(d > collision_threshold for d in object_max_disp.values())
        success = bool(done) and not collided
        
        result = EpisodeResult(
            task_name=task_info.name,
            episode_id=episode_id,
            success=success,
            total_steps=total_steps,
            elapsed_time_sec=elapsed,
            joint_positions=joint_positions,
            ee_positions=ee_positions,
            ee_orientations=ee_orientations,
            gripper_qpos=gripper_qpos_log,
            actions=actions_log,
            rewards=rewards_log,
            collided=collided,
            render_frames=render_frames,
            camera_frames=camera_frames,
        )
        
        # 動画保存
        if self.config.save_video and (render_frames or camera_frames):
            self._save_episode_videos(result, task_info)
        
        return result

    def _save_episode_videos(
        self,
        episode: EpisodeResult,
        task_info: TaskInfo,
    ) -> None:
        """エピソードの動画を保存"""
        try:
            import imageio
        except ImportError:
            logger.warning("imageio がインストールされていません。動画保存をスキップします。")
            return
        
        # 出力ディレクトリを作成
        video_dir = self.config.video_dir / task_info.name
        video_dir.mkdir(parents=True, exist_ok=True)
        
        success_str = "success" if episode.success else "fail"
        prefix = f"ep{episode.episode_id:03d}_{success_str}"
        
        # レンダリング動画を保存
        if episode.render_frames:
            render_path = video_dir / f"{prefix}_render.mp4"
            try:
                # is_batch=False必須: imageioのpyavプラグインはデフォルト(is_batch=True)だと
                # append_dataに渡した1枚のフレーム(H,W,C)をバッチ(フレーム枚数,H,W)と誤認識し、
                # 中身が壊れた0バイトファイルになる
                writer = imageio.get_writer(
                    render_path, fps=20, codec="libx264", is_batch=False
                )
                for frame in episode.render_frames:
                    writer.append_data(frame)
                writer.close()
                logger.info("レンダリング動画を保存: %s", render_path)
            except Exception as e:
                logger.warning("レンダリング動画保存エラー: %s", e)
        
        # 観測カメラ動画を保存
        for cam_name, frames in episode.camera_frames.items():
            if not frames:
                continue
            cam_path = video_dir / f"{prefix}_{cam_name}.mp4"
            try:
                writer = imageio.get_writer(
                    cam_path, fps=20, codec="libx264", is_batch=False
                )
                for frame in frames:
                    # (H, W, C) uint8 形式に変換
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                    writer.append_data(frame)
                writer.close()
                logger.info("%s カメラ動画を保存: %s", cam_name, cam_path)
            except Exception as e:
                logger.warning("%s カメラ動画保存エラー: %s", cam_name, e)

    def evaluate_tasks(
        self,
        policy: PolicyInterface,
        task_infos: list[TaskInfo],
        perturbation: PerturbationConfig,
    ) -> list[TaskResult]:
        results = []
        for i, task_info in enumerate(task_infos):
            logger.info(
                "=== タスク %d/%d: %s ===",
                i + 1, len(task_infos), task_info.name,
            )
            result = self.evaluate_task(policy, task_info, perturbation)
            results.append(result)
        return results
