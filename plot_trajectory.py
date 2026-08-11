#!/usr/bin/env python3
"""評価ログ (npz) から state / action の時系列をプロットするスクリプト。

使い方:
    # 単一エピソード
    python plot_trajectory.py results/trajectories/server_8000/TASK_NAME_ep000.npz

    # 複数エピソード (glob で一括)
    python plot_trajectory.py results/trajectories/server_8000/*.npz

    # 保存先を指定
    python plot_trajectory.py results/trajectories/server_8000/*.npz --out-dir /tmp/plots

評価コマンド例 (Docker コンテナ内):
    python -m pipeline --server-url http://localhost:8000 --track track1 \\
        --n-episodes 1 --max-steps 300 --save-trajectory
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # Docker (ヘッドレス環境) 対応: 画面を使わず PNG に書き出す
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# 各軸のラベル定義
# ---------------------------------------------------------------------------
STATE_LABELS = {
    "ee_pos":    ["x", "y", "z"],
    "ee_quat":   ["qx", "qy", "qz", "qw"],   # xyzw 順
    "gripper":   ["grip_L", "grip_R"],
}

ACTION_LABELS = ["dx", "dy", "dz", "droll (drx)", "dpitch (dry)", "dyaw (drz)", "gripper"]


def load_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def plot_episode(data: dict, title: str, out_path: Path) -> None:
    """1 エピソード分の state + action をひとつの PNG に描画する。"""
    T = len(data["actions"])  # タイムステップ数
    t = np.arange(T)

    success = bool(data["success"]) if "success" in data else None
    title_suffix = ""
    if success is not None:
        title_suffix = " [SUCCESS]" if success else " [FAIL]"

    # ----- レイアウト -----
    # 上段: ee_pos (3), ee_quat (4), gripper_qpos (2) = 3 グループ
    # 下段: action (7 = dx,dy,dz,droll,dpitch,dyaw,gripper)
    n_state_rows = 3   # ee_pos / ee_quat / gripper_qpos
    n_action_rows = 2  # action_xyz / action_rot+gripper
    n_rows = n_state_rows + n_action_rows
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3 * n_rows), sharex=True)
    fig.suptitle(title + title_suffix, fontsize=12, fontweight="bold")

    # ---- EE position ----
    ax = axes[0]
    ee_pos = data["ee_positions"]  # (T, 3)
    for i, label in enumerate(STATE_LABELS["ee_pos"]):
        ax.plot(t, ee_pos[:, i], label=label)
    ax.set_ylabel("EEF position [m]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- EE orientation (quat xyzw) ----
    ax = axes[1]
    ee_quat = data["ee_orientations"]  # (T, 4)
    for i, label in enumerate(STATE_LABELS["ee_quat"]):
        ax.plot(t, ee_quat[:, i], label=label)
    ax.set_ylabel("EEF quat (xyzw)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Gripper qpos ----
    ax = axes[2]
    grip = data["gripper_qpos"]  # (T, 2)
    for i, label in enumerate(STATE_LABELS["gripper"]):
        ax.plot(t, grip[:, i], label=label)
    ax.set_ylabel("Gripper qpos")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Action: xyz (並進) ----
    ax = axes[3]
    actions = data["actions"]  # (T, 7)
    for i, label in enumerate(ACTION_LABELS[:3]):
        ax.plot(t, actions[:, i], label=label)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Action: xyz")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Action: rotation + gripper ----
    ax = axes[4]
    for i, label in enumerate(ACTION_LABELS[3:]):
        ax.plot(t, actions[:, 3 + i], label=label)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Action: rot + grip")
    ax.set_xlabel("Step")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_multi_episode_summary(all_data: list[dict], all_titles: list[str], out_path: Path) -> None:
    """複数エピソードの action を重ね書きした比較グラフ。"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
    fig.suptitle("Action 比較 (全エピソード)", fontsize=12, fontweight="bold")

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for ep_idx, (data, title) in enumerate(zip(all_data, all_titles)):
        T = len(data["actions"])
        t = np.arange(T)
        actions = data["actions"]
        success = bool(data.get("success", False))
        ls = "-" if success else "--"
        c = colors[ep_idx % len(colors)]
        label = f"ep{ep_idx} {'OK' if success else 'NG'}"

        axes[0].plot(t, actions[:, 0], color=c, linestyle=ls, alpha=0.7, label=f"{label} dx")
        axes[0].plot(t, actions[:, 1], color=c, linestyle=":", alpha=0.5, label=f"{label} dy")
        axes[0].plot(t, actions[:, 2], color=c, linestyle="-.", alpha=0.5, label=f"{label} dz")

        axes[1].plot(t, actions[:, 3], color=c, linestyle=ls, alpha=0.7, label=f"{label} droll")
        axes[1].plot(t, actions[:, 4], color=c, linestyle=":", alpha=0.5, label=f"{label} dpitch")
        axes[1].plot(t, actions[:, 5], color=c, linestyle="-.", alpha=0.5, label=f"{label} dyaw")
        axes[1].plot(t, actions[:, 6], color=c, linestyle=(0, (3, 1)), alpha=0.5, label=f"{label} grip")

    for ax, ylabel in zip(axes, ["Action: xyz", "Action: rot + grip"]):
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
        # 凡例は多くなるので外に出す
        ax.legend(
            loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=6,
            borderaxespad=0, ncol=2,
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved (summary): {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="軌跡 npz の可視化")
    parser.add_argument(
        "npz_files",
        nargs="+",
        type=Path,
        metavar="NPZ_FILE",
        help="plot_trajectory.py で指定する .npz ファイル (glob 可)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots"),
        help="グラフの保存先ディレクトリ (デフォルト: ./plots/)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="複数エピソード比較グラフを生成しない",
    )
    args = parser.parse_args()

    npz_paths = sorted(args.npz_files)
    if not npz_paths:
        print("エラー: npz ファイルが指定されていません。", file=sys.stderr)
        sys.exit(1)

    all_data: list[dict] = []
    all_titles: list[str] = []

    for path in npz_paths:
        if not path.exists():
            print(f"警告: {path} が見つかりません。スキップします。", file=sys.stderr)
            continue

        data = load_npz(path)
        title = path.stem  # ファイル名（拡張子なし）をタイトルに使う
        out_png = args.out_dir / (path.stem + ".png")

        plot_episode(data, title=title, out_path=out_png)
        all_data.append(data)
        all_titles.append(title)

    # 複数エピソード比較
    if len(all_data) > 1 and not args.no_summary:
        summary_path = args.out_dir / "_summary_actions.png"
        plot_multi_episode_summary(all_data, all_titles, out_path=summary_path)

    print(f"\n完了: {len(all_data)} エピソードを {args.out_dir} に保存しました。")


if __name__ == "__main__":
    main()
