"""ポリシーサーバー（提出用テンプレート）

このファイルを編集して、自分のモデルを組み込んでください。
編集が必要なのは MyPolicy クラスの中身だけです。
それ以外のコード（サーバー部分、シリアライゼーション）は変更不可です。

ローカルテスト:
    pip install -r requirements.txt
    python policy_server.py                  # サーバー起動（port 8000）

    # 別ターミナルで評価実行
    python -m pipeline --server-url http://localhost:8000 --dry-run
"""

import argparse
import sys
from abc import ABC, abstractmethod
from pathlib import Path

# バンドルされたlerobotディレクトリをsys.pathに追加
_script_dir = Path(__file__).parent
_lerobot_src = _script_dir / "lerobot" / "src"
if _lerobot_src.exists() and str(_lerobot_src) not in sys.path:
    sys.path.insert(0, str(_lerobot_src))

# バンドルされたVLMベースモデルのパス（オフライン環境用）
_base_model_path = _script_dir / "base_model"

import msgpack
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response


# ============================================================
# ポリシーのインターフェース定義（変更不可）
# MyPolicy が満たすべき get_action() / reset() の仕様を定める。
# ============================================================


class BasePolicy(ABC):
    """ポリシーの基底クラス。get_action() と reset() を実装してください。"""

    @abstractmethod
    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """観測からアクションを推論する。

        Args:
            obs: 環境からの観測。以下のキーが含まれる:
                - "agentview_image": (128, 128, 3) uint8
                - "robot0_eye_in_hand_image": (128, 128, 3) uint8
                - "robot0_joint_pos": (7,) float
                - "robot0_eef_pos": (3,) float
                - "robot0_eef_quat": (4,) float
                - "robot0_gripper_qpos": (2,) float

        Returns:
            action: (7,) float32 — [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        ...

    @abstractmethod
    def reset(self, instruction: str = "") -> None:
        """エピソード開始時に呼ばれる。内部状態をリセットしてください。

        Args:
            instruction: タスクの言語指示（例: "pick up the red mug and place it on the shelf"）
        """
        ...


# ============================================================
# ここを編集する（MyPolicy の中身だけを自分のモデルに置き換える）
# ============================================================


class MyPolicy(BasePolicy):
    """LeRobot (v0.4.4) で学習したモデルを使用するポリシー（SmolVLA / ACT 対応）。
    
    使い方:
        1. LeRobot v0.4.4をインストール
        2. 学習済みモデルを model_weights/ に配置
           (config.json, model.safetensors, policy_preprocessor*, policy_postprocessor* 等)
        3. python policy_server.py で起動
    
    対応モデル:
        - SmolVLA: VLM (Vision-Language Model) ベースのポリシー
        - ACT: Action Chunking Transformer
    
    注意: Python 3.10.12環境用 (v0.4.4が最後のPython 3.10対応版)
    """

    def __init__(self, model_path: str = "model_weights"):
        """LeRobotモデルをロードする。
        
        Args:
            model_path: モデルのディレクトリパス（config.json, model.safetensors等を含む）
                       デフォルトは "model_weights" (policy_server.pyと同じディレクトリからの相対パス)
        """
        import torch
        import json
        from pathlib import Path
        from lerobot.configs.policies import PreTrainedConfig
        
        print(f"[MyPolicy] Initializing policy from: {model_path}")
        
        # モデルパスの確認
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        config_file = model_path_obj / "config.json"
        weights_file = model_path_obj / "model.safetensors"
        
        if not config_file.exists():
            raise FileNotFoundError(f"config.json not found in {model_path}")
        if not weights_file.exists():
            raise FileNotFoundError(f"model.safetensors not found in {model_path}")
        
        print(f"[MyPolicy] Found config.json and model.safetensors")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MyPolicy] Using device: {self.device}")
        
        # config.jsonからポリシータイプを判定
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        self.policy_type = config_dict.get("type", "unknown").lower()
        print(f"[MyPolicy] Detected policy type: {self.policy_type}")
        
        # ポリシータイプに応じて必要な config クラスをインポート
        # (draccus のチョイスレジストリに登録するため、from_pretrained 前に必須)
        if self.policy_type == "act":
            from lerobot.policies.act.configuration_act import ACTConfig  # noqa: F401
        elif self.policy_type == "smolvla":
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
        
        # モデル設定をロード
        config = PreTrainedConfig.from_pretrained(model_path)
        config.device = self.device
        config.use_peft = False
        config.pretrained_path = None
        
        print(f"[MyPolicy] Config loaded:")
        print(f"  - use_peft: {config.use_peft}")
        if hasattr(config, 'chunk_size'):
            print(f"  - chunk_size: {config.chunk_size}")
        if hasattr(config, 'n_action_steps'):
            print(f"  - n_action_steps: {config.n_action_steps}")
        
        # deviceを文字列に変換（v0.4.4のsafetensors互換性のため）
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # ポリシータイプに応じてモデルをロード
        if self.policy_type == "smolvla":
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from transformers import AutoTokenizer
            
            # ベースモデルをローカルから読み込む（採点環境は外部通信遮断のため）
            if _base_model_path.exists():
                print(f"[MyPolicy] Using bundled base model: {_base_model_path}")
                config.vlm_model_name = str(_base_model_path)
            else:
                print(f"[MyPolicy] WARNING: base_model/ not found, will attempt download from HuggingFace")

            print(f"[MyPolicy] Loading SmolVLAPolicy...")
            print(f"  - load_vlm_weights: {config.load_vlm_weights}")
            print(f"  - vlm_model_name: {config.vlm_model_name}")
            
            try:
                self.policy = SmolVLAPolicy.from_pretrained(
                    model_path, 
                    config=config,
                    strict=True
                )
                self.policy.eval()
                print(f"[MyPolicy] SmolVLAPolicy loaded successfully")
            except Exception as e:
                print(f"[MyPolicy] ERROR during model loading: {type(e).__name__}: {e}")
                raise
            
            # トークナイザーをロード（SmolVLA用）
            print(f"[MyPolicy] Loading tokenizer: {config.vlm_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.vlm_model_name,
                padding_side="right",
                trust_remote_code=True
            )
            print(f"[MyPolicy] Tokenizer loaded successfully")
            
        elif self.policy_type == "act":
            from lerobot.policies.act.modeling_act import ACTPolicy
            
            print(f"[MyPolicy] Loading ACTPolicy...")
            
            try:
                self.policy = ACTPolicy.from_pretrained(
                    model_path,
                    config=config,
                    strict=True
                )
                self.policy.eval()
                print(f"[MyPolicy] ACTPolicy loaded successfully")
            except Exception as e:
                print(f"[MyPolicy] ERROR during model loading: {type(e).__name__}: {e}")
                raise
            
            self.tokenizer = None  # ACTは言語モデルを使わない
            
        else:
            raise ValueError(f"Unsupported policy type: {self.policy_type}. Supported types: smolvla, act")
        
        self.instruction = ""
        self.action_count = 0
        
    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """観測からアクションを推論する。
        
        Args:
            obs: 環境からの観測
            
        Returns:
            action: (7,) float32 — [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        import torch
        import torch.nn.functional as F
        
        # 最初の数回だけ詳細情報を出力
        if self.action_count < 3:
            print(f"\n[MyPolicy] get_action called (count: {self.action_count}, policy: {self.policy_type})")
            print(f"[MyPolicy] Observation keys: {list(obs.keys())}")
            for key, value in obs.items():
                print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
        
        # ポリシータイプに応じて推論
        try:
            if self.policy_type == "smolvla":
                return self._get_action_smolvla(obs)
            elif self.policy_type == "act":
                return self._get_action_act(obs)
            else:
                raise ValueError(f"Unsupported policy type: {self.policy_type}")
        except Exception as e:
            print(f"[MyPolicy] ERROR in get_action: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            # エラー時はゼロアクションを返す
            return np.zeros(7, dtype=np.float32)
    
    def _get_action_smolvla(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """SmolVLA用のアクション推論。"""
        import torch
        import torch.nn.functional as F
        
        # 観測データをLeRobot形式にマッピング
        mapped_obs = {}
        
        # 画像のマッピングと前処理
        # 学習時のrename_mapに合わせて camera1, camera2, camera3 を使用
        if 'agentview_image' in obs:
            img = obs['agentview_image']  # (128, 128, 3) uint8
            # シミュレータの出力はデータセットの向きに対して180度回転しているため補正
            img = np.ascontiguousarray(np.rot90(img, k=2))  # (128, 128, 3)
            # さらに左右鏡像に補正
            # img = np.ascontiguousarray(np.flip(img, axis=1))  # (128, 128, 3)
            # (H,W,C) -> (C,H,W) に転置
            img = np.transpose(img, (2, 0, 1))  # (3, 128, 128)
            # uint8 [0,255] -> float32 [0,1]
            img = img.astype(np.float32) / 255.0
            # tensor変換 (128,128) のまま渡す → モデル内 resize_with_pad(512,512) が直接スケールする
            img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, 3, 128, 128)
            mapped_obs['observation.images.camera1'] = img_tensor.to(self.device)
        
        if 'robot0_eye_in_hand_image' in obs:
            img = obs['robot0_eye_in_hand_image']  # (128, 128, 3) uint8
            # シミュレータの出力はデータセットの向きに対して180度回転しているため補正
            img = np.ascontiguousarray(np.rot90(img, k=2))  # (128, 128, 3)
            # さらに左右鏡像に補正
            # img = np.ascontiguousarray(np.flip(img, axis=1))  # (128, 128, 3)
            # (H,W,C) -> (C,H,W) に転置
            img = np.transpose(img, (2, 0, 1))  # (3, 128, 128)
            # uint8 [0,255] -> float32 [0,1]
            img = img.astype(np.float32) / 255.0
            # tensor変換 (128,128) のまま渡す → モデル内 resize_with_pad(512,512) が直接スケールする
            img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, 3, 128, 128)
            # 学習時のrename_map (observation.images.wrist -> camera3) に合わせる。
            # camera2は学習時に存在しない（empty_cameras=0）ため設定しない。
            mapped_obs['observation.images.camera3'] = img_tensor.to(self.device)
        
        # 状態データの結合 (8次元に)
        # robot0_eef_pos (3) + robot0_eef_quat (4) + robot0_gripper_qpos[0] (1) = 8
        state_components = []
        if 'robot0_eef_pos' in obs:
            state_components.append(obs['robot0_eef_pos'])  # (3,)
        if 'robot0_eef_quat' in obs:
            state_components.append(obs['robot0_eef_quat'])  # (4,)
        if 'robot0_gripper_qpos' in obs:
            state_components.append(obs['robot0_gripper_qpos'][:1])  # (1,) - 最初の要素のみ
        
        if state_components:
            state = np.concatenate(state_components, axis=0)  # (8,)
            # バッチ次元を追加してGPUに転送 (1, 8)
            mapped_obs['observation.state'] = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
        
        # 言語指示をトークン化
        task_text = self.instruction if hasattr(self, 'instruction') and self.instruction else ""
        if not task_text:
            task_text = "Pick and place task"  # デフォルトの指示
        
        # トークン化（policy_preprocessor.jsonの設定に従う）
        tokenized = self.tokenizer(
            task_text,
            padding="max_length",
            truncation=True,
            max_length=48,
            return_tensors="pt"
        )
        # トークンをGPUに転送
        mapped_obs['observation.language.tokens'] = tokenized['input_ids'].to(self.device)  # (1, 48)
        # attention_maskをbool型に変換してGPUに転送（torch.whereがboolean tensorを要求するため）
        mapped_obs['observation.language.attention_mask'] = tokenized['attention_mask'].bool().to(self.device)  # (1, 48)
        
        if self.action_count < 3:
            print(f"[MyPolicy] Mapped observation keys: {list(mapped_obs.keys())}")
            for key, value in mapped_obs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                else:
                    print(f"  - {key}: {type(value)}")
        
        with torch.no_grad():
            action = self.policy.select_action(mapped_obs)
        
        # numpy配列に変換
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        action_array = action.astype(np.float32)
        
        # 1次元の場合は2次元に変換 (7,) -> (1, 7)
        if action_array.ndim == 1:
            action_array = action_array[np.newaxis, :]
        
        # Y, Rx, Rz方向を反転（インデックス1, 3, 5）
        action_array[0, 1] *= -1  # Y方向
        action_array[0, 3] *= -1  # Rx方向
        action_array[0, 5] *= -1  # Rz方向
        
        # 1次元に変換して返す (1, 7) -> (7,)
        action_array = action_array.squeeze(0)
        
        # 最初の数回だけアクション情報を出力
        if self.action_count < 3:
            print(f"[MyPolicy] Action generated: shape={action_array.shape}, dtype={action_array.dtype}")
            print(f"[MyPolicy] Action values: {action_array}")
        
        self.action_count += 1
        
        return action_array
    
    def _get_action_act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """ACT用のアクション推論。"""
        import torch
        import torch.nn.functional as F
        
        # 観測データをLeRobot形式にマッピング
        mapped_obs = {}
        
        # 画像のマッピングと前処理（ACT用）
        # config.json の input_features キーに合わせて front / wrist を使用
        if 'agentview_image' in obs:
            img = obs['agentview_image']  # (128, 128, 3) uint8
            # シミュレータの出力はデータセットの向きに対して180度回転しているため補正
            img = np.ascontiguousarray(np.rot90(img, k=2))  # (128, 128, 3)
            # さらに左右鏡像に補正
            img = np.ascontiguousarray(np.flip(img, axis=1))  # (128, 128, 3)
            # (H,W,C) -> (C,H,W) に転置
            img = np.transpose(img, (2, 0, 1))  # (3, 128, 128)
            # uint8 [0,255] -> float32 [0,1]
            img = img.astype(np.float32) / 255.0
            # tensor変換 (128,128) のまま渡す → モデル内 resize_with_pad(512,512) が直接スケールする
            img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, 3, 128, 128)
            mapped_obs['observation.images.front'] = img_tensor.to(self.device)
        
        if 'robot0_eye_in_hand_image' in obs:
            img = obs['robot0_eye_in_hand_image']  # (128, 128, 3) uint8
            # シミュレータの出力はデータセットの向きに対して180度回転しているため補正
            img = np.ascontiguousarray(np.rot90(img, k=2))  # (128, 128, 3)
            # さらに左右鏡像に補正
            img = np.ascontiguousarray(np.flip(img, axis=1))  # (128, 128, 3)
            # (H,W,C) -> (C,H,W) に転置
            img = np.transpose(img, (2, 0, 1))  # (3, 128, 128)
            # uint8 [0,255] -> float32 [0,1]
            img = img.astype(np.float32) / 255.0
            # tensor変換 (128,128) のまま渡す → モデル内 resize_with_pad(512,512) が直接スケールする
            img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, 3, 128, 128)
            mapped_obs['observation.images.wrist'] = img_tensor.to(self.device)
        
        # 状態データの結合（ACT用に調整）
        # robot0_eef_pos (3) + robot0_eef_quat (4) + robot0_gripper_qpos[0] (1) = 8
        state_components = []
        if 'robot0_eef_pos' in obs:
            state_components.append(obs['robot0_eef_pos'])  # (3,)
        if 'robot0_eef_quat' in obs:
            state_components.append(obs['robot0_eef_quat'])  # (4,)
        if 'robot0_gripper_qpos' in obs:
            state_components.append(obs['robot0_gripper_qpos'][:1])  # (1,) - 最初の要素のみ
        
        if state_components:
            state = np.concatenate(state_components, axis=0)  # (8,)
            # バッチ次元を追加してGPUに転送 (1, 8)
            mapped_obs['observation.state'] = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
        
        if self.action_count < 3:
            print(f"[MyPolicy] Mapped observation keys: {list(mapped_obs.keys())}")
            for key, value in mapped_obs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                else:
                    print(f"  - {key}: {type(value)}")
        
        with torch.no_grad():
            action = self.policy.select_action(mapped_obs)
        
        # numpy配列に変換
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        action_array = action.astype(np.float32)
        
        # 1次元の場合は2次元に変換 (7,) -> (1, 7)
        if action_array.ndim == 1:
            action_array = action_array[np.newaxis, :]
        
        # Y, Rx, Rz方向を反転（インデックス1, 3, 5）
        action_array[0, 1] *= -1  # Y方向
        action_array[0, 3] *= -1  # Rx方向
        action_array[0, 5] *= -1  # Rz方向
        
        # 1次元に変換して返す (1, 7) -> (7,)
        action_array = action_array.squeeze(0)
        
        # 最初の数回だけアクション情報を出力
        if self.action_count < 3:
            print(f"[MyPolicy] Action generated: shape={action_array.shape}, dtype={action_array.dtype}")
            print(f"[MyPolicy] Action values: {action_array}")
        
        self.action_count += 1
        
        return action_array
    
    def reset(self, instruction: str = "") -> None:
        """エピソード開始時のリセット。
        
        Args:
            instruction: タスクの言語指示
        """
        print(f"\n[MyPolicy] reset called with instruction: '{instruction}'")
        self.instruction = instruction
        self.action_count = 0  # アクションカウンタをリセット
        
        try:
            self.policy.reset()
            print(f"[MyPolicy] Policy reset successful")
        except Exception as e:
            print(f"[MyPolicy] ERROR in reset: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()


# ============================================================
# 以下は変更不可
# ============================================================


def deserialize_obs(data: bytes) -> dict[str, np.ndarray]:
    unpacked = msgpack.unpackb(data, raw=False)
    obs = {}
    for key, val in unpacked.items():
        arr = np.frombuffer(val["data"], dtype=np.dtype(val["dtype"]))
        obs[key] = arr.reshape(val["shape"]).copy()
    return obs


def serialize_action(action: np.ndarray) -> bytes:
    return msgpack.packb(
        {"data": action.astype(np.float32).tobytes()},
        use_bin_type=True,
    )


app = FastAPI(title="VLA Policy Server")
_policy: BasePolicy | None = None


def set_policy(policy: BasePolicy) -> None:
    global _policy
    _policy = policy


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset_policy(request: Request):
    body = await request.body()
    instruction = ""
    if body:
        import json
        data = json.loads(body)
        instruction = data.get("instruction", "")
    _policy.reset(instruction=instruction)
    return {"status": "ok"}


@app.post("/act")
async def act(request: Request):
    body = await request.body()
    obs = deserialize_obs(body)
    action = _policy.get_action(obs)
    return Response(
        content=serialize_action(action),
        media_type="application/x-msgpack",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print("=" * 60)
    print("Initializing Policy Server")
    print("=" * 60)
    
    try:
        policy = MyPolicy()
        set_policy(policy)
        print("=" * 60)
        print(f"✓ Policy initialized successfully")
        print(f"Policy server starting on {args.host}:{args.port}")
        print("=" * 60)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except Exception as e:
        print("=" * 60)
        print(f"✗ Failed to initialize policy: {type(e).__name__}: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        raise
