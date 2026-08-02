# PARC2026 LeRobot SmolVLA Tips & Memo

## 問題の概要

- LeRobot v0.6.0で学習したSmolVLAモデルをpolicy_server.pyで読み込もうとしたが、評価結果が0%になる問題が発生
- モデルが正しく読み込まれているか確認が必要

## 環境情報

### コンテナ環境（開発・評価用）

#### 初期セットアップ環境: `gifted_merkle`
- Python: 3.10.12
- 作業ディレクトリ: `/workspace/submission_template/`
- 検証・デバッグ用に使用

#### 本番想定環境: `for_parc2026`
- Python: 3.10.12
- PyTorch: 2.10.0+cpu
- torchvision: 0.25.0+cpu
- 作業ディレクトリ: `/workspace`
- ボリュームマウント: `./submission:/workspace/submission`
- 実際の提出用コードで動作確認

### 本番環境（判明した仕様）
- Python: 3.10.12
- PyTorch: 2.11.0+cu130
- CUDA: 13.0
- GPU: NVIDIA L4
- レンダリング: egl

**⚠️ 注意**: 本番環境のPyTorch 2.11.0はLeRobot v0.4.4の要件（`<2.11.0`）を満たさない可能性があります。動作確認では2.10.0を使用していますが、本番で問題が発生する場合はPyTorchのダウングレードが必要かもしれません。

## 発見した問題

### Python バージョン互換性の問題

1. **LeRobot v0.6.0はPython 3.12+が必要**
   - Python 3.12の新しい構文を使用（generic type parameter syntax: `class MyClass[T]:`）
   - `type`キーワード（型エイリアス）
   - 環境はPython 3.10.12 → 動作不可

2. **LeRobot v0.5.0+でPython 3.12要件が導入された**
   - v0.5.0リリース: 2025年3月9日
   - Breaking change: Python 3.12+ required

3. **LeRobot v0.4.4が最後のPython 3.10対応版**
   - v0.4.4リリース: 2025年2月28日
   - `requires-python = ">=3.10"`
   - **SmolVLAもサポートしている**（重要な発見）
   - ドキュメント: https://huggingface.co/docs/lerobot/v0.4.4/en/smolvla

## 解決方法

### 1. LeRobot v0.4.4のインストール

```bash
# コンテナ内で実行
cd /workspace/submission_template
rm -rf lerobot
git clone --branch v0.4.4 --depth 1 https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e . --no-deps
```

### 2. SmolVLA依存関係のインストール

```bash
# PyTorch (LeRobot v0.4.4互換バージョン)
pip install 'torch>=2.2.1,<2.11.0' 'torchvision>=0.21.0,<0.26.0' --index-url https://download.pytorch.org/whl/cpu

# 推論に必要な最小限のパッケージ
pip install 'transformers>=4.57.1,<5.0.0'
pip install 'num2words>=0.5.14,<0.6.0'
pip install 'accelerate>=1.7.0,<2.0.0'
pip install 'safetensors>=0.4.3,<1.0.0'

# LeRobot基本依存関係
pip install 'datasets>=4.0.0,<5.0.0'
pip install 'diffusers>=0.27.2,<0.36.0'
pip install 'jsonlines>=4.0.0,<5.0.0'
pip install 'deepdiff>=7.0.1,<9.0.0'
pip install 'pyserial>=3.5,<4.0'
pip install 'av>=15.0.0,<16.0.0'
```

**重要**: torchvisionをバージョン指定なしでインストールすると、最新版（0.28.0など）がインストールされtorchとの互換性がなくなるため、必ずバージョン範囲を指定してください。

### 3. config.jsonの修正

v0.6.0で保存されたconfig.jsonには、v0.4.4に存在しないフィールドが含まれているため削除が必要：

```bash
# コンテナ内で実行
sed -i '/\"pretrained_revision\":/d' /workspace/submission_template/model_weights/config.json
```

または手動で`"pretrained_revision": null,`の行を削除

### 4. policy_server.pyのコード修正

#### 修正1: importパスの変更

```python
# v0.6.0（動作しない）
from lerobot.configs import PreTrainedConfig

# v0.4.4（正しい）
from lerobot.configs.policies import PreTrainedConfig
```

理由: v0.4.4では`lerobot/configs/__init__.py`が存在せず、直接`policies.py`からインポートする必要がある

#### 修正2: deviceの文字列化

```python
# configを読み込んだ後、deviceを文字列に変更
config.device = "cpu"  # v0.4.4のsafetensors互換性のため
```

理由: v0.4.4の`safetensors`は`torch.device`オブジェクトではなく文字列が必要

#### 修正3: model_pathのデフォルト値変更

```python
# for_parc2026コンテナ用（/workspaceから起動）
def __init__(self, model_path: str = "submission/model_weights"):
```

理由: コンテナ内で`cd /workspace && python submission/policy_server.py`として起動するため、相対パスを調整

### 5. デバッグプリントの追加

モデル読み込み確認のため、以下のプリント文を追加：

```python
print(f"[MyPolicy] Initializing policy from: {model_path}")
print(f"[MyPolicy] Found config.json and model.safetensors")
print(f"[MyPolicy] Using device: {self.device}")
print(f"[MyPolicy] Config loaded:")
print(f"  - use_peft: {config.use_peft}")
print(f"  - load_vlm_weights: {config.load_vlm_weights}")
print(f"  - vlm_model_name: {config.vlm_model_name}")
print(f"[MyPolicy] Loading SmolVLAPolicy...")
print(f"[MyPolicy] Policy loaded successfully and set to eval mode")
```

## 成功時の出力例

```
============================================================
Initializing Policy Server
============================================================
[MyPolicy] Initializing policy from: model_weights
[MyPolicy] Found config.json and model.safetensors
[MyPolicy] Using device: cpu
[MyPolicy] Config loaded:
  - use_peft: False
  - load_vlm_weights: False
  - vlm_model_name: HuggingFaceTB/SmolVLM2-500M-Video-Instruct
  - chunk_size: 50
  - n_action_steps: 50
[MyPolicy] Loading SmolVLAPolicy...
Reducing the number of VLM layers to 16 ...
Loading weights from local directory
[MyPolicy] Policy loaded successfully and set to eval mode
✓ Policy initialized successfully
Policy server starting on 0.0.0.0:8000
============================================================
```

## モデルファイル形式の重要な発見

### LeRobot固有形式

`model_weights/`ディレクトリに含まれるファイル：
- `config.json`: SmolVLA設定（LeRobot形式）
- `model.safetensors`: ファインチューニング済み重み
- `policy_preprocessor.json`: 前処理パイプライン設定
- `policy_postprocessor.json`: 後処理設定

**重要**: これらはLeRobot固有の形式で、Hugging Face `transformers`では直接読み込めない

### transformersで直接読み込めない理由

1. LeRobotのポリシー設定形式が独自
2. 前処理・後処理パイプラインがLeRobot専用
3. モデル構造がLeRobotのポリシーフレームワークに依存

## トラブルシューティング

### エラー: `ModuleNotFoundError: No module named 'serial'`

**原因**: `pyserial`がインストールされていない  
**解決**: `pip install 'pyserial>=3.5,<4.0'`

### エラー: `ModuleNotFoundError: No module named 'av'`

**原因**: PyAVがインストールされていない  
**解決**: `pip install 'av>=15.0.0,<16.0.0'`

### エラー: `DecodingError: The fields 'pretrained_revision' are not valid`

**原因**: config.jsonにv0.4.4に存在しないフィールドが含まれている  
**解決**: `sed -i '/\"pretrained_revision\":/d' config.json`

### エラー: `SafetensorError: device device(type='cpu') is invalid`

**原因**: v0.4.4のsafetensorsが`torch.device`オブジェクトを受け付けない  
**解決**: `config.device = "cpu"` で文字列に変換

### エラー: `ModuleNotFoundError: No module named 'torchvision'`

**原因**: torchvisionがインストールされていない、または互換性のないバージョンがインストールされている  
**解決**: 
```bash
# 間違ったバージョンがある場合は削除
pip uninstall -y torch torchvision
# 正しいバージョンをインストール
pip install 'torch>=2.2.1,<2.11.0' 'torchvision>=0.21.0,<0.26.0' --index-url https://download.pytorch.org/whl/cpu
```

### エラー: `FileNotFoundError: Model path does not exist: model_weights`

**原因**: 起動ディレクトリとmodel_pathの相対パスが合っていない  
**解決**: `/workspace`から起動する場合は`model_path="submission/model_weights"`に変更

## 重要な教訓

1. **Python バージョンの互換性は重要**
   - ライブラリのバージョンアップでPython要件が変わることがある
   - 本番環境のPythonバージョンを事前に確認すべき

2. **ドキュメントの確認**
   - 各バージョンのドキュメントを確認することで、機能の有無がわかる
   - v0.4.4にSmolVLAが存在することを発見できた

3. **モデル形式の互換性**
   - LeRobotで学習したモデルはLeRobotで推論する必要がある
   - transformersと互換性がないため、直接読み込めない

4. **デバッグプリントの重要性**
   - モデル読み込みの各ステップで状態を出力することで、問題箇所を特定しやすい
   - 本番環境でも動作確認に役立つ

## 次のステップ

1. ✅ モデルが正しく読み込まれることを確認済み
2. ✅ サーバーが起動することを確認済み
3. ⏭️ 実際の評価を実行して、0%問題が解決したか確認
4. ⏭️ 必要に応じて本番環境（GPU L4, CUDA 13.0）での動作確認

## 参考リンク

- LeRobot v0.4.4 ドキュメント: https://huggingface.co/docs/lerobot/v0.4.4
- SmolVLA v0.4.4 ドキュメント: https://huggingface.co/docs/lerobot/v0.4.4/en/smolvla
- LeRobot GitHub: https://github.com/huggingface/lerobot
- Release v0.4.4: https://github.com/huggingface/lerobot/releases/tag/v0.4.4
- Release v0.5.0 (Python 3.12+): https://github.com/huggingface/lerobot/releases/tag/v0.5.0

---

**作成日**: 2026-08-03  
**対象プロジェクト**: PARC2026 LeRobot SmolVLA  
**解決した問題**: Python 3.10環境でLeRobot v0.6.0のSmolVLAモデルを読み込む方法
