#!/bin/bash
# LeRobot v0.4.4環境セットアップスクリプト
# Python 3.10.12環境でSmolVLAを使用するための設定

set -e  # エラーが発生したら即座に終了

# 使い方の表示
usage() {
    echo "使い方: $0 [--cpu|--gpu]"
    echo "  --cpu : CPU版PyTorchをインストール（デフォルト）"
    echo "  --gpu : GPU版PyTorchをインストール（CUDA対応）"
    exit 1
}

# デフォルトはCPU
DEVICE="cpu"

# 引数の解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu)
            DEVICE="cpu"
            shift
            ;;
        --gpu)
            DEVICE="gpu"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "不明なオプション: $1"
            usage
            ;;
    esac
done

echo "============================================================"
echo "LeRobot v0.4.4環境セットアップ開始"
echo "デバイス: $DEVICE"
echo "============================================================"

# カレントディレクトリの確認
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "スクリプトディレクトリ: $SCRIPT_DIR"
cd "$SCRIPT_DIR"

# 1. LeRobot v0.4.4のクローン
echo ""
echo "[1/5] LeRobot v0.4.4をクローン中..."
if [ -d "lerobot" ]; then
    echo "  既存のlerobot/ディレクトリを削除中..."
    rm -rf lerobot
fi
git clone --branch v0.4.4 --depth 1 https://github.com/huggingface/lerobot.git
echo "  ✓ LeRobot v0.4.4のクローン完了"

# 2. LeRobot v0.4.4のインストール
echo ""
echo "[2/5] LeRobot v0.4.4をインストール中..."
cd lerobot
pip install -e . --no-deps
cd ..
echo "  ✓ LeRobot v0.4.4のインストール完了"

# 3. PyTorchのインストール（LeRobot v0.4.4互換バージョン）
echo ""
echo "[3/5] PyTorch v0.4.4互換バージョンをインストール中..."
# 既存のtorch/torchvisionを削除してからインストール
pip uninstall -y torch torchvision 2>/dev/null || true

if [ "$DEVICE" = "gpu" ]; then
    echo "  GPU版PyTorchをインストール（CUDA対応）..."
    pip install 'torch>=2.2.1,<2.11.0' 'torchvision>=0.21.0,<0.26.0'
else
    echo "  CPU版PyTorchをインストール..."
    pip install 'torch>=2.2.1,<2.11.0' 'torchvision>=0.21.0,<0.26.0' --index-url https://download.pytorch.org/whl/cpu
fi
echo "  ✓ PyTorchのインストール完了"

# 4. SmolVLA推論に必要な依存関係のインストール
echo ""
echo "[4/5] SmolVLA推論用依存関係をインストール中..."

# transformers関連
pip install 'transformers>=4.57.1,<5.0.0' \
            'num2words>=0.5.14,<0.6.0' \
            'accelerate>=1.7.0,<2.0.0' \
            'safetensors>=0.4.3,<1.0.0'

# データ処理
pip install 'datasets>=4.0.0,<5.0.0' \
            'diffusers>=0.27.2,<0.36.0' \
            'jsonlines>=4.0.0,<5.0.0' \
            'deepdiff>=7.0.1,<9.0.0'

# システム依存
pip install 'pyserial>=3.5,<4.0' \
            'av>=15.0.0,<16.0.0'

# その他
pip install 'draccus==0.10.0' \
            'gymnasium>=1.1.1,<2.0.0'

echo "  ✓ 依存関係のインストール完了"

# 5. config.jsonの修正（pretrained_revisionフィールドの削除）
echo ""
echo "[5/5] config.jsonを修正中..."
if [ -f "model_weights/config.json" ]; then
    # pretrained_revisionフィールドを削除
    sed -i.bak '/\"pretrained_revision\":/d' model_weights/config.json
    echo "  ✓ config.jsonからpretrained_revisionを削除"
    
    # バックアップファイルが作成された場合は削除
    rm -f model_weights/config.json.bak
else
    echo "  ⚠ model_weights/config.jsonが見つかりません（スキップ）"
fi

# セットアップ完了
echo ""
echo "============================================================"
echo "LeRobot v0.4.4環境セットアップ完了！"
echo "============================================================"
echo ""
echo "インストールされたバージョン:"
python -c "import lerobot; print(f'  LeRobot: {lerobot.__version__}')" 2>/dev/null || echo "  LeRobot: 0.4.4"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'  torchvision: {torchvision.__version__}')"
python -c "import transformers; print(f'  transformers: {transformers.__version__}')"
echo ""
echo "GPU情報:"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  Device count: {torch.cuda.device_count()}') if torch.cuda.is_available() else None; print(f'  Current device: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else None"
echo ""
echo "サーバー起動コマンド:"
echo "  cd /workspace && python submission/policy_server.py --port 8000"
echo ""
