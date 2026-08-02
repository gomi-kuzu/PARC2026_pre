# 提出物テンプレート

## ディレクトリ構成

```
submission.zip
├── policy_server.py     # ← MyPolicy クラスを編集する（必須）
├── requirements.txt     # ← 追加依存があれば記載（必須）
└── model_weights/       # ← チェックポイント等を配置（任意）
```

## 手順

1. `policy_server.py` の `MyPolicy` クラスにモデルのロードと推論を実装する
2. モデル重みを `model_weights/` に配置する
3. 追加ライブラリがあれば `requirements.txt` に追記する
4. zip にまとめて提出する:
   ```bash
   zip -r submission.zip policy_server.py requirements.txt model_weights/
   ```

## ローカルテスト

```bash
# サーバー起動
pip install -r requirements.txt
python policy_server.py

# 別ターミナルで評価（Track 1 の example タスクで疎通確認）
python -m pipeline --server-url http://localhost:8000 --track track1 --n-episodes 2 --max-steps 10
```

`--track track1` を指定すると Track 1 の example タスクが実行される。
配布環境のセットアップおよび評価コマンドの詳細な使用方法は、リポジトリ直下の
[README.md](../README.md) を参照すること。

## 提出前セルフチェック（推奨）

提出から採点までは 1 回あたり 15〜20 分を要する。フル採点を待たずにローカルで問題を
検出できるよう、リポジトリ直下に提出物チェックスクリプト（[validate_submission.py](../validate_submission.py)）
を用意している。**提出前に必ず実行すること。**

```bash
# いずれもリポジトリ直下で実行する
# zip を丸ごと検査（静的チェック + サーバーを起動して I/O スモークテストまで）
python validate_submission.py submission.zip

# 展開済みディレクトリでも可（このテンプレートを指定する）
python validate_submission.py submission_template/

# サーバーを起動せず静的チェックのみ（高速）
python validate_submission.py submission.zip --static

# 依存を入れてから動的チェック（クリーンな環境での再現確認）
python validate_submission.py submission.zip --install
```

検査内容: zip 健全性 / Zip Slip・zip bomb / サイズ上限（20GB）/ 必須ファイル /
`policy_server.py` の構文・エンドポイント / `requirements.txt` の構文・外部ソース
禁止（`git+`・`--index-url` 等は不可）/ サーバー起動 → `/health`→`/reset`→`/act`
において action が float32 shape (7,) であること、NaN/Inf を含まないこと、レイテンシまでを確認する。

`PASS` かつ ERROR 0 件であれば提出可能である（WARN は推奨事項である）。採点環境でも
unzip 直後に同一の検査を実施する。

## 注意事項

- `policy_server.py` のサーバー部分（FastAPI エンドポイント、シリアライゼーション）は変更しないこと
- `requirements.txt` に `git+https://…` や `--index-url` 等の外部ソース指定は使用できない（採点環境は外部通信を遮断する）
- `get_action()` は 10 秒以内に応答すること。1 リクエストでも 10 秒を超過すると
  そのトラック全体が 0 点となる（詳細はリポジトリ直下の [README.md](../README.md) の
  「タイムアウト仕様」を参照すること）
- `reset()` はエピソードごとに呼び出される。内部状態（action chunking のキャッシュ等）をクリアすること
