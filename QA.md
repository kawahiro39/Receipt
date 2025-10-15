# QA / 運用マニュアル

Vanlee Receipt AI を Bubble から利用する際の手順と、デプロイ後の動作確認方法をまとめています。Cloud Run で FastAPI アプリを稼働させ、Bubble 側から API Connector などでエンドポイントを呼び出すことを前提としています。

## 1. 事前準備

1. Cloud Run 側で以下の環境変数または Secret を設定します。
   - `BUBBLE_API_BASE` (例: `https://system.vanlee.co.jp/version-test/api/1.1`)
   - `BUBBLE_API_KEY`
2. Bubble 側で API Connector を設定し、上記キーを Bearer トークンとして送信できるようにします。
3. FastAPI アプリの依存関係は `requirements.txt` に記載されています。ローカル検証時は `pip install -r requirements.txt` を実行してください。
4. Docker イメージには Tesseract OCR（英語・日本語データ同梱）がバンドルされるようになりました。Cloud Run 等の本番環境では追加のネイティブ依存関係を用意する必要はありません。ローカルマシン上で直接アプリを実行する場合のみ、以下の手順を参考に Tesseract をインストールしてください。

   ```bash
   sudo apt-get update && sudo apt-get install -y \
     tesseract-ocr \
     tesseract-ocr-eng \
     tesseract-ocr-jpn
   ```

   必要に応じて `OCR_LANGUAGE` 環境変数を `"jpn"` などに設定すると、日本語 OCR モードを強制できます。

## 2. Bubble Data API 接続テスト

Bubble の Data API へアクセスできることを確認します。以下はリポジトリルートで実行する Python スクリプトの例です（`requests` がインストールされ、アウトバウンド通信が許可されている必要があります）。

```bash
BUBBLE_API_KEY="<your-api-key>" \
BUBBLE_API_BASE="https://system.vanlee.co.jp/version-test/api/1.1" \
python - <<'PY'
from app import bubble_client

response = bubble_client.bubble_search(
    "Receipt",
    limit=1,
)
print(response)
PY
```

`curl` を利用する場合は次のように実行します。

```bash
curl \
  -H "Authorization: Bearer ${BUBBLE_API_KEY}" \
  -H "Content-Type: application/json" \
  "${BUBBLE_API_BASE:-https://system.vanlee.co.jp/version-test/api/1.1}/obj/Receipt?limit=1"
```

いずれも 2xx が返り、`Receipt` データタイプの JSON が得られれば成功です。

## 3. FastAPI エンドポイントの利用方法

Bubble のワークフローから以下のエンドポイントを呼び出してください。すべて `Content-Type: application/json` の POST リクエストです。

### 3.1 `/predict`
- 目的: 領収書画像から項目抽出と勘定科目推定を行い、必要に応じて Bubble 上の Receipt を作成/更新します。
- 入力例:
  ```json
  {
    "image_url": "https://example.com/receipt.jpg"
  }
  ```
- doc_id はサーバー側で `r_<YYYYMMDD>_<6桁ランダム英数字>` 形式に自動採番されます。
- 出力例: 抽出結果 (`extracted`)、推定カテゴリ (`category`)、使用モデル (`model_version`)、Bubble へ upsert した Receipt の `_id` (`receipt_id`) を含む JSON。
- Bubble 側では API Connector のアクションを作成し、レスポンスで返る `category.pred` 等をワークフローに渡す想定です。

### 3.2 `/feedback`
- 目的: ユーザーが修正した情報を Bubble 上の Feedback として保存し、該当 Receipt のステータスを `corrected` に更新します。
- 入力例:
  ```json
  {
    "receipt_id": "<Bubble Receipt _id>",
    "doc_id": "r_20251015_0001",
    "correct": {
      "category": "事務用品費",
      "vendor": "デンキチ",
      "date": "2025-10-12",
      "total": 36990
    },
    "reason": "用紙と電池の購入"
  }
  ```
- 出力: `{ "ok": true }`
- Bubble 側では修正フォーム送信時に呼び出し、レスポンスをトースト表示などで案内する運用を想定しています。

### 3.3 `/train`
- 目的: 蓄積した Feedback を取得し、モデルを追加学習したうえで Bubble の ModelVersion に保存します。
- 入力例:
  ```json
  {
    "since": "2025-10-01T00:00:00Z",
    "min_samples": 50
  }
  ```
- 出力例:
  ```json
  {
    "ok": true,
    "model_version": "sgd-tfidf-2025-10-15T12:00",
    "metrics": {"acc": 0.89, "n": 1203}
  }
  ```
- Bubble 管理者用のワークフロー（スケジュール API Workflow 等）から定期的に実行することを推奨します。

## 4. デプロイ後の動作確認

### 4.1 Cloud Run へのデプロイ
Cloud Build / Cloud Run でデプロイする際は、リポジトリルートに配置した `Dockerfile` が使用されます。手元でビルドを検証する場合は以下を実行してください。

```bash
docker build -t receipt-ai:latest .
```

ビルドが成功すると、FastAPI アプリがポート `8080` で起動するコンテナイメージが作成されます。

### 4.2 ヘルスチェック
Cloud Run デプロイ後に次のコマンドで FastAPI の起動を確認できます。

```bash
curl -s https://<cloud-run-host>/predict -X POST -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/placeholder.jpg"}'
```

HTTP 200 または 422（バリデーションエラー）が返ればアプリは起動しています。必要に応じてログを Cloud Logging で確認してください。

## 5. トラブルシューティング

- **Dockerfile が見つからない**: Cloud Build の実行ログに `unable to evaluate symlinks in Dockerfile path` が出る場合、`Dockerfile` がリポジトリルートに存在することを確認してください。
- **Bubble API エラー**: 401/403 が返る場合は API キーと Privacy Rules を再確認します。
- **タイムアウト**: OCR や学習処理が長引く場合、Cloud Run のタイムアウト設定とログを確認し、必要に応じて処理を非同期化してください。

以上を参考に、Bubble アプリから Vanlee Receipt AI を安定的に利用・運用してください。
