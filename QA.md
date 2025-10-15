# QA / 運用マニュアル

Cloud Run 上の領収書解析サービスを Bubble から安全に利用するための手順をまとめます。本書は `/ingest` → `/feedback` → `/train` のフローを前提としています。

## 1. 事前準備
1. Cloud Run のサービスに以下の環境変数を設定します。
   - `BUBBLE_API_BASE`（例: `https://example.com/version-test/api/1.1`）
   - `BUBBLE_API_KEY`
   - `OCR_ENGINE=local`
   - `OCR_LANGUAGE=jpn+eng`
   - `ADMIN_TOKEN=<管理者トークン>`
   - 必要に応じて `BUBBLE_SIGNATURE_SECRET` を設定し、Bubble 側の署名ヘッダと同期させてください。
2. Bubble の Privacy Rules で Receipt / Feedback / ModelVersion すべての Data API Create/Modify を許可します。開発 (version-test) と本番 (live) で URL・トークンを分離してください。
3. ローカルで検証する場合は `pip install -r requirements.txt` の後、`uvicorn app.main:app --reload` で起動します。Tesseract (jpn/eng) がインストール済みであることが前提です。

## 2. Bubble Data API 接続テスト
`/bubble-write-test` を利用するか、以下のコマンドで直接確認できます。

```bash
curl \
  -X POST \
  -H "Authorization: Bearer ${BUBBLE_API_KEY}" \
  -H "Content-Type: application/json" \
  "${BUBBLE_API_BASE}/obj/Receipt" \
  -d '{"status": "predicted", "source": "manual-check"}'
```

HTTP 200 が返り `id` が含まれていれば Bubble 側の書き込み許可は問題ありません。

## 3. FastAPI エンドポイントの利用
### 3.1 `/ingest`
- Bubble Workflow から `multipart/form-data` でファイルを送信します。
- `Idempotency-Key` を付与すると重複送信時に同じ `doc_id` が返ります。
- 署名を利用する場合は `X-Bubble-Signature: hmac=<base64>` を付与し、サーバ側に `BUBBLE_SIGNATURE_SECRET` を設定してください。
- 正常レスポンス例:
  ```json
  {
    "doc_id": "161234567890x",
    "extracted": {
      "date": "2024-10-01",
      "amount": 2800.0,
      "merchant": "カフェABC",
      "category": "会議費"
    },
    "candidates": {
      "amount": [{"value": 2800.0, "raw_text": "2,800", "confidence": 0.9}],
      "category": [{"label": "会議費", "confidence": 0.74}]
    }
  }
  ```
- Bubble ではレスポンスの `doc_id` を Receipt の id として保持してください。

### 3.2 `/feedback`
- 修正フォームなどから JSON を送信します。
- リクエスト例:
  ```json
  {
    "doc_id": "161234567890x",
    "patch": {
      "amount": {"value": 3000, "bbox": [100, 200, 260, 240]},
      "category": "会議費"
    }
  }
  ```
- サーバは Receipt を更新し、`Feedback` レコードをフィールドごとに生成します。
- レスポンス例: `{ "status": "ok", "feedback_ids": ["1689"], "updated_doc_id": "161234567890x" }`

### 3.3 `/train`
- 管理者ワークフロー (Scheduler or Manual) から `Authorization: Bearer <ADMIN_TOKEN>` を付けて呼び出します。
- 例:
  ```bash
  curl \
    -X POST \
    -H "Authorization: Bearer ${ADMIN_TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{"task": "category", "min_samples": 50}' \
    https://<service-url>/train
  ```
- レスポンス例:
  ```json
  {
    "trained": {"category": 120},
    "metrics": {"category": {"n": 120, "classes": ["会議費", "交際費"]}},
    "model_version_ids": {"category": "1700123456789"}
  }
  ```
- `trained` 件数が 0 の場合、`metrics.reason` にスキップ理由が入ります。

## 4. 運用チェックリスト
1. `/bubble-write-test` を定期的に実行し、Bubble 側の認証が切れていないか確認する。
2. Cloud Logging で `ocr_confidence` や処理時間をモニタリングする。
3. `/train` 実行後に `Feedback.processed_at` がセットされているか Bubble 側で確認する。
4. `ModelVersion` の `is_latest` が最新の学習で切り替わったことを確認する。

## 5. トラブルシューティング
- **400 unsupported_mime**: `file` の Content-Type を確認。PDF/JPEG/PNG/TIFF のみ受け付けます。
- **401 invalid_signature**: 署名ヘッダと `BUBBLE_SIGNATURE_SECRET` が一致しているか確認。
- **424 bubble_write_failed**: Bubble 側の Privacy Rules や API キー、通信エラーを確認。
- **422 ocr_decode_failed**: 画像が破損している可能性があります。OCR に入力できる形式へ再変換してください。

以上で Bubble と Cloud Run 間の受領書処理フローを構築できます。
