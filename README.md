# Receipt Processing Service (Bubble Data API Integration)

このリポジトリは、Bubble Data API をデータストアとして利用する領収書解析サービスの実装です。Cloud Run 上で FastAPI アプリケーションを稼働させ、OCR から抽出した情報を Receipt / Feedback / ModelVersion オブジェクトとして Bubble に保存します。

## 1. アーキテクチャ概要
- **FastAPI (`app/main.py`)**: `/ingest`、`/feedback`、`/train`、`/bubble-write-test` の各エンドポイントを提供。ファイル受領、バリデーション、Bubble への書き込みを一元管理します。
- **OCR (`app/ocr_extract.py`)**: Tesseract を前提としたローカル OCR パイプライン。Pillow で画像をロードし、`--oem 3 --psm 6` で実行します。失敗時はテスト向けのフェイルバック実装に切り替わります。
- **項目抽出 (`app/field_extractors/`)**: 金額・日付・店名を正規表現やヒューリスティックで抽出し、候補の信頼度を算出します。
- **カテゴリ分類 (`app/classifier.py`)**: TF-IDF + SGDClassifier による incremental learning。`predict_category` と `partial_train` を提供します。
- **モデル保存 (`app/model_store.py`)**: `pickle` 化したモデルを Base64 チャンクとして Bubble `ModelVersion` に保存。`is_latest` の切り替えも同時に行います。
- **Bubble Data API クライアント (`app/bubble_client.py`)**: 認証ヘッダやタイムアウトを共通化した薄い HTTP クライアント。Receipt/Feedback/ModelVersion を対象とした CRUD をラップします。
- **セキュリティ (`app/security.py`)**: HMAC 署名検証、管理者トークン検証、Idempotency-Key の TTL キャッシュを実装します。

## 2. 環境変数
| 変数名 | 必須 | 内容 |
| --- | --- | --- |
| `BUBBLE_API_BASE` | ✅ | `https://` から始まる Bubble Data API ベース URL（`/api/1.1` まで。末尾に `/obj` を含んでいても可）。|
| `BUBBLE_API_KEY` | ✅ | Bubble Data API の Bearer トークン。|
| `OCR_ENGINE` | ✅ | 現状は `local` 固定。その他の値は起動時にエラー。|
| `OCR_LANGUAGE` | ✅ | Tesseract の言語コード。推奨 `jpn+eng`。|
| `ADMIN_TOKEN` | ✅ | `/train` の Bearer 認証で利用。|
| `BUBBLE_SIGNATURE_SECRET` | 任意 | `X-Bubble-Signature` 検証用シークレット。ヘッダがあるのにシークレット未設定の場合は 401。|
| `TZ` | 任意 | タイムゾーン（推奨 `Asia/Tokyo`）。|

未設定または不正な値の場合、起動時に例外を投げてサービスが立ち上がりません。

`.env` ファイルが存在する場合は自動で読み込まれるため、環境変数を OS 側にエクスポートできない環境でも `.env` に上記キーを定義すれば 500 エラーを防げます。

## 3. エンドポイント仕様
### 3.1 `POST /ingest`
- **Body**: `multipart/form-data`
  - `file` (必須): PDF / 画像ファイル。
  - `image_url` (任意): Bubble に保存したい URL。
  - `source` (任意): 呼び出し元識別子。デフォルト `bubble-ui`。
- **Headers**:
  - `Idempotency-Key` (任意): 同一キーでリプレイした場合は同じ `doc_id` を返却。
  - `X-Bubble-Signature` (任意): HMAC 署名。`BUBBLE_SIGNATURE_SECRET` で検証。
- **処理フロー**:
  1. ファイルのサイズ/MIME を検証（15MB まで、PDF/JPEG/PNG/TIFF）。
  2. ローカル Tesseract で OCR → `field_extractors` で `date` / `amount` / `merchant` 候補を生成。
  3. `classifier.predict_category` で `category` を推定し、候補と確率を組み立て。
  4. Bubble `Receipt` に `status=predicted` で create。`raw_text`、`ocr_confidence`、`candidates_json` などを保存。
  5. レスポンス `{ doc_id, extracted, candidates }` を返却。
- **エラーコード**: 400（バリデーション）、401（署名失敗）、422（OCR デコード失敗）、424（Bubble 書き込み失敗）、500（想定外）。

### 3.2 `POST /feedback`
- **Body**: JSON
  - `doc_id` (必須): Bubble `Receipt` の id。
  - `patch` (必須): 修正内容の辞書。値がオブジェクト `{ "value": ..., "bbox": ... }` の場合は `Feedback.bbox_json` に保存。
  - `field_scope` (任意): フィードバック生成対象を限定するフィールド名配列。
- **処理フロー**:
  1. `Receipt` を patch し `status=corrected` に変更。
  2. `patch` の各キーに対して `Feedback` を create（`processed_at` は null）。
  3. `{ "status": "ok", "feedback_ids": [...], "updated_doc_id": ... }` を返却。

### 3.3 `POST /train`
- **Authorization**: `Bearer <ADMIN_TOKEN>` 必須。
- **Body**: JSON
  - `task`: `all` / `category` / `amount` / `date` / `merchant`（デフォルト `all`）。
  - `min_samples`: 学習に必要な件数（デフォルト 50）。
  - `test_ratio`: 将来の検証用パラメータ（現状未使用、0.0〜0.5）。
- **処理フロー**:
  1. `processed_at` が空の `Feedback` を取得（`task` に応じて `field` をフィルタ）。
  2. `category` の場合は対応する `Receipt` を読み込み `partial_train` で増分学習。その他タスクは値を集計してプレースホルダーの `ModelVersion` を保存。
  3. 保存したモデルの `id` を返し、消化した `Feedback` に `processed_at` と `model_version_trained_on` を付与。
  4. `{ trained, metrics, model_version_ids }` を返却。

### 3.4 `POST /bubble-write-test`
- 最小ペイロードで `Receipt` を create し、Bubble 書き込み経路を検証するヘルスチェック用エンドポイントです。

## 4. ディレクトリ構成
- `app/main.py` – FastAPI ルータ、バリデーション、Bubble API 連携。
- `app/ocr_extract.py` – OCR 実行とフォールバック。
- `app/field_extractors/` – 金額・日付・店名の候補抽出。
- `app/classifier.py` – 勘定科目分類器と学習ロジック。
- `app/model_store.py` – `ModelVersion` への保存/読込。
- `app/bubble_client.py` – Bubble Data API クライアント。
- `app/security.py` – 署名検証、管理者トークン、Idempotency。
- `tests/` – pytest によるユニット/統合テスト。
- `Dockerfile` – Tesseract (jpn/eng + tessdata_best) とランタイムをインストール。

## 5. ローカル実行 & テスト
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export BUBBLE_API_BASE="https://example.com/api/1.1"
export BUBBLE_API_KEY="dummy"
export OCR_ENGINE="local"
export OCR_LANGUAGE="jpn+eng"
export ADMIN_TOKEN="local-admin"
uvicorn app.main:app --reload
```

テストは `pytest` を実行してください。外部サービスはテスト内でモックできます。

## 6. デプロイと運用
1. Docker イメージをビルドし Artifact Registry へ push。
2. Cloud Run へデプロイ（必要な環境変数を設定）。
3. 初回起動後は以下のシナリオで整合確認を行います。
   1. `/bubble-write-test` で Bubble 書き込みが成功すること。
   2. `/ingest` へサンプルファイルを送信し、Bubble `Receipt` にレコードが作成されること。
   3. 同じ `doc_id` で `/feedback` を実行し、`status=corrected` と `Feedback` が生成されること。
   4. `/train` を実行し、`ModelVersion` が保存され `is_latest` が切り替わること。
4. Cloud Scheduler などから `/train` を定期呼び出しし、`Authorization: Bearer <ADMIN_TOKEN>` を付与してください。

## 7. プライバシー / Bubble 設定メモ
- Bubble 側で Receipt / Feedback / ModelVersion の Data API を有効化し、API トークンで Create/Modify を許可する Privacy Rules を設定してください。
- 開発環境（`version-test`）と本番環境（`live`）で URL とトークンを分離します。

## 8. ログと監視の推奨指標
- OCR 平均信頼度 (`ocr_confidence`)
- 金額抽出 accuracy@1
- カテゴリ F1 / accuracy
- Bubble 書き込み失敗率
- 全体の処理時間

## 9. ライセンス
社内利用を想定したテンプレートです。用途に合わせて適宜修正してください。
