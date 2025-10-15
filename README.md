# Vanlee Receipt AI（Cloud Run + Bubble Data API 連携）

## プロジェクト概要
Vanlee Receipt AI は、Google Cloud Run 上で稼働する FastAPI 製の領収書解析サービスです。OCR によるテキスト抽出から勘定科目推定、学習済みモデルの保存までをすべて Bubble Data API と連携して完結させ、外部ストレージや状態を持たないサーバレス構成を実現しています。

## システム構成
- **Cloud Run**: FastAPI アプリケーションを Docker コンテナとしてデプロイ。リクエストごとにスケールし、サーバレスで運用。
- **Bubble Data API**: 伝票 (`Receipt`)、フィードバック (`Feedback`)、モデルバージョン (`ModelVersion`) の CRUD を担当。モデルトレーニング結果も Base64 化して Bubble に保存します。
- **OCR エンジン**: RapidOCR（ONNX Runtime）で日本語レシートに最適化した OCR を実行し、失敗時のみ Tesseract へフェイルバックします。

## 主なディレクトリ
- `app/`
  - `main.py`: FastAPI エンドポイント定義。`/predict`・`/feedback`・`/train` を提供。
  - `ocr_extract.py`: 画像／PDF の取得、OCR 実行、抽出項目の正規表現解析。
  - `classifier.py`: TF-IDF + SGDClassifier による勘定科目推定と増分学習。
  - `bubble_client.py`: Bubble Data API の薄いクライアント。認証ヘッダーやタイムアウトを集約。
  - `model_store.py`: Bubble 上の `ModelVersion` にモデルを Base64 で保存／取得するユーティリティ。
- `tests/`: pytest ベースのユニットテスト。
- `Dockerfile`: Cloud Run 向けビルドレシピ。Tesseract に加えて RapidOCR が必要とする `libgomp1`・`libgl1` もインストールします。

## エンドポイント
### `POST /predict`
1. `image_url` または Base64 を受け取り、`ocr_extract.extract_all` で構造化情報へ変換。
2. `model_store.load_latest_model` で最新モデルを Bubble から読み込み、`predict_category` で勘定科目推定。
3. 結果を `Receipt` データ型に upsert（`bubble_create` / `bubble_update`）。
4. 抽出内容・カテゴリ候補をレスポンスとして返却。

### `POST /feedback`
- Bubble 上の修正内容を `Feedback` データ型に保存し、該当の `Receipt` を `corrected` ステータスへ更新。

### `POST /train`
- Bubble の `Feedback` からサンプルを取得し、`partial_train` で増分学習。生成したモデルは `model_store.save_model` で `ModelVersion` に保存します。

## 技術的ポイント
- **ステートレスなモデル保存**: Cloud Run のファイルシステムは揮発性のため、`pickle` 化したモデルを Base64 エンコードして Bubble にチャンク分割し保存。ロード時は自動で連結します。
- **柔軟な検索ソート**: `bubble_search` は `sort_field` と `descending` を受け取り、最新の `ModelVersion` を確実に取得します。
- **エラー設計**: RapidOCR のエラーは自動で Tesseract にフェイルオーバーし、それでも失敗した場合は OCR 取得・デコード・Bubble 通信の各フェーズで異なる例外を投げて HTTP ステータスにマッピング。
- **Cloud Run 運用前提**: すべての機能が REST API 経由で完結するため、永続ディスクやサードパーティへの依存は不要です。

## 環境変数
| 変数名 | 用途 | 備考 |
| --- | --- | --- |
| `BUBBLE_API_BASE` | Bubble Data API のベース URL | 末尾の `/obj` の有無は自動調整 |
| `BUBBLE_API_KEY` | Bubble Data API の認証トークン | 未設定時はリクエストを拒否 |
| `OCR_ENGINE` | OCR エンジン種別（`rapidocr` / `local`） | 省略時は `rapidocr` |
| `OCR_LANGUAGE` | Tesseract 言語コード | 例: `eng`, `jpn` |

## ローカル開発 & テスト
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## デプロイ手順（Cloud Run）
1. Google Artifact Registry へ Docker イメージをビルド & プッシュ。
   ```bash
   gcloud builds submit --tag asia-northeast1-docker.pkg.dev/<PROJECT>/receipt-ai:latest
   ```
2. Cloud Run へデプロイ。
   ```bash
   gcloud run deploy receipt-ai \
     --image=asia-northeast1-docker.pkg.dev/<PROJECT>/receipt-ai:latest \
     --region=asia-northeast1 \
     --set-env-vars=BUBBLE_API_BASE=https://<your-bubble-app>/api/1.1 \
     --set-secrets=BUBBLE_API_KEY=bubble-api-key:latest
   ```
3. デプロイ完了後、Bubble の API Connector から `/predict` などを呼び出して疎通確認。

## フィードバックとモデル運用
- Bubble で修正されたデータは `/feedback` に送信され、`Feedback` データ型へ蓄積。
- 管理者はスケジュールされたワークフローや CLI から `/train` を叩き、十分なサンプルが溜まったタイミングでモデルを更新。
- 保存された `metrics_json` を Bubble 上で可視化することで、モデルバージョンごとの精度を追跡可能です。

## 参考
- `QA.md`: Bubble からの利用手順をまとめた運用マニュアル。
- `tests/`: 重要フローの回帰テストを提供。Pull Request 前に `pytest` 実行を推奨します。

