# Bubble 運用マニュアル

Bubble から領収書解析サービスを安定運用するための具体的な手順とチェックポイントをまとめました。Bubble アプリ内での連携を構築するときは、下記の設定とワークフローを順に確認してください。

## 1. 接続設定
1. **API Connector プラグイン**
   - API 名: `Receipt API`
   - ベース URL: `https://system.vanlee.co.jp/version-test/api/1.1`
   - 認証ヘッダ: `Authorization: Bearer 99c107353970e7d1f2f1b36709cd3e04`
   - 利用するエンドポイント: `/obj/Receipt`、`/obj/Feedback`、`/obj/ModelVersion`
2. **FastAPI 側環境変数**
   - `BUBBLE_API_BASE=https://system.vanlee.co.jp/version-test/api/1.1`
   - `BUBBLE_API_KEY=99c107353970e7d1f2f1b36709cd3e04`
   - その他必須キー（`OCR_ENGINE=local`、`OCR_LANGUAGE=jpn+eng`、`ADMIN_TOKEN=<任意のトークン>`）も設定します。
3. **Privacy Rules**
   - Receipt / Feedback / ModelVersion データ型で Data API の Create & Modify を許可します。
   - 本番運用時は `live` バージョン用に別トークンとベース URL を準備し、Connector の Development / Live 設定で切り替えます。

## 2. 推奨ワークフロー構成
1. **領収書アップロードページ**
   - File Uploader コンポーネントを配置し、ファイル選択時に `/ingest` へ API Connector を介して `multipart/form-data` で送信。
   - レスポンスの `doc_id` を Receipt データ型のレコード ID として保存します。
2. **レビュー画面**
   - Receipt レコードを表示し、抽出された `amount`、`date`、`merchant`、`category` をフォームにバインド。
   - 修正完了時に `/feedback` エンドポイントへ JSON で送信し、修正内容を Bubble 側に蓄積します。
3. **学習トリガー**
   - Bubble Scheduler から日次/週次で `/train` を実行します。Authorization ヘッダに `Bearer <ADMIN_TOKEN>` をセットしてください。
   - 結果をログするために API Connector のレスポンスを `ModelVersion` データ型に書き込み、更新履歴を保持します。

## 3. 運用チェックリスト
- [ ] `/bubble-write-test` を 1 日 1 回実行し 200 が返るか監視する。
- [ ] API Connector のトークン有効期限を記録し、更新期限の 1 週間前にリフレッシュする。
- [ ] Bubble の Capacity Monitor で API 呼び出し数を確認し、閾値を超える前にワークフローを調整する。
- [ ] `/train` 実行後に `Feedback` レコードの `processed_at` が埋まっているか目視確認する。
- [ ] 新しい ModelVersion が作成されたら `is_latest` の整合性を確認し、必要に応じて古いモデルをアーカイブする。

## 4. トラブルシューティング
| 症状 | 確認ポイント | 対処 | 
| --- | --- | --- |
| 500 Internal Server Error | FastAPI 側の `.env` に必須キーが入っているか、`BUBBLE_API_BASE` が `https://` 始まりか | `.env` を修正しサービスを再起動 |
| 424 bubble_write_failed | Privacy Rules や API トークンの権限が不足していないか | Bubble 側の API 設定を見直し、必要なら新しいトークンを発行 |
| `/train` のレスポンスが `trained=0` | フィードバック件数が `min_samples` を満たしているか | Bubble の Feedback レコード数を確認し、しきい値を下げるかデータを追加 |
| Connector から 401 が返る | Authorization ヘッダのトークンが最新か | API Connector のヘッダを更新し、FastAPI 側の `BUBBLE_API_KEY` と揃える |

## 5. 変更管理
- Connector 設定を更新したら必ずバージョン履歴にコメントを残し、どのキーを適用したか記録します。
- FastAPI 側の環境変数は Terraform や Secret Manager で管理し、手動変更した場合は運用ノートに反映します。

上記の手順を定期的に見直すことで、Bubble アプリと領収書解析サービスの連携を安定させられます。
