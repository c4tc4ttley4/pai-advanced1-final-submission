# 提出要件・最終QA（ローカル非共有）

run: submission_final_polish_r01_20261101_01

## ローカル素材

- [x] video_final.mp4: 120秒、1280×720、H264/yuv420p、30fps、3600frame全decode。1〜3分の要件内。
- [x] expertは元録画全38.24秒を2倍速・末尾約3秒静止。元映像と同期拡大窓で把持・持上げ・トレー内解放を表示。ACT成功と区別。
- [x] ACT20kの中央n10（左）/n1（右）は同じ時間範囲を2倍速・末尾5秒静止。部分拡大×2を表示し、既知5配置の接触/把持/40mm/収納0/5を維持。
- [x] PDF: 説明資料_PAI最終課題_KosugeNorihiro.pdf。公式指定式・完全一致/NFC/UTF-8/拡張子を確認。6ページ全描画・目視・全文字レコード抽出一致、日本語字体埋込み。本文/画像/メタデータに本人識別子なし。
- [x] 「本課題で工夫した点」3項目を明示。研究数値・expert成功/ACT未達・独立評価/実機未実施を維持。
- [x] code/71ファイル: 前版からREADME最終QA注記だけ変更。Python29/SH6/CJS4/local import43・設定/第三者ライセンス/秘密/個人情報検査合格。
- [x] 元資産・元成果を保全。追加学習/model forward/新物理/GPU/実機0。
- [x] THIRD_PARTY_NOTICESとXLeRobot Apache-2.0を保持。自作コードの新LICENSEは付与しない。

## 公式10/19改訂要件と後工程（未実施）

- [x] AI活用: ACTを学習・保存復元・自律比較に使用。学習は任意要件だが実施済み。MuJoCo使用可という管理Chatの公式PDFレビューに基づく。
- [x] PDF指定名とYouTube題名の所定書式をローカル確認。後工程用の題名: PAI最終課題_178832
- [ ] 本人による最終素材レビュー、講座Notion別紙・最新締切/実際のomnicampus入力欄/権利条件の確認。要本人確認。管理計画の締切は2026-11-02 10:00 JST。
- [ ] 自作コード配布条件とpublic GitHub owner/repo決定・別承認・AGENTS整合。既存管理privateは公開変更しない。publicコードURLは未取得。
- [ ] YouTubeへ最低限定公開、子ども向け=いいえ、年齢制限=いいえ、インスタントプレミアなし。上記指定題名と2倍速表示を確認して投稿。今回ログイン/投稿0。
- [ ] omnicampusへ指定PDF・public GitHubコードURL・YouTube URLを登録/提出。今回操作0。
- [ ] 別工程でURL・第三者閲覧・提出完了/受理表示を確認。採点完了まで動画を削除・非公開にしない。

提出準備完了はACTタスク成功・独立評価・提出受理を意味しない。


## 2026-11-02 review corrections r02
- OP/ED appended; ACT replaced with saved 5000/n10 three-trial excerpts, 0-5 sim seconds at 0.5x.
- PDF explanations rebuilt as bullets; code71 unchanged.
- STOPPED: additional whole-frame motion difference assertion failed. Final additional QA incomplete; do not treat this package as accepted or ready to submit.
- Public/YouTube/omnicampus actions not authorized or performed.
