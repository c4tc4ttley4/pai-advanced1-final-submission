# XLeRobot 卓上片付け: ACTの学習と未達の分析

固定台車・機体左腕（モデル_R）でブロック1個をトレーへ収納するシミュレーション課題です。成功はトレー内で物体を離して2秒以上安定すること。expert scriptで成功教師5試行9531フレームを収集しましたが、ACT20k/n10・n1は既知5配置とも物体接触/把持/40mm/収納0/5です。独立評価・実機は未実施です。

## 環境と構成

既存Docker image pai-advanced1-final:step01（digest sha256:b53107c2155ba2539d738e27cdbb4430d439dba116dc74f52c95288cd07b8c46）、Python3.12.12、uv0.9.5、LeRobot0.4.2、torch2.7.1+cu128、MuJoCo3.3.7。依存定義はpyproject.toml/uv.lock、構築定義はDockerfile/compose.yaml。既存image/依存はこの制作中に変更していません。

- src/: 環境、接触expert、LeRobot v3記録/再生、ACTと安全判定。
- scripts/: 実験入口と保存済み結果の解析。ラッパーは取得済み素材を前提とするものもあります。
- configs/, scenes/: 関節順・物理・カメラと派生出典。
- source_manifest.json: 収録元/元SHA/提出copy SHA/変更箇所。
- asset_manifest.csv: 別資産の元相対path/容量/確認済みdigest。空欄はmetadata-only。
- reports/*.sha256: 過去時点の識別記録。現在の全ファイルhashではありません。

学習入力はRGB640x480のCHW float32[0,1]と実測6軸qposのみ。state/actionはrad絶対位置、順序はRotation_R, Pitch_R, Elbow_R, Wrist_Pitch_R, Wrist_Roll_R, Jaw_R。操作50Hz/物理500Hz。Jawは正方向が開口です。ソース内の初期のright-arm表記は旧モデル呼称で、現在の機体左腕/_R対応を変えていません。

## 実装に存在するコマンドと必要な前提

以下は実在する入口の記述です。本STEPで実モデル・物理を実行した再現確認はしていません。旧runへ上書きせず、新しい実験の承認・資源確認・旧入力ro/新出力rwのcreate→inspect検証が必要です。.envは非同梱で、.env.exampleからローカル値を用意します。Docker構築/外部取得をこの制作中に実行していません。

既存の専用Docker/uv内での入口（各mode/引数はソースのargparseと照合済み）:

    uv run --no-sync python scripts/step04.py record --output artifacts/step04/NEW_RUN --trial 0
    uv run --no-sync python scripts/step04.py validate --root data/step04/DATA_RUN --output artifacts/step04/VERIFY_RUN
    uv run --no-sync python scripts/step04.py replay --root data/step04/DATA_RUN --output artifacts/step04/VERIFY_RUN --trial 0
    uv run --no-sync python scripts/step05A.py audit --run NEW_ACT_RUN
    uv run --no-sync python scripts/step05A.py train --run NEW_ACT_RUN
    uv run --no-sync python scripts/step05A.py continue --run NEW_ACT_RUN
    uv run --no-sync python scripts/step05A.py reload --run NEW_ACT_RUN --checkpoint continued
    uv run --no-sync python scripts/step05A.py rollout --run NEW_ACT_RUN --checkpoint continued --trial 0

上のNEW_*やDATA_RUNは説明用プレースホルダーで、コピペで既存成果を上書きする指示ではありません。STEP4 recordはContactExpertから教師を作ります。configのdataset_rootは保存教師への既定参照です。初期STEP5Aの学習/continueは600更新までの設計で、上記だけで20000更新を再現するわけではありません。

5000/10000/20000系列はscripts/step05A_train5000.py、step05A_train20000_r02.pyに実装し、同じ5教師・batch4/fp32/AdamW lr1e-5・chunk50を維持しました。20k系列は5000のoptimizer/RNG/samplerから追加15000更新したものです。旧run専用spec/台帳とbundleが必要で、CPU/GPU制限・保全gateを含むラッパーを確認します。

保存20k推論/評価と教師診断のhistorical入口:

    node scripts/step05A_train20000_r02_run.cjs reload
    node scripts/step05A_train20000_r02_run.cjs rollout 20000 0
    node scripts/step05A_naction20000_r02_run.cjs gate
    node scripts/step05A_naction20000_r02_run.cjs rollout 20000 0
    node scripts/step05A_phase_visibility_r03_run.cjs preflight

これらは凍結された旧run IDを保持し、旧spec/保存bundle/代表入力と実行台帳が必要です。既存のattemptを再実行しない設計のため、既に実行したディレクトリでは拒否されます。提出copyでは4 CJSのROOTだけを__dirname基準に変更し、元実装/安全gate/予算/固定IDは変更していません。新runへの一般化は別設計が必要です。

## 別資産と再現性の範囲

教師: data/step04/dataset_20261029_01/、5episode・9531frame。チェックポイント: models/step05A_train_compare/train5000_20261031_01/step5000/、models/step05A_train_compare/train20000_r02_20261101_01/{step10000,step20000}/。各bundleは重み/専用統計/公式pre-post/代表入力/optimizer/RNG等とbundle.sha256.jsonを必要とします。元データ/重み/実験spec/動画はこのstagingに同梱せず、asset_manifestで識別します。架空のダウンロードURLは設けません。

XLeRobot外部モデルの取得処理はscripts/fetch_xlerobot.pyに固定commit/サイズ/元Git blob検証として存在します。この制作中は実行していません。取得と権利確認を済ませたthird_party/XLeRobot/simulation/mujoco/assets/がsceneから必要です。

構文/設定/ローカル参照検査のみを今回実施し、学習/forward/物理0です。コードだけで完全再現できると主張しません。学習内誤差や警告0を未知配置の成功率/実機安全へ読み替えず、旧5000/n1の中央0.478秒/侵入1.0537mm停止と残り4未実施も保持します。

配布条件はLICENSE_STATUS.mdを参照してください。public repo/YouTube/omnicampusは未作成・未投稿・未提出です。

## 提出前の最終静的検査

最終仕上げ時にも71ファイルを再点検しました。Python29件、SH6件、CJS4件、ローカルimport43参照を静的に確認しています。依存/データ/メッシュ/重み/旧run specが不足した状態の実行可能性は保証せず、学習・model forward・物理stepは実行していません。XLeRobotの元Apache-2.0 LICENSEを維持し、自作コードへの新ライセンス付与はしていません。外部公開は未実施で、公開先と配布条件は本人判断待ちです。
