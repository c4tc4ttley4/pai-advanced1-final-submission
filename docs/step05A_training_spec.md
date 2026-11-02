# STEP 5A 学習・復元・シミュレータ接続仕様

2026-10-29。保存済み5試行だけを使う接続確認。STEP 5Bの本収集・本学習、STEP 6の独立評価には進まない。設定の正本は `configs/step05A_act.json`、初回runは `act_20261029_01`。

## 継承とデータ

既存 `pai-advanced1-final:step01`、Docker＋`uv run --no-sync`、Python3.12.12、LeRobot0.4.2、torch2.7.1+cu128、TorchCodec0.3.0、MuJoCo3.3.7。CPU2・RAM8GiB・SHM1GiB・PID256・非root・ネットワークなし。既存イメージ・依存・lock・Composeを変更しない。

`data/step04/dataset_20261029_01`、repo_id `local/step04-cleanup`、全episode0..4、9531フレームを学習集合とする。独立検証集合なし。元データはコンテナ内read-only。全ファイルSHA-256、episode境界、統計検査はrunのdata_audit.jsonへ保存する。

使用腕は機体左腕・モデル_R系統。state/actionは `Rotation_R, Pitch_R, Elbow_R, Wrist_Pitch_R, Wrist_Roll_R, Jaw_R` の6次元rad絶対位置。グリッパも角度。stateは実測qpos、actionは実ctrl適用済みsent。左右反転・符号・順序・物理・元の制御は維持する。

入力は現在の `observation.images.workspace` と `observation.state` のみ。RGB CHW float32 0〜1、640×480。画像縮小・拡張・クロップなし。物体座標・phase・task・索引・成功フラグ・正解actionを推論へ渡さない。訓練でも入力キーを明示制限する。

## 統計とaction列

ACTConfigのMEAN_STDを公式processorで一度だけ適用する。state/actionの元mean/stdは全数値9531行で照合し、元値との差が最大約2.1e-5であることを記録した上で、全9531行のfloat64集計による正確なpopulation mean/stdを専用統計へ保存する。元meta画像std約1e-5は画素分布と照合し、各episodeから等間隔10画像（計50枚）の全画素を逐次集計したSTEP 5A専用mean/stdへ置き換える。元metaは不変。derived statsと元値・対象画像索引を別保存し、学習と推論で共用する。小stdとグリッパを含む往復誤差許容1e-6。画像は既に0〜1であり255除算を重複しない。

`chunk_size=50`、action時刻オフセット0..49 /50秒。公式LeRobotDatasetのdelta_timestampsで構成。episode末尾は最後のactionを反復しaction_is_pad=True。次episodeへ越境しない。全5試行の先頭・中間・末尾15標本を保存actionと完全一致で照合する。

## ACTと少量学習

公式ACTを画像＋実測stateで使用。ResNet18、transformer dim512・head8・FF3200・encoder4・decoder1、VAE latent32・encoder4、KL係数10、dropout0.1。専用キャッシュに既存バックボーン重みなし。`pretrained_backbone_weights=None` を明示し、画像バックボーンの事前学習なしの接続試験とする。重み取得・外部ログ・Hub保存なし。

seed42、fp32、worker0。最初3更新batch1、実測確認後batch4。ACTConfigのAdamW標準lr1e-5（backboneも同値）、weight_decay1e-4、betas(0.9,0.999)、eps1e-8、clip norm10、schedulerなし。予定累積300更新、smoke含む絶対上限1000。最初／最終の固定診断5観測についてVAE lossと、正解actionを渡さないpredict_action_chunkのrad MAEを分ける。lossは学習内診断で、汎化・最終性能とは呼ばない。

全モデルのdigestとaction head・backbone conv1の実パラメータ差、有限loss/gradient、更新数を保存する。モデルはsmokeとfinalの2点のみ。optimizer・step・Python/NumPy/torch/CUDA/sampler RNGを保存。fp32なのでscalerなし。診断呼出しを含めた後のRNGを保存し、保存時点からの再開とする。

## 保存と新規コンテナ復元

`models/step05A/<run>/smoke|final` に全重みsafetensors、ACTConfig、公式前後processor定義と統計safetensors、専用統計JSON、仕様、データhash、resume.ptを保存。代表画像/state5点と、resetして10actionを消費した保存直前の出力をbundleに同梱。全ファイルをhash化する。

学習コンテナ終了後、reloadコマンドが新しいコンテナでfinalだけから `ACTPolicy.from_pretrained(local_files_only=True, strict=True)` と保存processorを復元する。全モデルdigestを照合。eval/inference mode・独立比較前reset、全5×10actionを事前許容atol1e-6rad、rtol1e-6で比較。許容差を事後変更しない。

## 閉ループと安全

元ExpertEnvとSTEP 3シーンを使用し、ContactExpertを生成・実行しない。診断用のcontact_records/storage_state/grasp_poseは物理観測器だけで使い、結果をaction生成へ返さない。リセット時だけ既知初期配置を与える。

各tickで現在RGBと実測qposを取得し、公式pre→select_action→post→安全検査→ctrlへ直接代入する。補償・積分・速度制限を二重適用しない。`n_action_steps=10`、ensembleなし、独自queueなし。各tick1action消費、queue空時のみ新規推論。全tickで画像を取得するが、ACT再予測は10tickごと。物理500Hzを10回進め、操作・画像50Hzを維持する。壁時計に合わせたskipなし。

初回rollout前に、保存sentの各軸最大1tick差×10tick＋1e-6radをジャンプ停止閾値として固定する。元の最大指令速度0.6rad/sから10tickの指令変位0.12rad程度を基準とする診断ゲートであり、予測actionを補正するフィルターではない。範囲外・非有限値・閾値超えは適用前停止、黙ったclipなし。各2msの物理ステップで非有限状態・MuJoCo warning・既存max_contact_penetration_m超えのロボット接触を停止する。

中央既知配置から同一試行の5tick→100tick→最大2500tick（50sim秒）へ進む。最大3試行、追加する場合も保存5配置だけ。安全停止は失敗として残す。段階通過、action/ctrl一致、時刻20ms増分、実測関節の移動を確認する。接続確認と把持・持上げ・収納を分けて報告する。

把持は両指接触・環境支持なし・40mm以上持上げを1秒、相対滑り8mm以内。phaseに依存しない物理観測器で判定する。収納は元の箱全体の包含・底支持・高さ1mm・解放・壁接触なし・速度閾値・2秒安定を再利用。真の物体情報は診断専用。

traceは各tickの観測時刻/state/ACT raw rad/ctrl/出力元/物理進行/診断を逐次gzip、画像は代表PNGのみ。確認動画は5tickごとの10fps、観測50Hzと区別し実sim速度に合わせて逐次CPUエンコードする。画像をメモリに蓄積しない。

## 資源と再実行

ホストGPU0とコンテナcuda0/EGL0のUUID・PCIを照合。開始前3回の低負荷（<=30%）・安定VRAM・空き16GiB以上を条件に開始し、他計算PIDの追加・使用量1GiB超増加、空き8GiB未満、ディスク余裕10GiB未満、新規出力6GiB超で自分のコンテナだけ停止する。監視は離散観測で排他や無影響を保証しない。torch allocated/reservedとnvidia-smiプロセス使用量を区別する。

学習・GPU推論・描画の各コンテナ実行時間を保守的に合算し2700秒まで（初期化・保存も含む）。学習1800秒、smoke300秒、reload180秒、各rollout600秒の個別上限も残時間で短縮。監視はホストスクリプト、Pythonは全て既存コンテナ。audit→smoke→train→reload→rolloutを順次実行する。新規runは同じ設定でも新しいIDにし、既存ログ・モデルを上書きしない。

```bash
bash scripts/step05A_run.sh audit NEW_RUN
bash scripts/step05A_run.sh smoke NEW_RUN
# smokeの更新・時間・VRAM/RAMを確認後
bash scripts/step05A_run.sh train NEW_RUN
bash scripts/step05A_run.sh reload NEW_RUN
bash scripts/step05A_run.sh rollout NEW_RUN 0
```

前段の成功を確認してから次へ進む。STEP 5Bの着手は管理スレッド確認後の別セッション。

## 初回rollout後の限定修正

300更新モデルは中央配置の初回actionでPitch差0.14324rad、Elbow差0.18927radが事前閾値約0.12000radを超え、適用前に停止した。元制御・安全閾値・観測・データ・optimizer設定は変えず、同じ5試行で追加300更新（累積600）を1回だけ行う。最初のfinalは保存し、3点目の `continued/` へ保存する。これ以上の追加学習は今回行わない。

```bash
bash scripts/step05A_run.sh continue act_20261029_01
bash scripts/step05A_run.sh reload act_20261029_01 1 continued
bash scripts/step05A_run.sh rollout act_20261029_01 1 continued
```

trial1は修正後の比較のため同じ中央配置にする。trial0も失敗として残し、最大3rollout・最大1000更新・合計45分は維持する。安全停止だけに終わる場合は一部完了で引き継ぐ。
