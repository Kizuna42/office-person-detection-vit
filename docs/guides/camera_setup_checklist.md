# カメラ設置・撮影チェックリスト

現場での撮影担当者が座標変換に必要なデータを漏れなく取得できるようにするためのチェックリスト。
取得した値は `config/calibration_template.yaml` / `.json` をベースに `config.yaml` へ反映する。

---

## 1. 事前準備

- 最新のフロアマップ画像（`data/floormap.png`）と実寸スケールを確認
- マーカー（蛍光テープやA4紙）を 10〜20 点程度用意し、床面に貼付できるようにする
- メジャーまたはレーザー距離計、三脚、水平器
- ノートPC もしくはタブレット（対応点入力や撮影メモ書き込み用）

---

## 2. 現地で計測する値（config への対応表）

| 測定項目 | 目的 | config キー |
| --- | --- | --- |
| カメラ ID / 設置場所 | 複数台管理の識別子 | （任意）`camera_id` |
| カメラ投影位置（フロアマップ座標） | フロアマップ上にカメラを描画 | `camera.position_x`, `camera.position_y` |
| カメラ高さ（m） | 俯角計算 / UI表示 | `camera.height_m` |
| 視野に写る基準点 10〜20 点 | ホモグラフィ推定 | `homography.matrix` を求める元データ |
| フロアマップ原点と基準長さ | 実寸スケール確認 | `floormap.image_origin_{x,y}`, `image_*_mm_per_pixel` |
| カメラ内部パラメータ（焦点距離, 主点） | 歪み補正が必要な場合 | `calibration.intrinsics.*` |
| 歪み係数（k1〜k3, p1, p2） | 歪み補正 | `calibration.distortion.*` |

---

## 3. 撮影ルーチン

1. **フロアマップ対応点の設置**
   - ゾーンの境界や柱位置にマーカーを配置
   - それぞれの座標をフロアマップ画像上で計測し、一覧に記録
2. **撮影**
   - ターゲットとなる時刻帯で静止画を複数枚撮影（人が写っていない状態が理想）
   - マーカーが十分に写る角度で撮影し、対応点を後でクリックできるようにする
3. **メモ**
   - 三脚設置位置、角度、焦点距離（ズーム値）、カメラ ID をメモ
   - `camera.position_*` に対応するフロアマップ上の座標を記録
4. **データ回収**
   - 撮影した画像とメモを `output/calibration/` 等にまとめる

---

## 4. 撮影後の確認リスト

- [ ] 対応点 10 点以上について「カメラ画像座標」「フロアマップ座標」がペアで揃っている
- [ ] `config/calibration_template.yaml` をコピーし、各項目を実測値に更新した
- [ ] `output/calibration/correspondence_points_<camera>.json` を作成し、`scripts/evaluate_reprojection_error.py` で mean/max 誤差を確認した
- [ ] `camera.position_x / position_y / height_m` が `docs/architecture/floormap_integration.md` の原点定義と整合している

---

## 5. 提出物

1. `config/calibration_template.yaml`（もしくは `.json`）を実測値で埋めたファイル
2. 対応点 JSON (`output/calibration/correspondence_points_<camera>.json`)
3. 参照画像（ハイライト済みでも可）とマーカー配置図
4. 再投影誤差レポート (`output/calibration/reprojection_cam01_*.json/.png`)

これらを揃えることで、開発側は `ConfigManager` の検証を通しつつホモグラフィ精度を再現できる。***
