# 實驗紀錄

每跑完一個實驗新增一列。數字取 `best.pth` 對應的評估結果（含 flip TTA、cosine 距離、無 re-ranking）。
資料集：Market-1501（train 751 ids / 12936 imgs；query 3368；gallery 15913）。

| 日期 | 實驗 | Teacher | Student | KD 設定 | mAP | R1 | R5 | 參數量 | commit | 備註 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-09 | teacher_r50 | — | ResNet-50 | — | **86.6** | **95.0** | 98.2 | 25.1M | 44cee48 | 上界；Colab T4，~51 s/epoch，共約 2.3 h |
| | student_r18_baseline | — | ResNet-18 | — | | | | 11.2M | | 下界 |
| | student_r18_kd | ResNet-50 | ResNet-18 | logit 1.0, sim 1.0, T=4 | | | | 11.2M | | |

## 觀察

### teacher_r50（2026-10-09）
- 與 Bag-of-Tricks R50（mAP 85.9 / R1 94.5）相當，確認資料管線、BNNeck、評估協定正確。
- mAP 曲線：ep10 71.8 → ep30 75.1 → ep40 71.7（高 LR 階段的波動）→ ep50 82.9（第一次 LR 衰減後跳升）→ ep70 85.6 → ep120 86.6。ep70 之後只再漲 1 個點。
- CE 收斂在 ~1.05，接近 label smoothing ε=0.1、751 類下的理論下限，屬正常現象，不是沒學好。
- 評估（約 19k 張含 flip）每次約 2.5 分鐘，佔總時間約 8%。
- 注意：teacher 是用 label smoothing 訓練的。文獻（Müller et al. 2019）指出 LS 會壓縮 logit 中的類間相似度資訊，可能削弱 logit KD 的效果——之後可做一組無 LS 的 teacher 作為消融。
