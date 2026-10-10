# 實驗紀錄

每跑完一個實驗新增一列。數字取**最後一個 epoch** 的評估結果（含 flip TTA、cosine 距離、無 re-ranking）。
資料集：Market-1501（train 751 ids / 12936 imgs；query 3368；gallery 15913）。

| 日期 | 實驗 | Teacher | Student | KD 設定 | mAP | R1 | R5 | 參數量 | commit | 備註 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-09 | teacher_r50 | — | ResNet-50 | — | **86.6** | **95.0** | 98.2 | 25.1M | 44cee48 | 上界；Colab T4，~51 s/epoch，共約 2.3 h |
| 2026-10-09 | student_r18_baseline | — | ResNet-18 | — | 80.6 | 92.0 | 97.5 | 11.2M | 44cee48 | 下界；ep69 斷線後續跑 |
| 2026-10-09 | student_r18_kd | ResNet-50 | ResNet-18 | logit 1.0, sim 1.0, T=4 | **84.8** | **93.6** | 98.0 | 11.2M | d375e3e | **+4.2 mAP，補回 70% 差距**；~35 s/epoch |
| 2026-10-09 | osnet/1a_baseline | — | OSNet x0.25 | — | 64.3 | 83.2 | 94.6 | 0.2M | 766fe78 | **欠擬合**；論文 77.8 / 92.2；~36 s/epoch |
| 2026-10-10 | osnet/1c_lr1e-3 | — | OSNet x0.25 | — | 71.4 | 88.0 | 95.8 | 0.2M | 0c5ae15 | lr 1e-3，120 ep；+7.1 vs 1a |
| 2026-10-10 | osnet/1e_lr1e-3_240ep | — | OSNet x0.25 | — | **77.0** | **90.6** | 97.0 | 0.2M | 0c5ae15 | **B\***；lr 1e-3，240 ep，LR 降於 80/140；論文 77.8 / 92.2 |

## 觀察

### teacher_r50（2026-10-09）
- 與 Bag-of-Tricks R50（mAP 85.9 / R1 94.5）相當，確認資料管線、BNNeck、評估協定正確。
- mAP 曲線：ep10 71.8 → ep30 75.1 → ep40 71.7（高 LR 階段的波動）→ ep50 82.9（第一次 LR 衰減後跳升）→ ep70 85.6 → ep120 86.6。ep70 之後只再漲 1 個點。
- CE 收斂在 ~1.05，接近 label smoothing ε=0.1、751 類下的理論下限，屬正常現象，不是沒學好。
- 評估（約 19k 張含 flip）每次約 2.5 分鐘，佔總時間約 8%。
- 注意：teacher 是用 label smoothing 訓練的。文獻（Müller et al. 2019）指出 LS 會壓縮 logit 中的類間相似度資訊，可能削弱 logit KD 的效果——之後可做一組無 LS 的 teacher 作為消融。

### R50 → R18 傳統蒸餾（2026-10-09）
- **差距補回率 = (84.8 − 80.6) / (86.6 − 80.6) = 70%**；R1 +1.6。R18-KD 只比 teacher 低 1.8 mAP，參數不到一半。
- 這是 OSNet 實驗要對照的「傳統同質蒸餾」基準：OSNet 的補回率若明顯低於 70%，就證實 capacity gap。
- 損失量級：`kd_logit` ≈ 0.12、`kd_sim` ≈ 0.003（權重皆 1.0）。similarity 項的數值只有 logit 項的約 1/40，增益可能主要來自 logit KD——值得在 OSNet 階段 2 做「只用 logit」的對照，確認 similarity 項是否真的有貢獻。
- R18-KD（84.8）已經很接近 R50，可直接當 OSNet 的助教模型（plan 3a）：`outputs/student_r18_kd/best.pth`。
- 註：R18 baseline 在 ep69 斷線後從 last.pth 續跑；續跑不還原資料抽樣的亂數狀態，對結果影響應在單一 seed 雜訊範圍內。

### OSNet x0.25 1a baseline（2026-10-09）—— 欠擬合
- 最後 mAP 64.3 / R1 83.2，比論文（77.8 / 92.2）低 13.5 mAP。
- **證據指向欠擬合，而不是程式錯誤**：
  - 訓練損失沒收斂：最後 CE 1.26、triplet 0.13；R18 同配方收斂到 CE 1.07、triplet ≈ 0.002（label smoothing 下 CE 的下限約 1.05）。
  - ep40 第一次降 LR 時 CE 還在 1.37 且持續下降——LR 降得太早。降 LR 後 mAP 只從 61.4 升到 64.3；R50 同一時間點是 +11。
  - mAP 曲線 ep10 39.8 → ep40 61.4 → ep70 63.9 → ep120 64.3，ep70 之後學習率太小，幾乎停住。
- 原因：這套 BoT 配方（lr 3.5e-4、ep40/70 降 LR）是為大型預訓練 ResNet 調的；0.2M 參數的 OSNet 需要更大的學習率、更長的高 LR 階段。論文的 ImageNet 微調配方是 lr 1.5e-3、每 60 epoch 才降一次、共 150 epoch。
- 待確認：ImageNet 權重是否完整載入。此次 log 沒有載入統計；已在 train.py 加入 `pretrained: OSNet ImageNet: backbone N/558 tensors` 這行，下次執行確認是 558/558。
- 速度：~36 s/epoch，`data 0–1s`，資料載入不是瓶頸（GPU 上 depthwise 卷積本來就不快）。
- 下一步：1c（lr 1e-3）與新增的 1e（lr 1e-3 + 240 ep），1b/1d/1s 延後。

### OSNet x0.25 1c / 1e —— 欠擬合解決，選定 B*（2026-10-10）
- ImageNet 權重確認完整載入：`backbone 558/558 tensors, embed 7 tensors`。
- **1c（lr 1e-3）71.4 / 88.0**：比 1a 高 7.1 mAP，證實 LR 太小是主因；但 ep70 之後又停住（71.1 → 71.4），因為 ep40 就降 LR。
- **1e（lr 1e-3 + 240 ep，ep80/140 降 LR）77.0 / 90.6**：比 1a 高 12.7 mAP，距論文 77.8 只差 0.8。關鍵是高 LR 階段從 40 拉長到 80 epoch——ep80 降 LR 時 mAP 從 66.6 跳到 74.4。
- 1c 與 1e 前 40 epoch 的 mAP 完全相同（同 seed、同設定），確認訓練可重現。
- 1e 結束時 CE 1.115、triplet 0.014，仍略高於 R18 的 1.07 / 0.002，但 ep140 之後 mAP 只再漲 0.6——可能已接近 0.2M 參數的容量上限，再調 LR 預期收益有限。
- **決定：B\* = 1e 配方**（`configs/osnet/_bstar.yaml`），未達「及格線 79」但已逼近論文；接下來進入 KD。OSNet 的 teacher–baseline 差距為 86.6 − 77.0 = **9.6 mAP**（R18 是 6.0）。
- 時間成本：240 ep ≈ 2.6 h（~35 s/epoch）；KD 另加 teacher forward，預估 3–3.5 h/次。
