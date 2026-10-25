# 實驗紀錄

每跑完一個實驗新增一列。數字取**最後一個 epoch** 的評估結果（含 flip TTA、cosine 距離、無 re-ranking）。
資料集：Market-1501（train 751 ids / 12936 imgs；query 3368；gallery 15913）。CUHK03-NP 與跨域結果見下方各節。參數量欄位為推論模型（不含訓練用 ID 分類頭）：OSNet x0.25 0.2M、OSNet x1.0 2.2M。

| 日期 | 實驗 | Teacher | Student | KD 設定 | mAP | R1 | R5 | 參數量 | commit | 備註 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-09 | teacher_r50 | — | ResNet-50 | — | **86.6** | **95.0** | 98.2 | 25.1M | 44cee48 | 上界；Colab T4，~51 s/epoch，共約 2.3 h |
| 2026-10-09 | student_r18_baseline | — | ResNet-18 | — | 80.6 | 92.0 | 97.5 | 11.2M | 44cee48 | 下界；ep69 斷線後續跑 |
| 2026-10-09 | student_r18_kd | ResNet-50 | ResNet-18 | logit 1.0, sim 1.0, T=4 | **84.8** | **93.6** | 98.0 | 11.2M | d375e3e | **+4.2 mAP，補回 70% 差距**；~35 s/epoch |
| 2026-10-09 | osnet/1a_baseline | — | OSNet x0.25 | — | 64.3 | 83.2 | 94.6 | 0.2M | 766fe78 | **欠擬合**；論文 77.8 / 92.2；~36 s/epoch |
| 2026-10-10 | osnet/1c_lr1e-3 | — | OSNet x0.25 | — | 71.4 | 88.0 | 95.8 | 0.2M | 0c5ae15 | lr 1e-3，120 ep；+7.1 vs 1a |
| 2026-10-10 | osnet/1e_lr1e-3_240ep | — | OSNet x0.25 | — | **77.0** | **90.6** | 97.0 | 0.2M | 0c5ae15 | **B\***；lr 1e-3，240 ep，LR 降於 80/140；論文 77.8 / 92.2 |
| 2026-10-10 | osnet/2a_kd_r50 | ResNet-50 | OSNet x0.25 | logit 1.0, sim 1.0, T=4 | 82.6 | 93.3 | 97.8 | 0.2M | c464de6 | +5.6，補回 58% |
| 2026-10-10 | osnet/2c_kd_r50_at | ResNet-50 | OSNet x0.25 | 2a + attention 4.0 | 82.6 | 93.0 | 97.9 | 0.2M | c464de6 | AT 無增益 |
| 2026-10-10 | osnet/2d_kd_r50_logit | ResNet-50 | OSNet x0.25 | logit 1.0, T=4 | **82.7** | **93.1** | 97.7 | 0.2M | c464de6 | **D\***；+5.7，補回 59% |
| 2026-10-10 | osnet/3a_ta_r18kd | R18-KD (84.8) | OSNet x0.25 | logit 1.0, T=4 | 82.6 | 93.0 | 97.8 | 0.2M | 99aa918 | 助教無增益（vs D\* −0.1） |
| 2026-10-10 | osnet/3d_dkd_r50 | ResNet-50 | OSNet x0.25 | **DKD** α1 β8, T=4 | **85.0** | **93.9** | 97.9 | 0.2M | 99aa918 | **+8.0，補回 83%**；vs D\* +2.3 |
| 2026-10-10 | osnet/3c_multi_r50_r18kd | R50 + R18-KD | OSNet x0.25 | logit 1.0 ×0.5 each | 82.1 | 92.5 | 97.5 | 0.2M | 99aa918 | 雙 teacher 反而 −0.6 |
| 2026-10-11 | teacher_r50_nols | — | ResNet-50（無 LS） | — | **87.1** | 94.7 | 98.3 | 23.5M | edac964 | 無 label smoothing；best = final；CE 0.032 |
| 2026-10-11 | osnet/3g_kd_r50nols_logit | R50 無 LS | OSNet x0.25 | logit 1.0, T=4 | 83.8 | 93.4 | 97.7 | 0.2M | edac964 | vs D\* +1.1、vs DKD −1.2；補回 67% |
| 2026-10-11 | osnet/3h_dkd_r50nols | R50 無 LS | OSNet x0.25 | DKD α1 β8, T=4 | 84.5 | 94.0 | 98.0 | 0.2M | f35d426 | 未疊加：vs 3d −0.5 mAP / +0.1 R1 |
| 2026-10-11 | osnet/3d_dkd_r50_s1 | ResNet-50 | OSNet x0.25 | 3d，seed 1 | 85.0 | 94.1 | 98.1 | 0.2M | 58b7725 | seed 重複 |
| 2026-10-11 | osnet/3d_dkd_r50_s2 | ResNet-50 | OSNet x0.25 | 3d，seed 2 | 85.0 | 94.1 | 98.0 | 0.2M | 58b7725 | seed 重複 |
| 2026-10-11 | student_r18_dkd | ResNet-50 | ResNet-18 | DKD α1 β8, T=4 | **86.2** | **94.4** | – | 11.2M | db67f47 | vs R18-KD +1.4；補回 93%；距 teacher 僅 0.4 |
| 2026-10-11 | osnet/3d_dkd_r50_b16 | ResNet-50 | OSNet x0.25 | DKD α1 **β16** | 85.0 | 94.2 | – | 0.2M | db67f47 | 與 β8 相同 |
| 2026-10-12 | osnet/3d_dkd_r50_b4 | ResNet-50 | OSNet x0.25 | DKD α1 **β4** | 84.5 | 93.8 | – | 0.2M | db67f47 | vs β8 −0.5 |
| 2026-10-14 | clip/4a_dkd_clip | CLIP-ReID（×8.26） | OSNet x0.25 | DKD α1 β8, T=4 | 83.4 | 92.8 | 97.4 | 0.2M | 031d5d3 | vs F\* −1.6；跨域 11.3 |
| 2026-10-15 | clip/4a_dkd_clip_s1 | CLIP-ReID（×8.26） | OSNet x0.25 | 4a，seed 1 | 83.5 | 92.3 | 97.5 | 0.2M | d36d6ff | 4a = 83.45 ± 0.07 |
| 2026-10-14 | clip/4b_logit_clip | CLIP-ReID（×8.26） | OSNet x0.25 | logit 1.0, T=4 | 83.2 | 93.1 | 97.6 | 0.2M | 031d5d3 | vs D\* +0.6；跨域 10.1 |
| 2026-10-14 | clip/4c_simdist_clip | CLIP-ReID | OSNet x0.25 | simdist 1.0（無 logits） | 78.9 | 91.7 | – | 0.2M | d36d6ff | 只比 B\* +1.5 |
| 2026-10-15 | osnet_x1/6b_dkd_r50_x1 | ResNet-50 | **OSNet x1.0** | DKD α1 β8, T=4 | **88.3** | 95.3 | – | 2.2M | e34ae8d | **超過 teacher +1.8** |
| 2026-10-16 | osnet_x1/6c_dkd_clip_x1 | CLIP-ReID（×8.26） | **OSNet x1.0** | DKD α1 β8, T=4 | **89.5** | **95.4** | – | 2.2M | e34ae8d | vs 6b +1.2；距 CLIP-ReID 0.4 |
| 2026-10-17 | osnet_x1/6c_dkd_clip_x1_s1 | CLIP-ReID（×8.26） | **OSNet x1.0** | 6c，seed 1 | 89.5 | 95.2 | – | 2.2M | 34e14ef | **6c = 89.5 ± 0.0（2 seeds）** |
| 2026-10-18 | osnet_x1/6a_bstar_x1 | — | **OSNet x1.0** | — | 85.8 | 94.9 | – | 2.2M | 9853425 | x1.0 baseline；OSNet 論文 84.9 / 94.8 |
| 2026-10-18 | osnet_x1/6b_dkd_r50_x1_s1 | ResNet-50 | **OSNet x1.0** | 6b，seed 1 | 88.3 | 95.0 | – | 2.2M | 9853425 | **6b = 88.3 ± 0.0（2 seeds）** |

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

### 階段 2：直接蒸餾 R50 → OSNet x0.25（2026-10-10）
- **三組結果幾乎相同**：2a 82.6、2c 82.6、2d 82.7（R1 93.0–93.3）。差異 ≤ 0.1，在單一 seed 雜訊內。
  → **similarity 與 attention transfer 都沒有貢獻，增益全部來自 logit KD。** 與 R50→R18 中 `kd_sim` 只有 `kd_logit` 1/40 的觀察一致。`kd_at` 數值（0.235）不小，但沒有轉化成 mAP。
- **D\* = 2d（只用 logit KD）**：最簡單、結果最好。
- **差距補回率 59%**（+5.7 / 9.6），低於 R50→R18 的 70%：存在 capacity gap，但不算嚴重。絕對增益（+5.7）反而比 R18（+4.2）大，因為 OSNet 的起點更低、可補的空間更大。
- 佐證：訓練結束時 OSNet 的 `kd_logit` ≈ 0.215，R18 是 ≈ 0.12——0.2M 的學生較難貼近 teacher 的輸出分佈。
- **里程碑**：0.2M 參數的 OSNet-KD（82.7 / 93.1）**超過 11.2M 參數、無 KD 的 ResNet-18（80.6 / 92.0）**，參數少 56 倍；也達到計畫的「目標」等級（≥ 82）。
- 時間：~37–39 s/epoch（teacher forward 只多約 2–4 s），240 ep ≈ 2.7 h。
- 下一步（階段 3，都只用 logit 類 KD）：3a 助教 R18-KD、3d DKD、3c 雙 teacher。

### 階段 3：DKD 突破，助教與雙 teacher 無效（2026-10-11）
- **3d DKD：85.0 / 93.9**，比 D\*（logit KD）高 **+2.3 mAP**，比 B\* 高 +8.0，**補回 83%**——超過 R50→R18 傳統蒸餾的 70%。只比 R50 teacher 低 1.6 mAP，參數是它的 **1/116**。達到計畫「理想」等級（≥ 84）。
- **3a 助教（R18-KD）82.6**、**3c 雙 teacher 82.1**：都沒有超過直接用 R50 的 82.7。換「更好學的 teacher」沒有幫助。
- **解讀**：瓶頸不是 teacher 太強（capacity gap），而是**經典 logit KD 傳遞的訊號太弱**。我們的 teacher 用 label smoothing 訓練、且對訓練集的目標類別非常有把握，非目標類別的機率被壓得很平；經典 KD 中 NCKD（非目標類之間的相對關係，即「暗知識」）的權重與 teacher 的目標類信心耦合，信心越高越被壓抑。DKD 把 NCKD 獨立出來並乘上 β=8，正好放大了這部分訊號。
  - 這也解釋了助教為何沒用：R18-KD 同樣有 label smoothing 與高信心，換 teacher 不改變 NCKD 被壓抑的問題。
- log：DKD 結束時 `ce` 1.167（logit KD 是 1.105）——學生對 hard label 的擬合略差、但檢索更好，符合「學到更多類間關係」的解讀。`kd_dkd` ≈ 1.07。
- **待驗證**（單一 seed；+2.3 遠大於目前觀察到的 ±0.1–0.3 波動，但仍需重複）：
  1. R50→R18 用 DKD 是否也有同樣幅度的增益（一般性 vs. 只對大差距有效）→ `student_r18_dkd.yaml`
  2. β 敏感度 → `3d_dkd_r50_b4.yaml`、`3d_dkd_r50_b16.yaml`
  3. 機制：無 label smoothing 的 teacher + 經典 logit KD 能否追上 DKD → `teacher_r50_nols.yaml` + `3g_kd_r50nols_logit.yaml`
  4. 最終配方 3 個 seed
- 設定修正：`kd.losses` 改為整個取代、不與父設定合併（避免子設定意外繼承 logit/similarity）。已完成的實驗設定皆正確（log 中只出現預期的 KD 項）。

### 機制驗證：label smoothing 解釋了 DKD 優勢的約一半（2026-10-11）
- **無 LS 的 R50 teacher：87.1 / 94.7**，比有 LS 的 86.6 還高 0.5（ReID 檢索不一定需要 LS）。best = final，3g 用 best.pth 沒有測試集選擇問題。
- **3g：同樣的經典 logit KD，換成無 LS teacher → 83.8**，比 D\*（82.7）高 1.1，但仍比 DKD（85.0）低 1.2。
- **結論：label smoothing 是原因之一，但不是全部。** 去掉 LS 補回了 DKD 優勢（+2.3）的約 48%；剩下的一半來自 DKD 本身的損失設計（把目標類與非目標類解耦、放大 NCKD）。
- 注意兩個混在一起的因素：無 LS teacher 本身也較強（+0.5）。用各自 teacher 計算補回率：3g 67% vs D\* 59% vs DKD 83%。
- log：3g 結束時 `kd_logit` ≈ 0.447（D\* ≈ 0.215）——無 LS teacher 的輸出分佈更尖、非目標類差異更大，學生較難貼近但學到更多。
- **下一步：3h = DKD + 無 LS teacher**，看兩個效果能否疊加（`configs/osnet/3h_dkd_r50nols.yaml`）。

### 3h：DKD + 無 LS teacher 沒有疊加（2026-10-11）
- **3h 84.5 / 94.0** vs 3d（DKD + 有 LS teacher）85.0 / 93.9：mAP −0.5、R1 +0.1，在單一 seed 雜訊邊緣，**沒有疊加效果**。
- **這支持機制解釋**：「去掉 LS」和「DKD」作用在同一個地方——讓學生拿到更多非目標類的暗知識。teacher 不再壓抑暗知識後，DKD 的放大就沒有額外可補的空間，兩者是替代關係，不是互補。
- 可能還略有過頭：3h 的 `kd_dkd` ≈ 2.21（3d 是 1.07）、CE 1.217（3d 1.167）——無 LS teacher 的非目標類訊號本來就強，再乘 β=8 可能過度加權。β 掃描可以順便回答（若需要，可加做 3h + β=4）。
- **最佳模型仍是 3d（85.0 / 93.9）**，下載包維持 3d。
- **雜訊估計變得必要**：目前有多組比較的差距落在 0.5–1.2，沒有 seed 間變異的數字就無法判斷。下一步優先用 3d 再跑 2 個 seed。

### seed 重複：DKD 非常穩定（2026-10-12）
- **3d 三個 seed（42 / 1 / 2）：mAP 85.0 / 85.0 / 85.0，R1 93.9 / 94.1 / 94.1** → mAP 85.0 ± 0.0、R1 94.0 ± 0.1（n = 3）。
- 確認 seed 確實生效：兩次重跑從第 1 個 iteration 的損失就不同（ce 6.567 vs 6.565），中途評估也不同（ep20 78.3 vs 78.9），只是最終收斂到同一點。
- **雜訊基準**：同一設定的最終 mAP 變異 ≲ 0.1（n 很小，只能說明量級）。據此：
  - 3g 對 D\* 的 +1.1、3h 對 3d 的 −0.5、3c 對 D\* 的 −0.6 都遠大於 seed 變異，**可視為真實差異**；
  - 2a / 2c / 2d 之間 ≤ 0.1 的差異確實是「沒有差別」。
- 注意：雜訊只在 DKD 設定上測過；其他設定的變異假設同量級。論文最終表格建議主要比較組（B\*、D\*、3d）都報 3 seed 平均。

### R18-DKD 與 β 掃描（2026-10-12）
- **R18-DKD 86.2 / 94.4**：比 R18 傳統 KD（84.8）高 +1.4，補回率 70% → **93%**，只比 R50 teacher 低 0.4 mAP（參數不到一半）。
- **DKD 是普遍有效，不是只對大差距有效**：
  | 學生 | baseline | 傳統 KD | DKD | DKD − 傳統 KD | 補回率（KD → DKD） |
  |---|---|---|---|---|---|
  | ResNet-18 | 80.6 | 84.8 | 86.2 | +1.4 | 70% → 93% |
  | OSNet x0.25 | 77.0 | 82.7 | 85.0 | +2.3 | 59% → 83% |
  - 兩者的補回率都提升約 24 個百分點。OSNet 的絕對增益較大（+2.3 vs +1.4），但只是因為它剩下的差距較大；以「剩餘差距被補回的比例」看，R18 是 78%、OSNet 是 59%——**DKD 並沒有特別偏好差距大的學生**。
  - 因此「DKD 專門解決 capacity gap」的說法不成立；較準確的說法是：**在用 label smoothing 訓練的 ReID teacher 下，DKD 普遍優於傳統 KD；小學生仍保有較大的剩餘差距**。
- **β 掃描（OSNet）**：β4 84.5、β8 85.0、β16 85.0 → NCKD 權重不足時較差，β ≥ 8 後飽和。與「暗知識被壓抑、需要放大」的解釋一致，且 β8（原論文預設）就夠了，不需要調。
- 雜訊基準（3 seed ≲ 0.1）下，β4 的 −0.5 與 R18 的 +1.4 都是真實差異。

### 跨域 zero-shot：Market-1501 → DukeMTMC-reID（2026-10-12）
`tools/eval_matrix.py`，各 run 的 `last.pth`、無 re-ranking；結果存在 Drive `MyDrive/KD-ReID/eval_matrix.json`。下表為 flip TTA（無 flip 的數字一致低 0.2–0.5，排名不變）。

| run | 參數 | Market mAP / R1 | Duke mAP / R1 / R5 | 保留率（Duke ÷ Market mAP） | 跨域補回率 |
|---|---|---|---|---|---|
| teacher R50 | 23.5M | 86.6 / 95.0 | 19.7 / 32.9 / 47.0 | 22.7% | – |
| R18 baseline | 11.2M | 80.6 / 92.0 | 13.0 / 25.0 / 37.8 | 16.1% | 0% |
| R18 KD | 11.2M | 84.8 / 93.6 | 18.5 / 33.9 / 49.6 | 21.8% | 82% |
| OSNet B\*（1e） | 0.2M | 77.0 / 90.6 | 12.7 / 25.1 / 40.6 | 16.5% | 0% |
| OSNet D\*（2d，logit） | 0.2M | 82.7 / 93.1 | 16.8 / 30.4 / 46.2 | 20.3% | 59% |
| **OSNet F\*（3d，DKD）** | 0.2M | 85.0 / 93.9 | **20.3 / 34.9 / 50.3** | **23.9%** | **109%** |

（跨域補回率 = (KD − baseline) ÷ (teacher − baseline)，用 Duke mAP 計算。）

- **沒有過擬合到 Market**：蒸餾在同域的增益在跨域完全保留，排名與同域一致（B\* < D\* < F\*、R18 base < R18 KD），所有 KD 學生的跨域 mAP 都比對應 baseline 高 4–8。
- **F\* 跨域追平甚至略超 teacher**：Duke mAP 20.3 vs 19.7、R1 34.9 vs 32.9、R5 50.3 vs 47.0；保留率 23.9% 高於 teacher 的 22.7%。同域 F\* 還差 teacher 1.6 mAP，跨域則沒有差距。
  - 合理解釋：teacher 帶 label smoothing、對 Market 的 751 個 ID 過度自信；DKD 學到的是「類間相對關係」而非 Market 特定的硬決策，泛化較好。R18 KD 的跨域 R1（33.9）也高於 teacher（32.9），方向一致。
  - **注意**：只有單 seed，+0.6 mAP 的差距在跨域的雜訊量級未知（同域 seed 變異 ≲ 0.1，但跨域通常較大）。目前能下的結論是「F\* 跨域與 teacher 相當」，「超過 teacher」要等 3d 另兩個 seed 的跨域數字。
- **DKD 在跨域的優勢比同域更大**：同域 F\* − D\* = +2.3，跨域 +3.5（16.8 → 20.3）；同域補回率 83%，跨域 109%。
- 絕對數字（~20 mAP）是 Market → Duke 無適應的正常水準，這張表只看相對排名與保留率。
- 未納入：3g、3h。3d seed 1 / 2 與 R18-DKD 見下一節。

### 跨域補充：3d 三個 seed 與 R18-DKD（2026-10-12）
結果存在 Drive `MyDrive/KD-ReID/eval_matrix_seeds.json`（flip TTA，無 re-ranking）。

| run | Market mAP / R1 | Duke mAP / R1 / R5 | 保留率 |
|---|---|---|---|
| 3d seed 42 | 85.0 / 93.9 | 20.3 / 34.9 / 50.3 | 23.9% |
| 3d seed 1 | 85.0 / 94.1 | 20.9 / 35.0 / 51.0 | 24.6% |
| 3d seed 2 | 85.0 / 94.1 | 20.7 / 35.5 / 51.4 | 24.4% |
| **3d 平均 ± sd（n = 3）** | 85.0 ± 0.0 / 94.0 ± 0.1 | **20.6 ± 0.3 / 35.1 ± 0.3 / 50.9 ± 0.6** | 24.3% |
| teacher R50（單 seed） | 86.6 / 95.0 | 19.7 / 32.9 / 47.0 | 22.7% |
| R18 KD（logit + similarity） | 84.8 / 93.6 | 18.5 / 33.9 / 49.6 | 21.8% |
| **R18 DKD** | 86.2 / 94.4 | **17.9 / 32.8 / 46.2** | 20.8% |

- **F\* 跨域 ≥ teacher 成立**：三個 seed 的 Duke mAP 都高於 teacher（20.3–20.9 vs 19.7），平均 +0.9 ≈ 3 個 sd；R1 +2.2、R5 +3.9。跨域補回率 113%。保留但需註明：teacher 只有單 seed，其跨域變異未知。
- **跨域的 seed 雜訊比同域大**：同域 mAP sd ≈ 0.0，跨域 ≈ 0.3。之後比較跨域數字時，< 0.6 的差距不宜下結論。
- **R18-DKD 跨域沒有變好**：同域比 R18 KD 高 1.4，跨域反而 −0.6 mAP、R5 −3.4（保留率 21.8% → 20.8%）。−0.6 約 2 個 sd，屬邊緣；R5 的 −3.4 較明確。至少可以說 **DKD 對 R18 沒有帶來跨域增益**。
  - 因此「DKD 的跨域優勢比同域更大」**只對 OSNet 成立**，不能推廣成 DKD 的一般性質。
  - 可能原因（未驗證）：(1) R18 KD 多了 similarity 項，可能有助泛化；(2) R18 容量大，β = 8 放大的 NCKD 讓它更貼近 Market 特有的類間結構（R18-DKD 同域幾乎追平 teacher，跨域也落在 teacher 之下）；OSNet 容量小，只學得到較通用的部分。兩個 baseline 跨域幾乎相同（13.0 vs 12.7），所以不是單純的架構差異。
  - 要分辨需要 R18 只用 logit KD、或 R18-DKD 多 seed，屬選擇性實驗，不影響 F\* 的結論。

### 打包 F\*（2026-10-12）
- `tools/export_package.py --run outputs/osnet/3d_dkd_r50` → `MyDrive/KD-ReID/packages/kdreid-osnet025-dkd-v1.zip`。
- 匯出時用這份權重重新評估：flip 84.98 / 93.94 / R5 97.86，no flip 84.71 / 93.97，與訓練 log 一致。
- 數值檢查：獨立模型 vs 訓練模型 0.0；ONNX vs PyTorch 2.6e-6；`selftest` 兩個 backend 皆通過。
- selftest 在 Colab CPU 印出 ONNX 19.3 ms / PyTorch 48.2 ms：舊版 selftest 沒有暖機、取 20 次平均，含首次呼叫的初始化，且 Colab CPU 共用、較弱，**不代表部署延遲**（benchmark 暖機後中位數為 2.3 ms）。已改成暖機 10 次後取 50 次中位數（commit 見 git log）。

### 外部系統測試（使用者自有場景，2026-10-12；應用佐證，不進主表）
使用打包的 F\*（`kdreid-osnet025-dkd-v1`）在使用者的系統上測試，數字由使用者回報：

| 測試 | F\*（OSNet x0.25 + DKD） | 對照 |
|---|---|---|
| 實際場景 C01 → C02 | 93.80 R1 / 97.67 R5 / 84.90 mAP | CLIP-ReID：100 R1 / 97.85 mAP |
| 實際場景 C02 → C01 | 96.77 R1 / 89.44 mAP | – |
| Panorama temporal（210 / 713 queries） | 89.52 R1 / 68.68 mAP | – |
| 完整 2661-frame 系統，ReID 用 ONNX CUDA | 12.00 FPS | CLIP-ReID CUDA：10.27 FPS |
| 完整系統，ReID 用 CPU | 11.06 FPS | – |

- 在沒訓練過的真實場景能用，CPU 版整個系統仍有 11 FPS。
- 與 CLIP-ReID 仍有明顯精度差距（mAP 84.9 vs 97.9）。整體 FPS 只差 17%，代表瓶頸在偵測 / 追蹤等其他模組，ReID 本身已不是瓶頸。
- 這組資料規模小、與論文協定不同，只作為應用佐證；論文驗證依標準資料集（階段 5：DukeMTMC-reID 主表）。

### 決策：論文資料集與目標順序（2026-10-12）
- **不使用 DukeMTMC-reID**（已撤回；CVPR 2026 作者指引、ECCV 2026 倫理指引）。上面的 Market → Duke 跨域結果保留作內部參考，論文改用 Market → CUHK03-NP / MSMT17。
- 論文資料集：Market-1501、CUHK03-NP（detected 為主）、MSMT17（申請中）。
- 目標順序：先完成問題 1（DKD 讓 OSNet 逼近大模型：多資料集 + 統計穩定性），再做問題 2（跨相機系統的強模型）。
- 問題 2 的前置診斷：把 R50 teacher 打包進使用者系統，判斷 Panorama 落後 CLIP-ReID 是 teacher 的問題還是蒸餾的問題。

### 問題 2 前置診斷：實際場景中的 teacher / student / CLIP-ReID（2026-10-12；使用者系統，輔助證據）
TTA OFF、re-ranking OFF，三個模型同一套評估流程。數字由使用者回報。

| 場景（query / gallery） | CLIP-ReID mAP / R1 | R50 teacher mAP / R1 | OSNet F\* mAP / R1 |
|---|---|---|---|
| C01 → C02（129 / 124） | 97.85 / 100.00 | 81.56 / 91.47 | **84.90** / 93.80 |
| C02 → C01（124 / 129） | 97.65 / 100.00 | 84.81 / 97.58 | **89.44** / 96.77 |
| Panorama（210 / 713，同影片跨時間） | 78.01 / 91.43 | **70.10** / 89.05 | 68.68 / 89.52 |

- **瓶頸在 teacher**：R50 teacher 三個場景都落後 CLIP-ReID 7.9–16.3 mAP。這遠大於小樣本的雜訊（每個場景約 130–210 個 query，粗估 mAP 標準誤約 2 pp）。
- **沒有蒸餾退化的跡象**：跨相機兩個方向 student 比 teacher 高 3.3 / 4.6，Panorama 低 1.4。這和 Market → Duke 的結果（F\* 20.6 vs teacher 19.7）方向一致：小學生跨域不輸 teacher。單一場景的差距在雜訊邊緣，不宜個別解讀。
- **結論**：問題 2 的方向是「換更會泛化的 teacher（CLIP-ReID）」，而不是先改 DKD loss。
- 未完成：同場景的無 KD OSNet（B\*）與 logit KD（D\*），用來量化 KD 對跨相機的淨效益（notebook 7.6 第三格打包）。
- 注意：Market 上 CLIP-ReID 只比 R50 多約 3 mAP（文獻值，視 backbone 而定），在實際場景卻多 8–16。差異主要來自 CLIP 預訓練帶來的泛化，不是同域精度。這正是蒸餾想要轉移的東西。

### ⚠️ 「CUHK03-NP」第一輪其實是 DukeMTMC-reID（2026-10-13）
`configs/cuhk03/` 第一輪的 log 顯示資料集為 **702 人 / 16,522 張 / 8 台相機，query 2,228、gallery 17,661**。這正好是 DukeMTMC-reID 的官方分割；CUHK03-NP detected 應為 767 人 / 7,365 張，query 1,400、gallery 5,332，每人只出現在 2 台相機。
- **原因（程式錯誤）**：Kaggle 的 `archive_DukeMTMC.zip` 很可能把 `bounding_box_train/` 等資料夾直接解壓在 `/content/data` 根目錄。舊的 `_locate` 在找不到 `root/detected` 時，會先檢查 `root` 本身，結果任何「預設子資料夾不存在」的資料集名稱都會拿到 Duke。先前的 Market → Duke 跨域評估也是這樣找到 Duke 的（那次剛好是對的）。
- 修正：root 本身不再優先，只在它是唯一候選時才使用；有多個候選時用名稱關鍵字挑，仍分不出來就改用「官方分割大小完全相符」來挑，都不符則報錯並列出每個候選的實際數字。
- 修正：`build_dataset` 現在會核對官方分割大小，不符就直接報錯，並指出看起來像哪個資料集；log 也會印出實際讀到的資料夾。
- 這批結果**不能進論文**（Duke 已撤回），只作內部參考。它在第二個資料集上獨立重現了 Market 的排名與補回率型態（Final，最後一個 epoch）：

| Duke（內部參考） | mAP | 補回率 |
|---|---|---|
| R50 teacher | 77.1 / R1 87.3 | – |
| R18 baseline / KD / DKD | 70.2 / 74.5 / **76.2**（R1 84.2 / 85.7 / 86.7） | KD 62% → DKD 87% |
| OSNet B\* / D\* / F\* | 66.7 / 71.9 / **74.0**（R1 81.7 / 84.8 / 85.5） | D\* 50% → F\* 70% |

  - 和 Market 一樣：DKD 在兩個學生上都勝過傳統 KD（R18 +1.6、OSNet +2.2），OSNet 的補回率低於 R18。
- **確認（2026-10-13）**：`/content/data` 頂層確實有 Duke 的 `bounding_box_train/query/bounding_box_test`。修正後 `cuhk03np` 讀到 `cuhk03-np/detected`：767 人 / 7,365 張 / 2 台相機，query 1,400、gallery 5,332（700 人），與官方分割完全相符。
- Drive 上的輸出已改名為 `outputs/duke_internal/`（notebook 7.7 第一格），避免重跑真正的 CUHK03-NP 時從這些 checkpoint 續跑。

### 5a：CUHK03-NP（detected）主表，OSNet 部分（2026-10-13）
與 Market 完全相同的配方（`configs/cuhk03/`），只換資料集；KD teacher 為 CUHK03 teacher 的 `last.pth`。資料夾 `cuhk03-np/detected`，767 / 7,365（train）、700 / 1,400（query）、700 / 5,332（gallery），與官方分割相符。Final（最後一個 epoch），flip TTA，無 re-ranking；commit c210e74；單一 seed（42）。

| 實驗 | mAP | R1 | R5 | 補回率 | 備註 |
|---|---|---|---|---|---|
| cuhk03/teacher_r50 | **61.4** | 62.4 | – | – | 上界；120 ep，T4 ~23 s/epoch |
| cuhk03/osnet_bstar | 49.3 | 52.4 | 72.5 | – | 下界；best 49.4 |
| cuhk03/osnet_dstar（logit KD） | 55.8 | 58.4 | – | **54%** | +6.5 vs B\* |
| cuhk03/osnet_fstar（DKD） | **60.9** | **63.9** | – | **96%** | +11.6 vs B\*；vs D\* **+5.1**；距 teacher 0.5 mAP，R1 高 1.5 |

- **Market 的結論在第二個資料集重現，而且更明顯**：排名 B\* < D\* < F\*；DKD 相對 logit KD 的增益 +5.1（Market +2.3）；補回率 D\* 54% → F\* 96%（Market 59% → 83%）。
- teacher 與學生之間的空間只有 12.1 mAP（Market 9.6），teacher 本身也較弱（CUHK03 每人只有 2 台相機、小資料集），補回率接近 100% 時要和絕對差距一起報。
- 尚缺：CUHK03 的 seeds（若論文主表要求 3 seeds）。

**傳統基準 R50 → R18（2026-10-13，Final，120 ep）**

| 實驗 | mAP | R1 | 補回率 | 備註 |
|---|---|---|---|---|
| cuhk03/r18_baseline | 50.6 | 54.9 | – | 下界；T4 ~25 s/epoch，其中資料載入 ~14 s（R18 在 CUHK03 上受 2 vCPU 限制） |
| cuhk03/r18_kd（logit + sim） | 60.5 | 63.4 | 92% | +9.9 vs baseline |
| cuhk03/r18_dkd | **62.4** | **65.1** | **109%** | vs R18-KD **+1.9**；**超過 teacher 1.0 mAP** |

- 與 Market 一致：DKD 在兩個學生上都勝過傳統 KD（R18 +1.9、OSNet +5.1；Market 為 +1.4、+2.3），而且 OSNet 的增益較大。
- R18-DKD 超過 teacher（62.4 vs 61.4）：CUHK03 小、teacher 較弱（只訓練 120 ep、61.4），學生在 teacher 軟標籤的正則化下反超。單一 seed，差距 1.0 接近雜訊等級，論文中寫「與 teacher 相當」較穩健。
- 補回率超過 100% 是因為分母（teacher − baseline = 10.8）小，論文要同時報絕對差距。

### 5c：跨域 zero-shot Market-1501 → CUHK03-NP（2026-10-14）
`tools/eval_matrix.py`，各 run 的 `last.pth`，flip TTA，無 re-ranking；輸出 `MyDrive/KD-ReID/eval_matrix_cuhk03.json`。同域數字與先前 Final 完全一致（評估流程無誤）。

| 模型（Market 訓練） | 同域 mAP | → CUHK03-NP mAP / R1 | 保留率 |
|---|---|---|---|
| R50 teacher | 86.6 | 5.7 / 5.2 | 6.6% |
| R18 baseline / KD / DKD | 80.6 / 84.8 / 86.2 | 4.4 / **8.1** / 7.6（R1 4.0 / 7.9 / 7.1） | 5.5 / 9.6 / 8.8% |
| OSNet B\* | 77.0 | 6.2 / 5.3 | 8.1% |
| OSNet D\*（logit KD） | 82.7 | 8.7 / 7.8 | 10.5% |
| **OSNet F\*（DKD，3 seeds）** | 85.0 | **10.2 ± 0.9** / 9.5（10.5 / 10.8 / 9.2） | **12.0%** |

- 絕對數字很低是正常的：CUHK03 只有 2 台相機、影像品質與 Market 差異大，Market → CUHK03 直接遷移在文獻中也只有個位數到十出頭的 mAP。看排名與保留率。
- **Market → Duke 的型態完整重現**（兩個獨立的目標域）：
  1. OSNet 上 B\* < D\* < F\*，KD 增益在跨域保留下來，DKD 再往上加。
  2. **F\* 跨域勝過 teacher**：10.2 ± 0.9 vs 5.7，差距是 seed 標準差的 5 倍，保留率 12.0% vs 6.6%（Duke：20.6 vs 19.7，這次差距更大）。
  3. **R18-DKD 跨域不比 R18-KD 好**（7.6 vs 8.1；Duke 17.9 vs 18.5）。DKD 的跨域優勢只在 OSNet 成立，兩個目標域一致。
- 無 KD 的 OSNet B\* 同域低 teacher 9.6 mAP，跨域卻略高（6.2 vs 5.7），顯示 OSNet 架構本身的泛化先驗也有貢獻；但 B\* → F\* 的 +4.0 遠大於這個差距，主要增益來自蒸餾。
- 只有 F\* 有 3 seeds，其餘為單一 seed（F\* 的 seed 範圍 1.6 mAP，單一 seed 的小差距例如 R18-KD vs R18-DKD 的 0.5 在雜訊內）。Market seeds（5b）完成後，teacher / B\* / D\* 的 s1、s2 要補跑這個評估。

### 5b：Market-1501 主表 3 seeds（2026-10-14）
seed 42 / 1 / 2；Final（最後一個 epoch），flip TTA，無 re-ranking。學生的 KD 一律用 seed 42 的 teacher（`teacher_r50/best.pth`，與先前所有 KD 實驗相同）；teacher 的 seed 只用來量上界的波動。

| 實驗 | seed 42 / 1 / 2（mAP） | **mAP（平均 ± 標準差）** | **R1** | 補回率 |
|---|---|---|---|---|
| teacher R50 | 86.6 / 86.4 / 86.6 | **86.5 ± 0.1** | 94.9 ± 0.2 | – |
| OSNet B\*（無 KD） | 77.0 / 77.4 / 77.8 | **77.4 ± 0.4** | 90.8 ± 0.4 | – |
| OSNet D\*（logit KD） | 82.7 / 82.6 / 82.5 | **82.6 ± 0.1** | 93.2 ± 0.2 | 57% |
| **OSNet F\*（DKD）** | 85.0 / 85.0 / 85.0 | **85.0 ± 0.0** | 94.0 ± 0.1 | **83%** |

- **F\* − D\* = +2.4 mAP**，是兩者 seed 標準差的 20 倍以上；所有 seed 的 F\* 都高於所有 seed 的 D\*（範圍完全不重疊）。B\* < D\* < F\* 的排名在每個 seed 都成立。
- 補回率（以 3 seeds 平均計）：D\* 57% → F\* 83%，與單 seed 的 59% → 83% 一致。差距空間 9.1 mAP。
- B\* 的波動最大（± 0.4），KD 讓學生變得更穩定（D\* ± 0.1、F\* ± 0.0）：teacher 的軟標籤降低了訓練的隨機性。
- 單 seed 時 B\* 為 77.0，3 seeds 平均 77.4，更接近 OSNet 論文的 77.8，表示我們的 B\* 配方確實是忠實的強 baseline。

### 5c 補完：跨域 Market → CUHK03-NP，3 seeds（2026-10-14）
seed 1、2 的評估輸出在 `MyDrive/KD-ReID/eval_matrix_cuhk03_seeds.json`；同域數字與各 run 的 Final 一致。flip TTA，無 re-ranking。

| 模型（Market 訓練） | → CUHK03-NP mAP，seed 42 / 1 / 2 | **平均 ± 標準差** | 保留率 |
|---|---|---|---|
| R50 teacher | 5.7 / 4.9 / 6.0 | **5.5 ± 0.6** | 6.4% |
| OSNet B\* | 6.2 / 6.1 / 6.0 | **6.1 ± 0.1** | 7.9% |
| OSNet D\*（logit KD） | 8.7 / 8.6 / 7.8 | **8.4 ± 0.5** | 10.1% |
| **OSNet F\*（DKD）** | 10.5 / 10.8 / 9.2 | **10.2 ± 0.9** | **12.0%** |

- 3 seeds 下 teacher < B\* < D\* < F\* 的排名成立：F\* 跨域幾乎是 teacher 的兩倍（10.2 vs 5.5）。最低的 F\* seed（9.2）高於任何 teacher seed（最高 6.0）與 B\* seed；F\* 與 D\* 的範圍只在邊緣接近（9.2 vs 8.7），平均差 +1.8。
- 跨域的 seed 波動（± 0.5–0.9）遠大於同域（± 0.0–0.4），單一 seed 的跨域數字只能看大致排名；這也是論文要報 3 seeds 的理由。
- R18 系列在 CUHK03 跨域只有單一 seed（R18-KD 8.1 vs R18-DKD 7.6，差距在雜訊內），結論維持「R18 上 DKD 沒有跨域優勢」，不寫成「更差」。

### 階段 4 前置：CLIP-ReID teacher 檢查（2026-10-14）
官方 ViT-CLIP-ReID Market checkpoint（無 SIE / OLP），`tools/check_teacher.py`。

- **檢索與官方一致**：flip 89.9 / R1 95.4，無 flip 89.7 / 95.4（官方 log 約 89.8 / 95.7，無 re-ranking）。權重、正規化、特徵皆正確。
- **類別順序是對的，但 ID 分類頭幾乎是平的**（乾淨訓練影像、整個訓練集 12,936 張）：

| teacher / head | top-1 | top-5 | 平均最大機率（T=1） |
|---|---|---|---|
| CLIP-ReID mean（KD 預設） | 78.6% | 95.1% | **0.008** |
| CLIP-ReID cls（768 維） | 77.6% | 94.9% | 0.012 |
| CLIP-ReID proj（512 維） | 79.5% | 95.4% | 0.006 |
| R50 teacher（參考） | 100.0% | 100.0% | 0.952 |

  - 751 類的均勻分布最大機率是 0.0013，CLIP-ReID 的 0.008 只比均勻高一點。原因：CLIP-ReID 第二階段的 ID 學習主要靠文字原型（i2t loss），線性分類頭在 5e-6 的學習率下只附帶訓練。
  - 加了資料增強後 top-1 再降到 ~64%（第一次檢查的數字）。
  - top-5 95% 且順序與我們的編號一致（ID 編號規則已實測相同），所以**不是順序錯開**；「重新對應 top-1」91% 比 top-1 高，是因為多個類別被預測成少數幾個「熱門」類別，不是排列錯位。
- **後果**：直接用這組 logits 做 KD（T=4），teacher 的軟標籤幾乎是均勻分布，學生幾乎學不到東西；DKD 的 TCKD 項更會把學生的目標類機率往 teacher 的 ~1% 拉，和 CE 互相衝突。
- **處理（commit d9cb380）**：teacher 新增 `logit_scale`（temperature scaling, Guo et al. 2017），在 logit KD / DKD 之前把 teacher logits 乘上一個常數；排名與 top-1 不變，只修正銳利度。數值由 `check_teacher.py` 在訓練集上以 NLL 最小化擬合，寫進 config 固定。另加 **4c**：只用 CLIP 特徵的 in-batch 相似度分布（simdist）蒸餾，完全不用分類頭。
- **擬合結果（2026-10-14）**：logits 每張影像的標準差只有 0.23；temperature scaling 擬合出 **logit_scale = 8.26**。縮放後 T=1 最大機率 0.008 → 0.809、p(true) 0.713；KD 實際看到的 T=4 軟標籤最大機率 0.0021 → 0.063。已寫入 `configs/clip/_teacher_clip.yaml`（4a / 4b 使用；4c 不用 logits）。
- **成功標準（訓練前先訂）**：CLIP teacher 對問題 2 算成功，若 (1) Market 同域至少一組 ≥ 84.5（F\* − 0.5），且 (2) 跨域 Market → CUHK03-NP 明顯高於 F\* 的 10.2 ± 0.9，或 (3) 使用者系統的跨相機 / Panorama 優於 R50 版 F\*。(2)、(3) 是主要目的；只贏 (1) 不算。

### 階段 4：CLIP-ReID teacher → OSNet x0.25（2026-10-14）
B\* 配方、T=4、seed 42，teacher 為官方 CLIP-ReID（logit_scale 8.26）。Final（epoch 240），flip TTA，無 re-ranking。
**有效性確認**：config.yaml 每次啟動都會被重寫，不能當證據；改看 log：4a 第一次啟動 03:47:53 UTC，晚於 logit_scale commit（031d5d3，03:42:38 UTC）；4b 接在 4a 後（07:01 UTC）。ep 1 的 kd_dkd = 22.2、kd_logit = 4.08，遠大於未縮放的平坦 logits 會給的值，確認縮放有生效。

| 學生 | teacher / KD | Market mAP / R1 | → CUHK03-NP mAP / R1 |
|---|---|---|---|
| B\*（3 seeds） | – | 77.4 ± 0.4 / 90.8 | 6.1 ± 0.1 |
| D\*（3 seeds） | R50 / logit | 82.6 ± 0.1 / 93.2 | 8.4 ± 0.5 |
| F\*（3 seeds） | R50 / DKD | **85.0 ± 0.0** / 94.0 | 10.2 ± 0.9 |
| **4a** | CLIP / DKD | 83.4 / 92.8 | **11.3** / 11.2 |
| **4b** | CLIP / logit | 83.2 / 93.1 | 10.1 / 9.9 |
| **4c** | CLIP / simdist（無 logits） | 78.9 / 91.7 | 待評估 |

- **更強的 teacher 沒有帶來更強的同域學生**：CLIP-ReID 比 R50 高 3.4 mAP（89.9 vs 86.5），但 DKD 學生反而低 1.6（83.4 vs 85.0）。logit KD 則略好（83.2 vs 82.6）。
- **跨域方向一致但幅度不足**：4a 11.3 vs F\* 10.2 ± 0.9（+1.1，約 1.3 個標準差，單 seed）；4b 10.1 vs D\* 8.4 ± 0.5（+1.7，約 3.4 個標準差）。換成 CLIP teacher 後 logit KD 的跨域增益比較明確。
- **只用 CLIP 特徵關係（4c）很弱**：78.9，只比 B\* 高 1.5，遠低於任何 logit 類 KD。CLIP 的優勢沒有透過 in-batch 相似度分布傳過去；和階段 2 的 similarity 無貢獻一致。
- **對照事前標準**：(1) 同域 ≥ 84.5：**未達成**（最佳 83.4）。(2) 跨域「明顯」> 10.2：4a 方向正確但未達顯著，需要 seeds。(3) 使用者系統：未測。
- 可能原因（待驗證）：CLIP-ReID 的 ID 頭本身只有 78.6% 訓練集 top-1，即使縮放後，軟標籤品質仍不如 R50 的頭（100%）；CLIP 的知識主要在特徵，而 DKD / logit KD 只吃 logits。
- **使用者系統測試（2026-10-15，TTA OFF、re-ranking OFF；輔助證據，不進主表）**：

| 場景（query / gallery） | F\*（R50 → DKD）mAP / R1 | 4a（CLIP → DKD）mAP / R1 | 差距 mAP |
|---|---|---|---|
| C01 → C02（129 / 124） | 84.90 / 93.80 | 84.78 / 92.25 | −0.12 |
| C02 → C01（124 / 129） | 89.44 / 96.77 | 87.77 / 98.39 | −1.67 |
| Panorama（210 / 713） | 68.68 / 89.52 | 68.28 / 87.62 | −0.40 |

  - 判讀：**打平**。三個差距都在小樣本雜訊內（每個場景粗估 mAP 標準誤約 2 pp），R1 差異只有 1–2 個 query。CLIP-ReID 本身在這些場景比 R50 teacher 高 7.9–16.3 mAP，但這個優勢沒有傳到學生身上。對照事前標準 (3)：**未達成**。
  - 兩種解釋待區分：(a) 容量：0.2M 的學生裝不下；(b) 傳遞管道：DKD / logit KD 只傳 logits，而 CLIP-ReID 的優勢在特徵、它的 ID 頭偏弱（訓練集 top-1 78.6%）。區分實驗：OSNet x1.0 學生分別用 R50 / CLIP teacher。
  - 待補：F\* s1 / s2 在使用者系統的波動（只需打包），4a seed 1。
- **4a seed 1（2026-10-15）**：Final 83.5 / R1 92.3（seed 42：83.4 / 92.8）→ **4a = 83.45 ± 0.07**，確定低於 F\* 85.0 ± 0.0（差 1.55，約 20 個標準差）。CLIP teacher 讓 x0.25 學生在同域變差是穩定的現象，不是 seed 雜訊。
- **下一步（plan 6，`configs/osnet_x1/`）**：OSNet x1.0 學生、相同 B\* 配方，6b（R50 / DKD）vs 6c（CLIP / DKD），6a 無 KD。6c > 6b 表示是容量問題；6c ≤ 6b 表示是傳遞管道問題。

### 6b：OSNet x1.0 ← R50，DKD（2026-10-15）
B\* 配方（lr 1e-3、240 ep），teacher 為 `teacher_r50/best.pth`（與 F\* 相同）。中途斷線一次，從 epoch 97 續跑。
- **Final（epoch 240）mAP 88.3 / R1 95.3**（best 88.5）。T4 約 48 s/epoch。
- **學生超過 teacher**：88.3 vs R50 86.5 ± 0.1（+1.8）。x0.25 的 F\* 是 85.0（低 teacher 1.5）。參考值：OSNet x1.0 原論文 Market（無 RK）84.9 / 94.8，6a 跑完才有我們自己的 x1.0 baseline。
- 意義：x1.0 的容量足以超越 R50 teacher，所以 6c（CLIP teacher）若仍 ≤ 88.3，就不能用「學生裝不下」來解釋 CLIP 沒有幫助。

### 6c：OSNet x1.0 ← CLIP-ReID，DKD（2026-10-16）— 容量假說成立
與 4a 相同的 teacher 區塊（logit_scale 8.26），學生換成 x1.0，B\* 配方。完整跑完 240 epoch（log 從 15:02 UTC 啟動到 19:25 UTC，約 4.4 h，T4 約 66 s/epoch）。
- **Final（epoch 240）mAP 89.5 / R1 95.4**（best 89.5）。

| 學生 | R50 teacher（86.5）DKD | CLIP-ReID teacher（89.9）DKD | CLIP − R50 |
|---|---|---|---|
| OSNet x0.25（0.2M） | 85.0 ± 0.0（F\*，3 seeds） | 83.45 ± 0.07（4a，2 seeds） | **−1.55** |
| OSNet x1.0（2.2M） | 88.3（6b） | **89.5**（6c） | **+1.2** |

- **符號翻轉**：同一個 CLIP teacher、同樣的 DKD，對 x0.25 是負面的，對 x1.0 是正面的。這支持「容量」解釋：x0.25 裝不下 CLIP 額外的知識，x1.0 可以。之前的另一個解釋（只傳 logits 送不過去）被否定：logits 管道對夠大的學生是有效的。
- **6c 幾乎追平 CLIP-ReID 本身**：89.5 vs 89.9（−0.4），參數 2.2M vs 約 86M（~1/40）。6b 也超過了自己的 R50 teacher（88.3 vs 86.5）。
- 注意：x1.0 兩組都是單一 seed。x0.25 DKD 的 seed 標準差只有 0.0–0.1，+1.2 很可能是真實的，但論文前要補 6b / 6c 的 seed。
- 待辦：6b / 6c 跨域評估（Market → CUHK03-NP）、打包 6c 到使用者系統（問題 2 的真正標準）、6a（x1.0 無 KD）baseline、x1.0 的部署成本（延遲）。

### 使用者系統：OSNet x1.0（6b / 6c）與速度（2026-10-17；輔助證據，不進主表）
TTA OFF、re-ranking OFF、與先前相同的 query / gallery；舊結果全部重現。兩個套件 PyTorch / ONNX 自檢通過（最大誤差 < 3e-6）。數字由使用者回報，完整報告在使用者機器 `reid_project/experiments/kdreid_x1_eval_20261016/`。

| 場景（query / gallery） | CLIP-ReID | R50 teacher | F\* x0.25（3 seeds） | 6b x1.0 ← R50 | **6c x1.0 ← CLIP** |
|---|---|---|---|---|---|
| C01 → C02（129 / 124） | 97.85 / 100 | 81.56 / 91.47 | 85.05 ± 0.13 | 85.55 / 90.70 | **90.91** / 95.35 |
| C02 → C01（124 / 129） | 97.65 / 100 | 84.81 / 97.58 | 89.19 ± 0.21 | 91.64 / 98.39 | **93.35** / 97.58 |
| Panorama（210 / 713） | 78.01 / 91.43 | 70.10 / 89.05 | 68.92 ± 0.51 | 70.00 / 87.14 | **71.73** / 90.00 |

- **F\* 在實際場景的 seed 波動很小**（mAP 標準差 0.13 / 0.21 / 0.51），可以當雜訊基準。
- **6c 明顯勝出**：比 6b 高 5.36 / 1.71 / 1.73 mAP，比 F\* 平均高 5.86 / 4.16 / 2.81，都遠大於 F\* 的 seed 波動。和 CLIP-ReID 的差距縮小 46% / 48% / 33%，剩 6.94 / 4.30 / 6.28。Panorama 首次超過 R50 teacher（71.73 vs 70.10）。
- **6b 只小幅勝過 F\***（+0.5 / +2.45 / +1.08）：x1.0 的增益主要來自 CLIP teacher，不只是學生變大。
- 回頭看 4a（x0.25 ← CLIP）：相對 F\* 平均是 −0.27 / −1.42 / −0.64，其中 C02 → C01 的 −1.42 約為 7 個標準差，**4a 在實際場景其實是略差**，不是打平（先前用單一 seed 的 F\* 判讀為打平）。和 Market 的結論一致：x0.25 裝不下 CLIP。
- **對照問題 2 事前標準**：6c 達成 (1) Market 89.5 ≥ 84.5、(3) 使用者系統三個場景都優於 R50 版 F\*。(2) 跨域待測。**問題 2 以 x1.0 學生達成**；限制：6b / 6c 各只有一個 seed。

**速度**（Ryzen 7 4800H / RTX 3050 Laptop 4GB，接電，ONNX，三輪平均，含前處理與特徵回傳；ReID crop 吞吐量，不是整段影片 FPS）：

| 執行方式 | batch | F\* x0.25 ms／張 | 6c x1.0 ms／張 | CLIP-ReID ms／張 |
|---|---|---|---|---|
| CPU | 1 | 7.96 | 18.48 | – |
| CPU | 32 | 5.02 | 21.37 | – |
| CUDA | 1 | 12.88 | 15.92（另一輪 14.69） | 20.70 |
| CUDA | 32 | 2.03 | **3.44**（另一輪 3.64） | 10.98 |

- 6c 在 GPU batch 32 約 275–290 張/秒，比 CLIP-ReID 快約 3 倍（batch 1 約 1.4 倍）；ONNX CUDA 比 PyTorch GPU 快，部署採 ONNX CUDA。
- CPU 上 6c 是 F\* 的 2.3–4.3 倍成本：純 CPU 或邊緣裝置仍以 F\* 為主。
- 尚未量測：整個管線（YOLO + 追蹤 + ReID 共用 GPU）的實際 FPS。

### Plan 7：新方法（程式就緒，2026-10-17，commit 418bc92）
- **7a 同家族、比原 teacher 更強的助教**：CLIP-ReID → OSNet x1.0（6c，89.5）→ OSNet x0.25，DKD。階段 3 的助教（R18-KD 84.8）比 R50 弱、架構也不同而失敗；6c 比 R50 強且與學生同為 OSNet。對照：F\* 85.0 ± 0.0、4a 83.45 ± 0.07。
- **7b / 7c 文字原型蒸餾**：CLIP-ReID 的身份知識是透過每個身份學出的文字 prompt（第二階段 i2t loss）訓練的，線性 ID 頭只是附帶訓練、幾乎是平的（訓練集 top-1 78.6%）。`logit_head: text` 載入官方 checkpoint 的 prompt learner 與 CLIP 文字編碼器，算出 751 個文字原型，teacher logits = 投影後 CLS 特徵 · 原型（即 CLIP-ReID 訓練時的 i2t logits，未正規化點積）。
  - 實作驗證（CPU、合成的官方格式 checkpoint）：原型與獨立計算一致、文字編碼器是因果的（EOT 之後的 token 不影響結果）、改一個身份的 context 只改變該身份的原型、端到端 DKD 可訓練。
  - **未在真權重上驗證的假設**：end-of-text 位置。依官方模板 "A photo of a X X X X person." 推得 11（SOT a photo of a | X×4 | person . EOT）；`check_teacher` 會掃描 8–14，正確位置的 top-1 應明顯最高。
  - 事前預期：原型 top-1 應明顯高於 78.6%；7b 若 > 4a（83.45）表示 x0.25 的失敗部分來自 teacher 訊號品質而不只是容量；7c 若 > 6c（89.5）則是 x1.0 的新最佳。
  - **真權重驗證（2026-10-17，`check_teacher --config configs/clip/teacher_eval_text.yaml`）**：
    | head | top-1 乾淨訓練集 | top-1 增強 | 平均最大機率 |
    |---|---|---|---|
    | mean / cls / proj（線性頭） | 78.6 / 77.6 / 79.5% | 63.6 / 61.7 / 65.4% | 0.006–0.012 |
    | **text（原型）** | **100.0%** | **99.8%** | **0.871** |
    - EOT 掃描：8→34.5%、9→73.6%、10→95.4%、**11→100.0%**、12→93.4%、13→82.2%、14→63.3%：推得的 11 正確，假設成立。
    - 檢索 89.9 / 95.4（不受影響，原型只改 logits）。
    - 溫度縮放擬合出 27.79，但這是**退化解**：訓練集全部分對時，NLL 隨 scale 單調下降，值只是最佳化停下的位置，會把 teacher 變成 one-hot（T=4 仍是 1.0000）。原始 logits 的最大機率 0.871 已和一般 CE 訓練的 teacher 一樣自信，所以 **`logit_scale: 1.0`**（不縮放）。`check_teacher` 已加上退化偵測（fit set top-1 ≥ 99% 時警告）。
    - 註：原型在訓練集上 100% 是因為 CLIP-ReID 第一階段就是用訓練影像學這些 prompt；這和任何 teacher 在自己訓練集上接近 100% 相同，不是問題。

#### Plan 7 結果（2026-10-17~18，commit 9157797，各 1 seed，Market-1501）：三個都是負結果
| 實驗 | student ← teacher | Final mAP / R1 | 對照 | 差 |
|---|---|---|---|---|
| 7a | x0.25 ← 6c（OSNet x1.0，89.5）DKD | 83.6 / 92.3 | F\* 85.0 ± 0.0 / 4a 83.45 ± 0.07 | −1.4 / +0.15 |
| 7b | x0.25 ← CLIP-ReID 文字原型 DKD | 83.8 / 93.1 | F\* 85.0 ± 0.0 / 4a 83.45 ± 0.07 | −1.2 / +0.35 |
| 7c | x1.0 ← CLIP-ReID 文字原型 DKD | 88.4 / 94.6 | 6c 89.5 ± 0.0 / 6b 88.3 | **−1.1** / +0.1 |

- 7a：同家族、比 R50 強的助教（6c 89.5）一樣救不了 x0.25，結果和直接用 CLIP（4a）相同（83.6 vs 83.45）。x0.25 從 CLIP 系知識（不論直接或經 6c 轉手）最多到約 83.5，從 R50 知識能到 85.0 → **x0.25 的問題不是「teacher 和學生架構差太多」，而是 CLIP 系的 soft label 對它不合用**。階段 3 的結論（助教無效）再次成立。
- 7b：比 4a 高 0.35（事前標準「7b > 4a」字面上達成，但只是單 seed、幅度小），仍低 F\* 1.2。文字原型改善了訊號品質，但不足以翻轉 x0.25 的結論。
- 7c：比 6c 低 1.1（6c 兩個 seed 都是 89.5，差距遠大於波動）→ 文字原型雖然在訓練集 top-1 100%，當 teacher 卻比幾乎平的線性頭差。**teacher 在訓練標籤上的準確度不等於蒸餾品質**。
- 觀察（不是證明）：最後一個 epoch 的學生訓練 CE。x1.0：6c 1.58 / 1.57（兩 seed）> 7c 1.14 ≈ 6b 1.06；mAP 6c 89.5 > 7c 88.4 ≈ 6b 88.3。文字原型 teacher 很銳利（最大機率 0.871），學生被帶得更貼合訓練標籤，結果和 R50 teacher 幾乎一樣；線性頭 ×8.26 的 teacher 比較軟，學生的 CE 降不下去，mAP 反而最高。x0.25 沒有同樣的乾淨型態（7b CE 1.27、4a / F\* 不同），所以只當成 x1.0 的觀察。
- 結論：**Plan 7 結束，三個方法都不採用**。主表維持 F\*（x0.25）與 6c（x1.0）。可以寫進論文的負結果：(1) 助教（不論同家族或更強）無法讓 x0.25 吸收 CLIP 知識；(2) CLIP-ReID 自己的文字原型分類器（訓練集 100%）作為 teacher 比它附帶的線性頭差。
- 沒做的診斷（不建議花單位）：調 `logit_scale` 讓文字原型變軟（會變成在測試集上調參）。

### 跨域 Market → CUHK03-NP：OSNet x1.0（2026-10-17）
`eval_matrix_x1.json`，last.pth，flip TTA，無 re-ranking；同域數字與 Final 一致（6b 88.3、6c 89.5）。

| 模型（Market 訓練） | 同域 mAP | → CUHK03-NP mAP / R1 | 保留率 |
|---|---|---|---|
| R50 teacher（3 seeds） | 86.5 | 5.5 ± 0.6 | 6.4% |
| OSNet x0.25 F\*（R50 / DKD，3 seeds） | 85.0 | 10.2 ± 0.9 | 12.0% |
| OSNet x1.0 6b（R50 / DKD） | 88.3 | 9.4 / 8.9 | 10.6% |
| **OSNet x1.0 6c（CLIP / DKD）** | 89.5 | **16.7 / 17.1** | **18.7%** |

- **CLIP teacher 帶來的泛化傳到 x1.0 學生上了**：6c 16.7，比 6b 高 7.3、比 F\* 高 6.5（約 7 個 F\* seed 標準差），是 R50 teacher 的 3 倍。
- **學生變大本身不會讓跨域變好**：6b（x1.0 ← R50）9.4 甚至略低於 F\*（x0.25 ← R50）10.2，同域卻高 3.3。跨域增益幾乎全部來自 CLIP teacher。
- 對照 x0.25：4a（x0.25 ← CLIP）跨域 11.3，只比 F\* 高 1.1；x1.0 上同一個 teacher 帶來 +7.3。和同域、使用者系統的結論一致：x0.25 裝不下 CLIP 的知識。
- **問題 2 事前標準全部達成（6c，單 seed）**：(1) Market 89.5 ≥ 84.5；(2) 跨域 16.7 明顯 > 10.2 ± 0.9；(3) 使用者系統三個場景都優於 F\*。
- 尚缺參考值：CLIP-ReID teacher 本身的 Market → CUHK03-NP zero-shot（用來算學生保留了 teacher 多少跨域能力）。

### 使用者系統：整個管線 FPS（2026-10-17；輔助證據，不進論文）
同一部 89 秒影片（2,661 幀），YOLOv5su + ByteTrack + ReID，RTX 3050 Laptop 4GB，接電，各兩輪；正式 pipeline 未修改，身份門檻沿用原值。

| ReID 後端 | 平均 FPS | 相對 CLIP | 顯存峰值 | YOLO＋接縫 ms/幀 | ByteTrack ms/幀 | ReID ms/幀 |
|---|---|---|---|---|---|---|
| CLIP-ReID（PyTorch CUDA） | 9.37 | 1.00× | 1,493 MiB | 61.70 | 1.87 | 20.81 |
| 6c x1.0（ONNX CUDA） | 9.72 | 1.04× | 1,584 MiB | 58.03 | 1.81 | 21.57 |
| F\* x0.25（ONNX CUDA） | 11.07 | 1.18× | 1,200 MiB | 55.31 | 1.70 | 12.95 |

- **管線瓶頸是 YOLO，不是 ReID**：YOLO 約 55–62 ms/幀，佔 70% 以上；ReID 即使降到 0，也只能到約 16–17 FPS。
- **6c 在管線裡沒有變快**：單張 crop 吞吐量快 3 倍（batch 32），但每幀的 ReID 時間和 CLIP 差不多。可能原因：每幀人數少、batch 小（batch 1 時只快 1.4 倍）；前處理 / 搬運等固定成本佔大頭；6c 的身份決策不同，處理了更多 crop（使用者回報）。兩輪之間差約 1 FPS，4% 的差距在波動內。
- **部署判讀**：在這條 GPU 管線（YOLO 主導、人數少）上，換成 6c 幾乎不增加 FPS，卻比 CLIP-ReID 低 4–7 mAP。6c / F\* 的速度優勢要在 ReID 負載大的情境才會顯現：純 CPU / 邊緣裝置、人多的畫面、單張 GPU 跑多路影片、顯存受限。管線中的辨識準確率尚未驗證。
- 論文的速度數字應報單張 crop 的推論成本（參數、FLOPs、固定 batch 的延遲），不報整個管線 FPS（受偵測器與場景主導）。

- **6c seed 1（2026-10-17）**：Final 89.5 / R1 95.2，和 seed 42（89.5 / 95.4）相同 → **6c = 89.5 ± 0.0（2 seeds）**。x1.0 ← CLIP 的同域結果和 x0.25 F\* 一樣穩定；vs 6b 的 +1.2 遠大於這個波動（6b 仍是單 seed）。

### 6a 與 6b seed 1：OSNet x1.0 的 2×2 補齊（2026-10-18）
- **6a（x1.0 無 KD，B\* 配方）Final 85.8 / 94.9**：比 OSNet 原論文 x1.0（84.9 / 94.8，無 RK）高 0.9，和 x0.25 的 B\* 一樣是忠實的強 baseline。
- **6b seed 1 Final 88.3 / 95.0**，和 seed 42（88.3 / 95.3）相同 → **6b = 88.3 ± 0.0（2 seeds）**。6c − 6b = +1.2 現在兩邊都有 2 個 seed，標準差都是 0.0。

| 學生 | 無 KD | ← R50（86.5）DKD | ← CLIP-ReID（89.9）DKD | CLIP − R50 |
|---|---|---|---|---|
| OSNet x0.25（0.2M） | 77.4 ± 0.4 | 85.0 ± 0.0（+7.6） | 83.45 ± 0.07（+6.05） | −1.55 |
| OSNet x1.0（2.2M） | 85.8 | 88.3 ± 0.0（+2.5） | 89.5 ± 0.0（+3.7） | **+1.2** |

- **R50 teacher 幾乎沒有比 x1.0 baseline 強**（86.5 vs 85.8，只差 0.7），DKD 卻讓學生 +2.5，比 teacher 高 1.8。在這裡傳統的補回率沒有意義（357%），論文對 x1.0 應報「相對 baseline 的增益」與「是否超過 teacher」。這和 born-again network 的現象一致：teacher 的軟標籤即使來自不比學生強的模型，也有正則化效果。
- **以 CLIP teacher 計的補回率**：x1.0 (89.5 − 85.8) ÷ (89.9 − 85.8) = **90%**；x0.25 (83.45 − 77.4) ÷ (89.9 − 77.4) = **48%**（R50 teacher 是 83%）。容量的說法有了更清楚的數字：同一個 CLIP teacher，大學生補回 90%，小學生只有 48%。
- 6b_s1 / 6c_s1 / 6a 的跨域評估還沒做（只需評估）。

### 跨域參考：CLIP-ReID 本身與 x1.0 補齊（2026-10-18，`eval_matrix_x1_ref.json`）
`eval_matrix --teacher-configs configs/clip/teacher_eval.yaml`（官方 Market checkpoint，不經訓練）＋ 6a / 6b_s1 / 6c_s1 的 last.pth；flip TTA，無 re-ranking。同域數字與各自 Final 一致（CLIP 89.9、6a 85.8、6b_s1 88.3、6c_s1 89.5）。

| 模型（Market 訓練） | 同域 mAP | → CUHK03-NP mAP / R1 | n | 保留率 |
|---|---|---|---|---|
| **CLIP-ReID teacher（官方權重）** | 89.9 | **39.2 / 41.0** | – | **43.6%** |
| R50 teacher | 86.5 | 5.5 ± 0.6 | 3 | 6.4% |
| OSNet x1.0 無 KD（6a） | 85.8 | 6.7 / 6.6 | 1 | 7.8% |
| OSNet x1.0 ← R50 DKD（6b） | 88.3 | 8.8 ± 0.8（9.4 / 8.2） | 2 | 10.0% |
| **OSNet x1.0 ← CLIP DKD（6c）** | 89.5 | **17.1 ± 0.6**（16.7 / 17.5） | 2 | 19.1% |
| OSNet x0.25 F\*（R50 DKD） | 85.0 | 10.2 ± 0.9 | 3 | 12.0% |

- **CLIP-ReID 的跨域能力遠高於所有學生**：39.2，是 6c 的 2.3 倍、R50 teacher 的 7 倍。來源很可能是 CLIP 在大量網路影像上的預訓練（Market 微調後仍保留），不是 ReID 方法本身。
- **蒸餾在同域幾乎完整傳遞，跨域只傳了一部分**：6c 同域達 CLIP 的 99.6%（89.5 / 89.9），跨域只有 44%（17.1 / 39.2）。這是學生和 CLIP 之間剩下最大的差距。
- 原本的「蒸餾學生跨域勝過 teacher」只對 R50 teacher 成立；論文要分開寫：(1) 對 R50，學生跨域比 teacher 好（10.2 vs 5.5）；(2) 對 CLIP，學生拿到 teacher 跨域能力的 44%，比 R50 學生多一倍（17.1 vs 8.8），但遠不到 teacher 本身。
- x1.0 跨域：6a 6.7 < 6b 8.8 < 6c 17.1。KD（R50）+2.1，換 CLIP teacher 再 +8.3；x1.0 無 KD 的 6.7 和 x0.25 B\* 的 6.1 差不多 → 學生變大本身幾乎不帶來跨域能力。
- 推論（待驗證）：只用 Market 影像蒸餾，學生只學到 teacher 在 Market 分布上的行為；CLIP 的泛化在「沒看過的分布」上才顯現。在目標域（或其他行人影像）的**無標註**影像上做特徵蒸餾，可能補回這段差距——這是方法 4（無標註目標域蒸餾）的直接動機。
- 部署 benchmark 第一次在 GPU 上跑時出錯：fp16 計時把模型就地轉成 half，下一個 fp32 計時就 dtype 不符；CLIP 的 LayerNorm 也不支援 half 權重。兩者已修正並加測試，需要重跑。

### 部署成本：固定硬體 benchmark（2026-10-18，Colab T4 + 2 vCPU，`benchmark_t4.json`）
fp16 bug 修正後（commit 8e95b31）一次跑完五個模型，隨機權重（只量大小與速度）。完整表見 `docs/paper_tables.md` 表 6。

| 模型 | 參數 | GMACs | ORT CPU b1 (ms) | GPU fp16 b32 (ms/張) | Market mAP |
|---|---|---|---|---|---|
| CLIP-ReID ViT-B/16 | 86.1M | 11.08 | 255.1 | 2.30 | 89.9 |
| ResNet-50 | 23.5M | 4.05 | 81.1 | 1.03 | 86.5 |
| ResNet-18 | 11.2M | 1.99 | 40.1 | 0.42 | 86.2 |
| OSNet x1.0 | 2.2M | 0.98 | 22.2 | 0.83 | 89.5 |
| OSNet x0.25 | 0.2M | 0.082 | 3.2 | 0.54 | 85.0 |

- CPU：x1.0 比 CLIP-ReID 快 11.5 倍（精度 −0.4），x0.25 快 80 倍（比 R50 快 25 倍，精度 −1.5）。
- GPU（PyTorch eager）：OSNet 受 kernel 啟動成本限制（x0.25 batch 1 與 32 都約 16–17 ms），batch 1 比 R50 還慢；batch 32 fp16 時 x1.0 比 CLIP-ReID 快 2.8 倍，和使用者 RTX 3050 的 3.2 倍一致。OSNet 的部署優勢在 CPU / 邊緣裝置。
- 參數、MACs 和 OSNet 原論文一致；ONNX 數值差 ≤ 6e-5。

### 5a 補完：CUHK03-NP 3 seeds 與 OSNet x1.0（2026-10-18~19，commit 0486890）
同一個 Colab cell 依序跑完 8 個 run（約 13 h，全部 exit 0）。KD 一律用 CUHK03 teacher 的 `last.pth`（log 已確認）。Final，flip TTA，無 re-ranking。

| 實驗 | seed 42 / 1 / 2（mAP） | **mAP** | **R1** | 補回率 |
|---|---|---|---|---|
| R50 teacher | 61.4 | 61.4 | 62.4 | – |
| OSNet x0.25 B\* | 49.3 / 49.8 / 49.3 | **49.5 ± 0.3** | 52.3 ± 0.4 | – |
| OSNet x0.25 D\*（logit KD） | 55.8 / 55.5 / 55.5 | **55.6 ± 0.2** | 58.2 ± 0.3 | 51% |
| **OSNet x0.25 F\*（DKD）** | 60.9 / 60.8 / 61.4 | **61.0 ± 0.3** | 63.6 ± 0.5 | **97%** |
| OSNet x1.0 無 KD | 67.0 | 67.0 | 70.3 | – |
| OSNet x1.0 ← R50 DKD | 69.2 | 69.2 | 71.4 | – |

- **CUHK03 的主表結論在 3 seeds 下成立**：F\* − D\* = +5.4（Market +2.4），D\* − B\* = +6.1；每個 seed 都是 B\* < D\* < F\*，範圍不重疊。F\* 與 teacher 相當（61.0 vs 61.4）。單 seed 時的數字（49.3 / 55.8 / 60.9）幾乎不變。
- **x1.0 在 CUHK03 上比 R50 teacher 強很多**（67.0 vs 61.4），DKD 仍 +2.2。和 Market（6a 85.8 → 6b 88.3，teacher 86.5）同型態，這裡 teacher 更弱。x1.0 baseline 和 OSNet 原論文 CUHK03 detected 67.8 一致。
- CLIP-ReID 官方沒有 CUHK03 權重（只有 MSMT17 / Market / Duke / Occ-Duke / VeRi / VehicleID），x1.0 ← CLIP 的第二個資料集改在 MSMT17 做。

### Plan 8b：無標註 CUHK03 影像上的蒸餾（UDA，2026-10-19，Kaggle T4，commit 7eb56f6）
x1.0 ← CLIP-ReID（6c 配方）＋每個 iteration 32 張 CUHK03-NP 訓練集影像（不用標籤），在上面做 cosine + simdist + logit KD。Kaggle T4 約 90 s/epoch，共 6.3 h；log 確認 `unlabeled: 7365 images from cuhk03np train split`，所有 `u_kd_*` 有限並穩定下降。Final，flip TTA，無 re-ranking。

| 模型 | Market mAP / R1 | → CUHK03-NP mAP / R1 | 設定 |
|---|---|---|---|
| CLIP-ReID teacher | 89.9 / 95.4 | 39.2 / 41.0 | zero-shot |
| 6c（x1.0 ← CLIP DKD，2 seeds） | 89.5 ± 0.0 / 95.3 | 17.1 ± 0.6 / 17.6 | zero-shot |
| **8b（6c + 無標註 CUHK03 影像）** | **88.7 / 95.1** | **31.7 / 33.1** | **UDA** |

- **目標域大幅提升**：CUHK03 從 17.1 → 31.7（+14.6，約 24 個 seed 標準差），補回學生與 CLIP 之間跨域差距的 **66%**（事前門檻 25%，即 22.6）。學生達到 CLIP 本身跨域能力的 81%，參數約 1/40。
- **來源域付出代價**：Market 88.7，比 6c 低 0.8（6c 兩個 seed 都是 89.5）。事前判斷標準要求下降 ≤ 0.3，**這一項沒有達成**。照事前寫的規則，8b 是「目標域成功、來源域未達標」的部分成功，不是完全成功。
- 假說得到支持：學生在 Market 影像上已經學到 CLIP 的同域行為，但只有在目標域影像上被要求模仿 CLIP，CLIP 的泛化才傳得過來。
- 還不能下結論的部分（等 8a）：(1) 跨域增益有多少來自「目標域影像」、多少來自「多了 3 個損失項」；(2) Market −0.8 是新損失本身造成的（8a 也會掉），還是混入另一個域的影像造成的（8a 不會掉）。
- 注意：這是 UDA（用了 CUHK03 訓練集影像，沒有用標籤），論文中要和 zero-shot 欄位分開。不在測試集上調 `unlabeled.weight`；若要處理來源域的代價，要事先寫好規則（例如只看 Market 的訓練集 / 驗證切分），不能看 CUHK03 測試集來挑。

### Plan 8a（對照）與 plan 8 結論（2026-10-20）
8a = 6c + 每個 iteration 32 張 **Market** 訓練集影像（不用標籤），損失、batch、運算量都和 8b 相同，只差影像的分布。Colab 跑到 epoch 215 斷線，2026-10-20 從 last.pth 續跑完（log：`Resumed ... at epoch 215`）。Final，flip TTA，無 re-ranking。

| 模型 | 額外的無標註影像 | Market mAP / R1 | → CUHK03-NP mAP / R1 |
|---|---|---|---|
| 6c（2 seeds） | 無 | 89.5 ± 0.0 / 95.3 | 17.1 ± 0.6 / 17.6 |
| **8a（對照）** | Market | **89.5 / 94.9** | **15.3 / 14.5** |
| 8b（UDA） | CUHK03-NP 訓練集 | 88.7 / 95.1 | 31.7 / 33.1 |

- **跨域增益完全來自目標域影像，不是新的損失**：同樣的三個損失、同樣多的影像，換成 Market 影像（8a）跨域沒有變好，甚至略低（15.3 vs 17.1 ± 0.6，單 seed）；換成 CUHK03 影像（8b）就是 31.7。新的損失本身沒有幫助，有用的是「在目標域影像上模仿 CLIP」。
- **Market 的 −0.8 也來自目標域影像，不是新的損失**：8a 的 Market 和 6c 完全一樣（89.5）。8b 的下降來自混入另一個域的影像，可能原因：(1) BN 的 running statistics 被兩個域的影像混在一起；(2) 學生容量有限，要同時貼合兩個域的 teacher 行為。
- 這組對照讓假說成立得很乾淨：學生在來源域影像上已經學到 CLIP 能給的全部（8a 加再多 Market 影像也沒用）；CLIP 的泛化只能在目標域影像上傳遞。
- 對照事前標準：8b 跨域達成（31.7 ≥ 22.6）、Market 未達成（88.7 < 89.2）；8a 沒有接近 8b，所以「資料而非損失」的說法成立。
- 可能的下一步（未決定，要事先寫規則，不能看 CUHK03 測試集挑參數）：domain-specific BN（兩個域各自的 BN 統計，推論時用目標域的），或只在無標註 batch 上凍結 BN 統計，看能否在保住跨域增益的同時消除 Market 的下降。

### AdaBN 診斷第一輪：重估方法本身有偏差（2026-10-21，commit da9e167）
`eval_matrix --bn-recal`，第一版用**乾淨影像**（測試用 resize，無資料增強）重估 BN 統計量。6c_s1、8a 在 Colab；8b 尚未評估。

| run | Market（原統計量） | Market（Market 乾淨影像重估） | CUHK03（原統計量） | CUHK03（CUHK03 乾淨影像重估） |
|---|---|---|---|---|
| 6c_s1 | 89.5 | **83.8** | 17.5 | 21.2 |
| 8a | 89.5 | **83.9** | 15.3 | 21.3 |

- **工具沒有通過自我檢查**：用同一個資料集（Market）的訓練影像重估，應該要重現原本的 89.5，結果掉到 83.8。原因：權重是在**有資料增強**（random erasing、pad/crop）的 batch 統計量下訓練的，乾淨影像的統計量（例如變異數較小）和它不一致。這一輪的 Market 數字不能拿來判斷 8b 的下降，事前規則 1 的比較暫不成立。
- 修正（同日）：重估預設改用該 run 的**訓練資料增強**（`--bn-recal-transform train`）；先確認 6c / 8a 在 Market 上重估後能重現 89.5（±0.3），通過了才用它判斷 8b。
- 已經可以看出的一點：即使是這個有偏差的版本，6c 只做 AdaBN 在 CUHK03 上是 21.2（+3.7），遠低於 8b 的 31.7 → 方法 4 的增益不是單純 BN 適應能解釋的（仍需用修正後的版本、加上 8b + AdaBN 再確認）。

### AdaBN 診斷第二輪（訓練資料增強重估，commit 8b61e55）：Market 下降來自 BN 統計量（2026-10-23~24）
6c_s1、8a 在 Colab；8b 在 Kaggle（掛 8b notebook 的輸出）。重估用各 run 的訓練資料增強、固定 seed 打亂的 batch，只用影像、不用標籤。flip TTA，無 re-ranking。

| run | Market（原統計量） | Market（Market 訓練影像重估） | CUHK03（原統計量） | CUHK03（CUHK03 訓練影像重估） |
|---|---|---|---|---|
| 6c_s1 | 89.5 | 89.6 | 17.5 | 22.9 |
| 8a | 89.5 | 89.5 | 15.3 | 22.9 |
| **8b** | 88.7 | **90.0** | 31.7 | **33.0** |

- **工具通過自我檢查**：6c、8a 用 Market 自己的影像重估後是 89.6 / 89.5，重現原數字（事前門檻 89.5 ± 0.3）。
- **事前規則 1 成立**：8b 用 Market 的 BN 統計量是 **90.0 ≥ 89.2** → 8b 的 −0.8 完全來自混合了兩個域的 BN 統計量，權重本身沒有變差（甚至比 6c 高 0.5，單 seed）。修正方式：每個域一套 BN 統計量（同一組權重；推論 Market 用 Market 統計量、CUHK03 用 CUHK03 統計量），不需要重新訓練。
- **方法 4 的增益超過單純 BN 適應（事前門檻 +5）**：同樣做 AdaBN，6c 在 CUHK03 是 22.9、8a 是 22.9，8b 是 **33.0**，多 **+10.1**。AdaBN 本身對 6c 也有 +5.4（17.5 → 22.9），這是已知方法的效果，要和方法 4 分開報。
- **方法 4 的最終數字（每個域各自的 BN 統計量）**：Market 90.0、CUHK03 33.0。補回學生與 CLIP 跨域差距 (33.0 − 17.1) / (39.2 − 17.1) = **72%**；達到 CLIP 跨域能力的 84%。兩項事前標準都達成，但要註明：Market 的修正是在第一次未達標後，依事先寫好的規則 1 決定的。
- 限制：8b 只有 1 個 seed；每域 BN 統計量需要目標域的無標註影像（UDA 設定本來就有）。
- 實務意義：部署到新的相機環境時，只要用該環境的無標註影像重估 BN 統計量（幾分鐘、不用訓練），就能得到這個域的版本。
- 備註：第一輪（乾淨影像重估）的錯誤示範了為什麼要先做自我檢查：沒有檢查的話，會把工具本身的 −5.7 誤當成模型的問題。

### Plan 8b seed 1（Kaggle T4，2026-10-24~25，7.5 h）：方法 4 = 2 seeds
同 8b 設定、`seed=1`，訓練後跑一般評估與 `--bn-recal`（訓練資料增強重估）。Final（= 表中 trained 列）。

| | Market（訓練時統計量） | Market（Market 重估） | CUHK03（訓練時統計量） | CUHK03（CUHK03 重估） |
|---|---|---|---|---|
| 8b seed 42 | 88.7 | 90.0 | 31.7 | 33.0 |
| 8b seed 1 | 88.7 | 90.0 | 31.5 | 33.8 |
| **8b（2 seeds）** | 88.7 ± 0.0 | **90.0 ± 0.0** | 31.6 ± 0.1 | **33.4 ± 0.6** |

- 兩個 seed 的型態完全一致：訓練時統計量下 Market −0.8；各域重估後 Market 90.0、CUHK03 +1.4~2.3。
- **方法 4（每域 BN 統計量）最終：Market 90.0 ± 0.0、CUHK03 33.4 ± 0.6（2 seeds）**。對照 6c 89.5 ± 0.0 / 17.1 ± 0.6；6c + AdaBN 22.9（1 seed）→ +10.5；補回學生與 CLIP 跨域差距的 (33.4 − 17.1) / (39.2 − 17.1) = **74%**；達到 CLIP-ReID 跨域能力的 85%，參數約 1/40。

