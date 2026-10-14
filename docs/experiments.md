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
