# OSNet x0.25 學生模型訓練計畫

> 目標：在 Market-1501 上做出「盡可能好的 OSNet x0.25」，並證明每一步的提升來自哪裡。
> 原則：**先把 baseline 做強，再談蒸餾**——KD 的增益要相對一個調好的 baseline 才有說服力。

## 0. 參考數字與成功標準

| 模型 | 參數 | Mult-Adds | mAP | R1 | 來源 |
|---|---|---|---|---|---|
| ResNet-50 teacher（我們） | 23.5M | 4.05G | 86.6 | 95.0 | `docs/experiments.md` |
| ResNet-18 baseline（我們） | 11.2M | 1.99G | 80.6 | 92.0 | 同上 |
| ResNet-18 + KD（我們，傳統基準） | 11.2M | 1.99G | 84.8 | 93.6 | 同上；**補回率 70%** |
| OSNet x1.0（論文） | 2.2M | 979M | 84.9 | 94.8 | Zhou et al. ICCV 2019, Table 6 |
| **OSNet x0.25（論文）** | **0.2M** | **82M** | **77.8** | **92.2** | 同上；只用 CE + label smoothing |

參數為部署用（不含分類層），GMACs 由 `tools/benchmark.py` 實測（256×128）。

成功標準（**是目標，不是預測**）：

| 等級 | OSNet x0.25 mAP | 說明 |
|---|---|---|
| 及格 | ≥ 79 | 我們的 baseline 配方（BNNeck + triplet）勝過論文 77.8 |
| 目標 | ≥ 82 | KD 補回 baseline 與 teacher 差距約一半 |
| 理想 | ≥ 84 | 以 1/125 參數逼近 R50 teacher（差 < 3 點） |

**報告最後一個 epoch 的數字**（log 的 `Final (epoch N)`），不用 best.pth 的——那是在測試集上挑的。

除了 mAP，每個實驗都報：**差距補回率** = (KD − baseline) / (teacher − baseline)，以及參數量 / FLOPs / 延遲。

**雜訊規則**：單一 seed 的 mAP 波動約 ±0.3。一個改動只有 **+0.5 mAP 以上** 才算有效、保留到下一階段；最終配方用 3 個 seed 報平均 ± 標準差。

---

## 階段 0：基礎建設 ✅ 完成（2026-10-09）

**完成內容與驗證**
- `models/osnet.py`：OSNet x0.25/0.5/0.75/1.0。用 torchreid 原始碼建同一個模型、存權重、載入我們的版本，**輸出逐位元一致（差異 0）**。
- 實測尺寸（`tools/benchmark.py`）與論文吻合：

  | 模型 | 參數（部署） | GMACs | PyTorch CPU b1 | **ONNX Runtime CPU b1** |
  |---|---|---|---|---|
  | ResNet-50 | 23.5M | 4.05 | 48 ms | 29.4 ms |
  | ResNet-18 | 11.2M | 1.99 | 24 ms | 15.3 ms |
  | OSNet x1.0 (512-d) | 2.17M | 0.98 | 34 ms | 11.0 ms |
  | **OSNet x0.25 (512-d)** | **0.203M** | **0.082** | 17 ms | **2.3 ms** |

  （2 核心雲端容器、batch 1，僅供相對比較；正式部署表要在固定硬體上重測。）
  **發現**：PyTorch eager 下 OSNet x0.25 只比 R18 快 1.4 倍，ONNX Runtime 下快 6.6 倍——OSNet 的許多小型 depthwise 運算在 eager 模式下被框架開銷淹沒。部署比較一律用 ONNX Runtime。
- KD 損失：logit、DKD、feature、similarity、simdist、RKD、attention transfer；多 teacher（各自權重與 KD 覆寫）；`freeze_backbone_epochs`；`color_jitter`。每個損失都有性質測試（例如 RKD 對特徵縮放不變、AT 與通道數無關）。
- 舊的 R18 KD 設定數值完全不變（新舊程式輸出逐值比對一致）。
- 設定檔：`configs/osnet/` 下 1a–1d、1s、2a–2c，檔名即實驗編號。
- 訓練 log 新增每個 epoch 的資料載入時間 `(data Ns)`，用來判斷 Colab 2 vCPU 是否成為瓶頸。

**原本的待辦項目**

| 項目 | 內容 |
|---|---|
| OSNet x0.25 / x1.0 | 自己實作 `models/osnet.py`（相容 torchreid 權重格式），支援載入 ImageNet 預訓練權重；`embed_dim` 可設（128 原生 / 512 加 FC，論文用 512） |
| 部署評估 `tools/benchmark.py` | 參數量、FLOPs、GPU/CPU 延遲（batch 1 / 64）、ONNX 匯出並比對輸出 |
| KD 擴充 | 注意力遷移（attention transfer）、DKD、RKD（距離 + 角度）、多 teacher、ranking/similarity-distribution KD |
| 訓練效率 | `eval_period` 前期拉長（評估一次 2.5 分鐘）；資料載入是 Colab 2 vCPU 的瓶頸，評估是否改用預先縮放的圖片 |

---

## 階段 1：做強 OSNet x0.25 baseline（無 KD，約 4 次訓練）

所有比較都以 `1a` 為起點，一次只改一個因素。

| ID | 設定 | 想回答的問題 |
|---|---|---|
| 1a | ImageNet 預訓練 + 與 R18 baseline 相同的 BoT 配方，120 ep | 我們的配方能否超過論文 77.8 |
| 1b | 1a + `embed_dim=512`（vs 128） | 小網路是否需要較寬的嵌入 |
| 1c | 1a + 較高學習率 / 不同 warmup（論文用 1.5e-3 + 前 10 ep 凍結 backbone） | 小網路的最佳 LR |
| 1d | 1a + 240 ep | 小網路是否需要更長訓練 |
| 1e | 1a + lr 1e-3 + 240 ep | **1a 結果加入**：1a 明顯欠擬合，直接測「高 LR + 長訓練」的組合 |

**1a 結果（2026-10-09）**：mAP 64.3 / R1 83.2，比論文低 13.5 mAP，訓練損失未收斂（CE 1.26 vs R18 的 1.07）→ 欠擬合，LR 太小、降得太早。執行順序改為 **1c → 1e**，1b / 1d / 1s 視結果再跑。詳見 `experiments.md`。

**1c / 1e 結果（2026-10-10）**：1c 71.4 / 88.0；**1e 77.0 / 90.6**（距論文 77.8 只差 0.8）。關鍵是高 LR 階段要夠長（80 epoch）。

→ **B\* = 1e 配方**（lr 1e-3、240 ep、ep80/140 降 LR），寫在 `configs/osnet/_bstar.yaml`，所有階段 2/3 實驗都繼承它。1b / 1d / 1s 暫緩（不影響 KD 結論，有空再補）。
→ 另訓練一次 **從零開始（無 ImageNet）** 作為參考：論文圖中寫「Train from scratch」，要量化預訓練的貢獻。

---

## 階段 2：直接蒸餾 R50 → OSNet x0.25（傳統 KD 基準，約 3–4 次）

| ID | KD 損失 | 備註 |
|---|---|---|
| 2a | logit 1.0 + similarity 1.0, T=4 | 與 R18 KD 相同設定，直接對照（R18 的補回率是 70%） |
| 2b | 2a + feature（2048→embed 投影） | 特徵對齊是否幫得上維度差 4–16 倍的學生 |
| 2c | 2a + attention transfer（權重 4.0） | R50（last_stride=1）與 OSNet 在 256×128 輸入下最後特徵圖都是 16×8，空間注意力可直接對齊，與通道數無關。權重 4 ≈ 原論文 β=1000（他們對 128 個位置取平均並乘 1/2，我們是加總） |
| 2d | **只用 logit**（新增） | R50→R18 中 `kd_sim` 只有 `kd_logit` 的 1/40，確認 similarity 項是否有貢獻 |
| 2e | 最佳組合 + T ∈ {2, 8} | 只在 2a–2d 有明顯差異時才做 |

執行順序：**2a → 2c → 2d**，2b（feature）優先度最低。每次約 3–3.5 小時。OSNet 的差距是 86.6 − 77.0 = 9.6 mAP。

→ 產出 **D\***（直接 KD 最佳）與它的差距補回率。

**階段 2 結果（2026-10-10）**：2a 82.6、2c 82.6、**2d 82.7**——similarity 與 attention 都沒有貢獻，增益全部來自 logit KD。**D\* = 2d（logit only）**，補回率 **59%**（R18 是 70%），capacity gap 存在但不嚴重。0.2M 的 OSNet-KD 已超過 11.2M、無 KD 的 R18（80.6）。

**對階段 3 的影響**：既然只有 logit 訊號有用，階段 3 集中在「讓 logit 訊號更好學」：助教模型（3a）、DKD（3d）、雙 teacher（3c）；3e（RKD / simdist）降為最低優先。2b（feature）不再跑。拿來和 R50 → R18 的補回率比：若 OSNet 的補回率明顯較低，就證實了 capacity gap，階段 3 的動機成立。

---

## 階段 3：縮小 capacity gap（核心方法，約 6–8 次）

執行順序（2026-10-10 更新）：**3a → 3d → 3c**，都以 B\* 為底、只用 logit 類 KD；設定檔 `configs/osnet/3a_ta_r18kd.yaml`、`3d_dkd_r50.yaml`、`3c_multi_r50_r18kd.yaml`。

**階段 3 結果（2026-10-11）**：**3d DKD 85.0 / 93.9（補回 83%，達「理想」≥84）**；3a 助教 82.6、3c 雙 teacher 82.1 都不如直接 logit KD（82.7）。
→ 瓶頸不是 capacity gap，而是經典 logit KD 對「暗知識」（非目標類關係）傳遞太弱——teacher 用 label smoothing 且信心很高時尤其嚴重；DKD 把這部分獨立加權後問題解決。
→ **研究主軸調整**：從「用助教縮小差距」改為「為什麼 DKD 對極小學生特別有效」。

**機制驗證結果（2026-10-11）**：無 LS teacher 87.1；3g（經典 KD + 無 LS teacher）83.8 → 去掉 LS 補回 DKD 優勢的約一半，另一半來自 DKD 的損失設計。新增 **3h = DKD + 無 LS teacher**（`3h_dkd_r50nols.yaml`），列為最優先。

**接下來（依序）**：
| ID | 實驗 | 回答的問題 | 時間 |
|---|---|---|---|
| 3h ✅ | `3h_dkd_r50nols.yaml` | 未疊加：84.5（3d 85.0）→ 兩者作用在同一處 | 完成 |
| seeds ✅ | `3d_dkd_r50.yaml seed=1 / seed=2` | 3 seed 皆 85.0（R1 93.9–94.1）→ 變異 ≲ 0.1，0.5 以上的差距可信 | 完成 |
| R18-DKD ✅ 86.2（+1.4 vs KD，補回 93%）→ DKD 普遍有效，非只對大差距 | `student_r18_dkd.yaml` | DKD 對同質、差距小的 R18 是否也 +2 以上？（一般性） | ~1.3 h |
| 3d-β | `3d_dkd_r50_b16.yaml`、`3d_dkd_r50_b4.yaml` | β 敏感度；小學生是否需要更大的暗知識權重 | 2 × 2.7 h |
| LS 機制 | `teacher_r50_nols.yaml` → `3g_kd_r50nols_logit.yaml` | 無 LS teacher + 經典 KD 能否追上 DKD | 2.3 h + 2.7 h |
| seeds | 最終配方與 D\* 各 3 seed | 結論是否穩定 | 4 × 2.7 h |

助教（3b OSNet x1.0）與 RKD/simdist（3e）暫停：3a/3c 已顯示換 teacher 不是瓶頸。

### 3A. 助教模型（Teacher Assistant, TAKD）
Mirzadeh et al., AAAI 2020：teacher 與 student 差距太大時，插入中間大小的助教，分段蒸餾。

| ID | 路徑 | 說明 |
|---|---|---|
| 3a | R50 → **R18-KD** → OSNet x0.25 | 直接重用正在跑的 R18 KD 結果當助教，零額外成本 |
| 3b | R50 → **OSNet x1.0** → OSNet x0.25 | 助教與學生**同架構家族**（相同的 omni-scale 結構與歸納偏置），預期比跨家族的 R18 更好傳遞；需先訓練 OSNet x1.0（+1 次） |

### 3B. 多 teacher 聯合
| 3c | R50 + 最佳助教同時當 teacher（加權） | 大 teacher 提供上限、助教提供學生學得動的訊號 |

### 3C. 更適合 ReID 的蒸餾訊號
| 3d | logit KD 換成 **DKD**（Zhao et al., CVPR 2022） | 把目標類與非目標類分開加權；我們的 teacher 用了 label smoothing，非目標類的「暗知識」被壓平，DKD 可單獨放大這部分 |
| 3e | similarity KD 換成 **RKD / 相似度分佈 KD** | ReID 的評估是排序；對齊 batch 內「誰比誰更像」的分佈，而不是相似度的絕對值，對小學生較寬容 |

### 3D. 訓練策略
| 3f | 最佳配方 + 長訓練（240–300 ep） | Beyer et al., CVPR 2022「A good teacher is patient and consistent」：KD 需要比一般訓練長得多的時程；我們已經讓 teacher 與 student 看同一張增強後的圖（consistent），只差 patient |
| 3g | 自蒸餾（born-again）：用最佳 OSNet x0.25 當 teacher 再訓一個 OSNet x0.25 | 成本低，常有 +0.5–1 的額外增益 |

→ 產出 **F\***（最終配方）。

---

## 階段 4：更強 teacher —— CLIP-ReID（問題 2；程式已就緒，2026-10-12）

**為什麼**：問題 2 的診斷顯示瓶頸在 teacher 的泛化，不在蒸餾。CLIP-ReID 在 Market 同域只比 R50 高約 3 mAP，在實際場景卻高 8–16。

**做法**（`kdreid/models/clipreid.py`，`configs/clip/`）：
- teacher = CLIP-ReID 官方公開的 ViT-B/16 Market checkpoint（ViT-CLIP-ReID，無 SIE / OLP）。用公開權重是為了可重現，KD-ReID 不依賴任何外部系統。
- 只重建影像分支：ViT + 兩個 BNNeck + 兩個身分分類頭；文字分支與 prompt learner 不載入。實作已逐位元對照官方程式（stride 16 / 12 輸出完全相同）。
- 輸入在模型內從 ImageNet 正規化轉成 CLIP 正規化（mean = std = 0.5），學生的資料流程不變。
- logits = 768 維與 512 維兩個分類頭的平均（`logit_head`）。檢索特徵 = 兩者 BN 前串接，與官方測試一致。
- 不支援 SIE（需要相機 id 才能推論）。
- 先跑 `tools/check_teacher.py`：(1) 用我們的評估器重現官方精度；(2) 確認類別順序和我們的訓練標籤對齊（logit KD / DKD 的前提）。

**實驗**：4a = CLIP → OSNet（DKD，F\* 配方）、4b = CLIP → OSNet（logit KD）。和 R50 teacher 的 F\* / D\* 比較同域、Market → CUHK03-NP / MSMT17 跨域，以及使用者系統（輔助）。
**限制**：官方 checkpoint 只有 Market / MSMT17 / Duke，**沒有 CUHK03-NP**。CLIP teacher 的實驗先在 Market（之後 MSMT17）做；CUHK03-NP 需要自己訓練 CLIP-ReID（兩階段，成本高），之後再決定。

### 原始想法（保留）
- CLIP-ReID → OSNet x0.25 直接蒸餾，與 R50 → OSNet x0.25 比較；
- CLIP-ReID → 助教 → OSNet x0.25；
- 驗證「更強的 teacher 是否反而更難教」，以及階段 3 的方法是否在更大差距下更有價值。

## 驗證：評估協定與跨域泛化（2026-10-11 加入）

**Re-Ranking**：所有報告數字都 **沒有** re-ranking（`evaluator.py` 只用 cosine 距離；repo 內無 RK 程式）。唯一的測試期加成是 flip TTA，`eval_matrix.py` 會同時列出無 flip 的數字。文獻比較一律對照無 RK 的數字（例：BoT R50 無 RK 85.9 mAP）。

**跨域 zero-shot**（只在 Market 訓練，直接測沒看過的資料集）：
- 對象：teacher R50、R18 baseline / KD、OSNet B\* / D\*（logit KD）/ DKD。
- 資料集：~~DukeMTMC-reID~~（已撤回，論文不使用，見階段 5）；CUHK03-NP（已支援，PNG）；MSMT17 需另外申請，格式不同，待支援。
- 判斷標準：跨域每個模型都會大幅下降（文獻中 Market→Duke 直接遷移常見 R1 約 40%、mAP 20–30%，僅供量級參考）。重點是：
  1. DKD 學生的跨域排名是否仍 ≥ logit KD 學生與 B\*；
  2. **保留率** = 跨域 mAP ÷ 同域 mAP，是否與 teacher 相當——若 DKD 學生同域高、跨域保留率明顯較低，代表它過度擬合了 teacher 在 Market 上的偏差。
- 真實場景（例如 YOLO 切出的路口行人）沒有 ID 標註，無法算 mAP；可做檢索結果的視覺化抽查，或自行標一小組（例如 2 台相機、20–30 人）當內部測試集。

## 階段 5：論文驗證（2026-10-12 決定）

**最終目標順序：先完成問題 1，再做問題 2。**
1. **問題 1（論文主幹）**：OSNet x0.25 能否透過 DKD 逼近大模型？Market 上已很有說服力，缺統計穩定性與第二、三個資料集。
2. **問題 2（之後，可能的新方法）**：OSNet 能否成為實際跨相機系統的強模型？使用者的 Panorama 測試仍落後 CLIP-ReID。先做 teacher 診斷再決定方向（見下）。

### 資料集
- **不使用 DukeMTMC-reID**：CVPR 2026 作者指引將其列為已撤回資料集（使用需特別理由並受更嚴格審查），ECCV 2026 倫理指引表示不應使用。先前的 Market → Duke 跨域結果只作內部參考，論文改用 Market → CUHK03-NP / MSMT17。
- 論文資料集：**Market-1501**（已完成）、**CUHK03-NP**（detected 為主，labeled 選擇性；使用者下載中）、**MSMT17**（申請中，可能需數天；到手後寫讀取程式，建議 V2）。

### 5a：CUHK03-NP 主表（`configs/cuhk03/`，與 Market 同配方，只換資料集；KD teacher 用 CUHK03 teacher 的 `last.pth`）

| 順序 | config | 角色 |
|---|---|---|
| 1 | `cuhk03/teacher_r50` | 上界；KD 的前置 |
| 2 | `cuhk03/osnet_bstar` | OSNet 下界（不需 teacher，可並行） |
| 3 | `cuhk03/osnet_fstar` | **主要結果**（DKD） |
| 4 | `cuhk03/osnet_dstar` | 傳統 logit KD 對照 |
| 5–7 | `cuhk03/r18_baseline` / `r18_kd` / `r18_dkd` | 傳統基準（成本不夠可只在 Market 報） |

判斷標準：排名 B\* < D\* < F\* 是否重現；補回率是否與 Market 同量級（D\* 59%、F\* 83%）；R18 上 DKD 是否仍 > KD。
CUHK03-NP 規模小（767 人 / 7.4k 張，每人只在 2 台相機），Market 配方可能不是最佳，但主表優先「同配方」，不另外調參。

### 5b：統計穩定性（Market）
teacher、B\*、D\* 各補 seed 1、2（notebook 7.8）；F\* 已有 3 seeds。主表報 3 seeds 平均 ± sd。

### 5c：跨域（只需評估）
Market → CUHK03-NP（notebook 7.7 第二格），之後加 Market → MSMT17。

### 5d：MSMT17（資料到手後）
先只跑 OSNet B\*，比較 240 epoch 與 ~120 epoch（訓練集是 Market 的 2.5 倍，120 epoch 的步數與 Market 相當），定配方後跑 teacher / D\* / F\*。

### 問題 2 的前置診斷：teacher 在實際場景的表現
把 Market R50 teacher 打包（notebook 7.6 第二格，`export_package.py` 已支援 ResNet），在使用者系統跑同樣的 C01→C02、C02→C01、Panorama：
- R50 teacher 也明顯落後 CLIP-ReID → 瓶頸是 teacher → 階段 4（CLIP-ReID teacher）。
- R50 teacher 接近 CLIP-ReID、OSNet 落後 → 蒸餾流失跨相機資訊 → 設計跨相機關係蒸餾（在 MSMT17 上驗證）。
CLIP-ReID 也需補 C02→C01 與 Panorama 的數字。

**診斷結果（2026-10-12）：瓶頸在 teacher。** R50 teacher 在三個實際場景都落後 CLIP-ReID 7.9–16.3 mAP；F\* 在兩個跨相機方向反而比 teacher 高（+3.3 / +4.6），Panorama −1.4。看不到蒸餾退化。
→ 問題 2 的方向定為 **CLIP-ReID 當 teacher**（階段 4），不先改 DKD loss。尚待：實際場景的 B\* / D\*（量化 KD 淨效益）。
→ 順序不變：問題 1（CUHK03-NP、seeds、MSMT17）優先。CLIP-ReID teacher 的程式整合不需要 GPU，可以並行準備。

### 部署表
參數、FLOPs、CPU/GPU 延遲、ONNX，對照 R50 / R18 / OSNet x1.0（固定硬體）。外部系統測試只作應用佐證，不進主表。

---|---|---|---|
  | 1 | `duke/teacher_r50` | 上界；KD 的前置 | ~3h |
  | 2 | `duke/osnet_bstar` | OSNet 下界 | ~3h |
  | 3 | `duke/osnet_fstar` | **主要結果**（DKD） | ~3.5h |
  | 4 | `duke/osnet_dstar` | 傳統 logit KD 對照 | ~3.5h |
  | 5–7 | `duke/r18_baseline` / `r18_kd` / `r18_dkd` | 傳統基準 | 各 ~2h |

  判斷標準：(1) 排名 B\* < D\* < F\* 是否重現；(2) 補回率是否與 Market 同量級（D\* 59%、F\* 83%）；(3) R18 上 DKD 是否仍 > KD。全部跑完後做 Duke → Market 反向 zero-shot。
- 第三個資料集（論文常見）：MSMT17 最有份量（規模最大，且 Duke 已被原作者撤下）；CUHK03-NP 已支援。視資料取得情況決定。
- 部署表：參數、FLOPs、CPU/GPU 延遲、ONNX，對照 R50 / R18 / OSNet x1.0（固定硬體）。
- 外部系統測試（使用者自有場景）只作為應用佐證，不進主表。

---

## 執行順序與 GPU 預算

```
現在       R18 baseline（續跑中）→ R18 KD           ← 3a 的助教
階段 0     ✅ OSNet、benchmark、新 KD 損失             （不佔 GPU）
階段 1     1a → 1b/1c/1d → 決定 B*                   4–5 次
階段 2     2a → 2b/2c                               3–4 次
階段 3     3a → 3b(含訓練 OSNet x1.0) → 3c → 3d/3e → 3f → 3g   7–8 次
```

預估每次 OSNet x0.25 訓練（120 ep，含 teacher forward）在 T4 約 1.5–2 小時；全部約 20 次、30–40 GPU 小時。Colab 免費版每天可用時數有限，必要時考慮 Colab Pro 或縮短評估頻率。

**每個階段結束時的決策點**：先看結果再決定下一階段要跑哪些，不必全部照表執行。例如若 2a 的補回率已接近 R18 的情況，capacity gap 不明顯，階段 3 就以 3d–3g 為主，助教模型降為次要。

## 參考文獻
- Zhou et al., *Omni-Scale Feature Learning for Person Re-Identification*, ICCV 2019 — https://arxiv.org/abs/1905.00953
- Luo et al., *Bag of Tricks and A Strong Baseline for Deep Person Re-identification*, CVPRW 2019
- Hinton et al., *Distilling the Knowledge in a Neural Network*, 2015
- Mirzadeh et al., *Improved Knowledge Distillation via Teacher Assistant*, AAAI 2020
- Zhao et al., *Decoupled Knowledge Distillation*, CVPR 2022
- Park et al., *Relational Knowledge Distillation*, CVPR 2019
- Zagoruyko & Komodakis, *Paying More Attention to Attention*, ICLR 2017
- Beyer et al., *Knowledge Distillation: A Good Teacher is Patient and Consistent*, CVPR 2022
- Xie et al., *Towards a Smaller Student: Capacity Dynamic Distillation for Efficient Image Retrieval*, CVPR 2023 — https://arxiv.org/abs/2303.09230
- Li et al., *CLIP-ReID*, AAAI 2023


## 階段 3 結論（2026-10-12）

- **最終 OSNet x0.25 配方 F\* = 3d**：B\* 配方 + R50（有 LS）teacher + DKD（α1、β8、T4）。**85.0 mAP / 94.0 R1（3 seeds）**。
- 主要發現：
  1. 小學生需要更長的高學習率階段（64.3 → 77.0）。
  2. ReID 蒸餾的增益全部來自 logit 訊號；similarity / attention / 助教 / 雙 teacher 無效或有害。
  3. DKD 普遍優於傳統 KD（R18 +1.4、OSNet +2.3），β ≥ 8 飽和；約一半的優勢可由 teacher 的 label smoothing 壓抑暗知識來解釋（3g），且與「去掉 LS」不疊加（3h）。
  4. 「助教模型縮小 capacity gap」的假設不成立（3a、3c）。
- 剩下：見「階段 5：論文驗證」。
