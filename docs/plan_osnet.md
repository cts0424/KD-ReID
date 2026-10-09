# OSNet x0.25 學生模型訓練計畫

> 目標：在 Market-1501 上做出「盡可能好的 OSNet x0.25」，並證明每一步的提升來自哪裡。
> 原則：**先把 baseline 做強，再談蒸餾**——KD 的增益要相對一個調好的 baseline 才有說服力。

## 0. 參考數字與成功標準

| 模型 | 參數 | Mult-Adds | mAP | R1 | 來源 |
|---|---|---|---|---|---|
| ResNet-50 teacher（我們） | 25.1M | ~4.1G* | 86.6 | 95.0 | `docs/experiments.md` |
| OSNet x1.0（論文） | 2.2M | 979M | 84.9 | 94.8 | Zhou et al. ICCV 2019, Table 6 |
| **OSNet x0.25（論文）** | **0.2M** | **82M** | **77.8** | **92.2** | 同上；只用 CE + label smoothing |

\* R50 FLOPs 為 last_stride=1、256×128 的估計值，待部署評估工具實測。

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

→ 產出 **B\***（最佳 baseline 配方），之後所有實驗都以 B\* 為底。
→ 另訓練一次 **從零開始（無 ImageNet）** 作為參考：論文圖中寫「Train from scratch」，要量化預訓練的貢獻。

---

## 階段 2：直接蒸餾 R50 → OSNet x0.25（傳統 KD 基準，約 3–4 次）

| ID | KD 損失 | 備註 |
|---|---|---|
| 2a | logit 1.0 + similarity 1.0, T=4 | 與 R18 KD 相同設定，直接對照 |
| 2b | 2a + feature（2048→embed 投影） | 特徵對齊是否幫得上維度差 4–16 倍的學生 |
| 2c | 2a + attention transfer（權重 4.0） | R50（last_stride=1）與 OSNet 在 256×128 輸入下最後特徵圖都是 16×8，空間注意力可直接對齊，與通道數無關。權重 4 ≈ 原論文 β=1000（他們對 128 個位置取平均並乘 1/2，我們是加總） |
| 2d | 最佳組合 + T ∈ {2, 8} | 只在 2a–2c 有明顯差異時才做 |

→ 產出 **D\***（直接 KD 最佳）與它的差距補回率。拿來和 R50 → R18 的補回率比：若 OSNet 的補回率明顯較低，就證實了 capacity gap，階段 3 的動機成立。

---

## 階段 3：縮小 capacity gap（核心方法，約 6–8 次）

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

## 階段 4：更強 teacher —— CLIP-ReID（之後）
- CLIP-ReID → OSNet x0.25 直接蒸餾，與 R50 → OSNet x0.25 比較；
- CLIP-ReID → 助教 → OSNet x0.25；
- 驗證「更強的 teacher 是否反而更難教」，以及階段 3 的方法是否在更大差距下更有價值。

## 階段 5：收尾
- F\* 跑 3 個 seed；
- DukeMTMC-reID 上重複 baseline / D\* / F\*，確認不是只對 Market 有效；
- 部署表：參數、FLOPs、CPU/GPU 延遲、ONNX，對照 R50 / R18 / OSNet x1.0。

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
