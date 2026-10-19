# 方法 4：無標註影像上的蒸餾（plan 8；程式完成 2026-10-19，待跑）

2026-10-19。目的：補上「學生在跨域只拿到 CLIP-ReID 44% 能力」這個缺口，同時是論文可以提出的方法貢獻。

## 動機（已有數據）

| | 同域 Market mAP | 跨域 Market → CUHK03-NP mAP |
|---|---|---|
| CLIP-ReID teacher | 89.9 | **39.2** |
| OSNet x1.0 ← CLIP DKD（6c，2 seeds） | 89.5（99.6%） | **17.1 ± 0.6（44%）** |
| OSNet x1.0 ← R50 DKD（6b） | 88.3 | 8.8 ± 0.8 |

同域幾乎全部學到，跨域只學到不到一半。

**假說**：學生只在 Market 影像上模仿 teacher，所以只學到 teacher 在 Market 分布上的行為。CLIP 的泛化要在沒看過的影像分布上才會顯現，學生從來沒有在那種影像上被要求模仿它。

## 方法

每個 iteration 除了原本的 Market 標註 batch，再取一個**無標註**影像 batch：

- Market batch：和 6c 完全相同（CE + triplet + DKD）。
- 無標註 batch 不能用 CE、triplet 或 DKD（都需要身份標籤），改用三種不需要標籤的蒸餾：
  1. **特徵蒸餾**：學生特徵經線性 projector 後，和 teacher 特徵做 cosine 損失（新的 `cosine` 項）。
  2. **批次內關係蒸餾**：讓學生的相似度分布接近 teacher（已有 `simdist`）。
  3. **logit KD**：teacher 對 751 個 Market 身份的機率分布本身就是軟標籤，在任何影像上都能算，不需要知道真實身份（已有 `logit`）。

實作（commit 見 experiments.md）：

- `build_unlabeled_loader`：`unlabeled.dataset`（只取訓練集影像、標籤換成 −1）或 `unlabeled.folder`（任何資料夾），訓練用的資料增強、shuffle。
- Trainer：每個 iteration 多取一個無標註 batch，學生與 teacher 各做一次 forward，只算 `unlabeled.losses` 裡的項（只允許不需要標籤的 `DistillLoss.LABEL_FREE`，寫 dkd 會直接報錯），乘上 `unlabeled.weight`，log key 以 `u_` 開頭。
- 新的 KD 項 `cosine`：1 − cos(projector(學生), teacher)。原本的 `feature` 是對單位向量逐元素做 MSE，等於 (2 − 2cos) / 維度，CLIP 的 1280 維下數值太小，權重沒辦法和 CE / DKD 比較；`cosine` 的數量級是 O(1)。
- 設定：`configs/plan8/_unl.yaml`（6c 配方 + `unlabeled: {batch_size: 32, weight: 1.0, losses: {cosine: 1, simdist: 1, logit: 1}}`），`8a_unl_market.yaml`、`8b_unl_cuhk03.yaml`。
- 8a 的對照設計：額外的 batch 也是 32 張，只是改成 Market 影像。運算量和看到的影像數都和 8b 一樣，唯一差別是影像的分布。
- 注意：學生在無標註 batch 上是 train mode，BN 的 running statistics 也會被目標域影像更新，這是 8b「資料」效果的一部分。

## 實驗（x1.0 ← CLIP，6c 配方；每個約 6 h：多一個 32 張的 batch，學生與 CLIP 各多一次 forward）

| 編號 | 無標註影像來源 | 回答的問題 | 評估設定 |
|---|---|---|---|
| 8a | Market 訓練集（不用標籤；和 8b 一樣多一個 32 張的 batch） | **對照組**：效果來自新的損失，還是新的資料？ | zero-shot |
| 8b | CUHK03-NP 訓練集（不用標籤） | 目標域影像能補回多少？ | **無監督域適應（UDA）**，不能叫 zero-shot |
| 8c | MSMT17 訓練集（不用標籤） | 不看目標域，只增加影像多樣性夠不夠？ | **domain generalization**，仍是 zero-shot（MSMT17 → CUHK03 沒有重疊） |

8c 要等 MSMT17 資料到手；8a、8b 現在就能做。

## 事前判斷標準（跑之前寫下，避免事後挑）

- **成功**：8b 的 Market → CUHK03 ≥ 22.6，也就是補回 17.1 → 39.2 這段差距的 ≥ 25%；同時 Market 同域下降 ≤ 0.3（≥ 89.2）。
- **8a 對照**：若 8a 也接近 8b，代表是損失的功勞、不是資料，論文的說法要改。
- **8c**：若 8c 有 8b 的一半以上增益，就是更強的結論，因為不需要目標域資料。
- **失敗**：8b < 19（+2 以內），方法 4 不寫進論文，作為負結果記錄。

## 風險

- 4c（x0.25，Market 上只用 simdist）只有 78.9，關係型 KD 單獨使用很弱。但這裡它是**加在** DKD 之上，學生也是 x1.0，不是同一個情況。8a 會直接回答。
- 8b 用了 CUHK03 訓練影像，必須在論文中清楚標成 UDA 設定，和 zero-shot 數字分開，不可混在同一欄。
- 成本：8a + 8b 各約 6 小時，合計約 13 個單位（兩個 runtime 並行可在一晚跑完）。
