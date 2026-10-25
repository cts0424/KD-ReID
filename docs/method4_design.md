# 方法 4：無標註影像上的蒸餾（plan 8；8a / 8b 完成，Market 下降已解決 2026-10-24）

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

## 結果

| | Market mAP | → CUHK03-NP mAP | 對照事前標準 |
|---|---|---|---|
| 6c（基準，2 seeds） | 89.5 ± 0.0 | 17.1 ± 0.6 | – |
| **8b（UDA）** | 88.7（−0.8） | **31.7（+14.6，補回 66%）** | 跨域：**達成**（≥ 22.6）；Market：**未達成**（門檻 ≥ 89.2） |
| 8a（對照，Market 影像） | 89.5（±0） | 15.3（−1.8） | 沒有接近 8b → 「資料而非損失」成立 |

判讀：部分成功，因果關係清楚。跨域增益（+14.6）和 Market 下降（−0.8）都來自目標域影像，新的損失本身沒有作用（8a 跨域 15.3、Market 89.5）。下一步若要消除 Market 下降，候選是 domain-specific BN，需事先寫好判斷規則。

## 下一步：Market 下降的修正（2026-10-21，跑之前寫下）

**先做不用重新訓練的診斷**。8b 的 BN running statistics 是 Market 和 CUHK03 影像混合估出來的（每個 iteration 64 張 Market + 32 張 CUHK03）。如果 −0.8 主要來自這個混合，那麼只要用某個域自己的訓練影像重新估 BN 統計量（AdaBN，權重完全不動），就能各自恢復。

工具：`tools/eval_matrix.py --bn-recal`。每個測試資料集多跑一次：先用該資料集**訓練集**的影像（不用標籤）重新估計所有 BN 的 running mean / var，評估完再還原。評估對象：6c_s1、8a、8b。

事前判斷規則（只用 Market 的數字做決定，CUHK03 的數字照實報告、不拿來挑設定）：

1. **8b + Market BN 統計量 ≥ 89.2**：下降來自 BN 統計量。修正方式就是「每個域各自一套 BN 統計量」，推論時用該域的統計量，不需要重新訓練；8b 的 Market 欄改報這個數字，並註明。
2. **< 89.2**：BN 統計量不是主因（或不只是），改試訓練時的方法：在無標註 batch 上不更新 BN 統計量（domain-specific BN 的簡化版），重新訓練一次（約 6 h）。

**必須同時檢查的對照（避免高估方法 4）**：AdaBN 本身就是一種已知的 UDA 方法。所以 CUHK03 的比較要用同樣的 BN 處理：
- 6c + CUHK03 BN 統計量（只做 AdaBN、沒有方法 4）vs 8b + CUHK03 BN 統計量。
- 只有後者明顯更高（≥ +5 mAP）時，才能說方法 4 的增益超過單純的 BN 適應。若 6c + AdaBN 就接近 31.7，方法 4 的貢獻要重新評估。

**補充（2026-10-21，第一輪之後、看 8b 結果之前）**：第一輪用乾淨影像重估 BN，6c / 8a 在 Market 上就掉到約 83.8（工具本身有偏差）。規則加一個前提：重估改用訓練資料增強，且 6c 與 8a 在 Market 上重估後必須重現原數字（89.5 ± 0.3），工具通過這個檢查後，才用規則 1 / 2 判斷 8b。若仍無法重現，放棄 AdaBN 診斷，直接走規則 2（訓練時不在無標註 batch 上更新 BN 統計量）。

## AdaBN 診斷結果（2026-10-24）

| run | Market（原 / Market 重估） | CUHK03（原 / CUHK03 重估） |
|---|---|---|
| 6c_s1 | 89.5 / 89.6 | 17.5 / 22.9 |
| 8a | 89.5 / 89.5 | 15.3 / 22.9 |
| **8b** | 88.7 / **90.0** | 31.7 / **33.0** |

- 工具自我檢查通過（6c、8a 重估後重現 89.5）。
- 規則 1 成立：Market 下降來自混合的 BN 統計量，用每域各自的統計量即可消除（90.0，不需要重新訓練）。
- 對照通過：同樣做 AdaBN，8b 比 6c 高 +10.1（門檻 +5），方法 4 的增益不是單純 BN 適應。
- **方法 4 最終**：Market 90.0、CUHK03 33.0（補回跨域差距 72%）。兩項事前標準都達成，Market 的修正依規則 1、於第一次未達標後決定，論文中要說明。單 seed。
- **seed 1（2026-10-25）重現**：Market 90.0、CUHK03 33.8 → **方法 4 = Market 90.0 ± 0.0、CUHK03 33.4 ± 0.6（2 seeds）**，補回跨域差距 74%。

