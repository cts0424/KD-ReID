# 論文表格草稿

從 `docs/experiments.md` 整理（2026-10-18）。全部數字：最後一個 epoch（`Final`）、cosine 距離、flip TTA、**無 re-ranking**。`n` = seed 數；有 n ≥ 2 的寫平均 ± 標準差。**待補**表示論文前必須補上的格子，最後一節是清單。

補回率 = (學生 − baseline) ÷ (teacher − baseline)，baseline 是同架構、同配方、不用 KD 的學生。

---

## 表 1：Market-1501 主表（蒸餾方法比較）

| 學生（參數） | teacher | KD | mAP | R1 | n | 補回率 |
|---|---|---|---|---|---|---|
| ResNet-50 teacher（23.5M） | – | – | 86.5 ± 0.1 | 94.9 ± 0.2 | 3 | – |
| OSNet x0.25（0.2M）B\* | – | – | 77.4 ± 0.4 | 90.8 ± 0.4 | 3 | – |
| OSNet x0.25 D\* | R50 | logit KD | 82.6 ± 0.1 | 93.2 ± 0.2 | 3 | 57% |
| **OSNet x0.25 F\*** | R50 | **DKD** | **85.0 ± 0.0** | **94.0 ± 0.1** | 3 | **83%** |
| ResNet-18（11.2M） | – | – | 80.6 | 92.0 | 1 | – |
| ResNet-18 | R50 | logit + similarity KD | 84.8 | 93.6 | 1 | 70% |
| ResNet-18 | R50 | DKD | 86.2 | 94.4 | 1 | 93% |

- 文字重點：DKD 在兩種學生上都勝過傳統 KD（OSNet +2.4、R18 +1.4），補回率都提高約 25 個百分點；F\* 的參數是 R50 的 1/116。
- R18 的 teacher 是單 seed 86.6（補回率用 86.6 計）。

## 表 2：CUHK03-NP（detected）主表

| 學生 | KD | mAP | R1 | n | 補回率 |
|---|---|---|---|---|---|
| ResNet-50 teacher | – | 61.4 | 62.4 | 1 | – |
| OSNet x0.25 B\* | – | 49.3 | 52.4 | 1 | – |
| OSNet x0.25 D\* | logit KD | 55.8 | 58.4 | 1 | 54% |
| **OSNet x0.25 F\*** | **DKD** | **60.9** | **63.9** | 1 | **96%** |
| ResNet-18 | – | 50.6 | 54.9 | 1 | – |
| ResNet-18 | logit + similarity KD | 60.5 | 63.4 | 1 | 92% |
| ResNet-18 | DKD | 62.4 | 65.1 | 1 | 109% |
| OSNet x1.0 | R50 DKD | **待補** | | | |

- 和 Market 相同的配方，不另外調參。排名 B\* < D\* < F\* 重現，DKD − logit KD 更大（+5.1）。
- 補回率 > 100% 是因為分母小（10.8），要同時報絕對差距；寫成「與 teacher 相當」。
- **待補**：seeds（至少 F\*、D\*、B\*）；OSNet x1.0 ← R50（6b 配方）。x1.0 ← CLIP 需要 CLIP-ReID 的 CUHK03 權重（待確認官方是否釋出）。

## 表 3：teacher 強度 × 學生容量（Market-1501）

| 學生 | 無 KD | ← R50（86.5）DKD | ← CLIP-ReID（89.9）DKD | CLIP − R50 |
|---|---|---|---|---|
| OSNet x0.25（0.2M） | 77.4 ± 0.4（n=3） | 85.0 ± 0.0（n=3） | 83.45 ± 0.07（n=2） | **−1.55** |
| OSNet x1.0（2.2M） | **待補（6a，跑中）** | 88.3（n=1，**s1 跑中**） | **89.5 ± 0.0**（n=2） | **+1.2** |

- 參考：CLIP-ReID ViT-B/16 官方權重 89.9 / 95.4（我們的評估程式重測，與論文一致），約 86M 參數。
- 文字重點：同一個 teacher 和同樣的 DKD，對 x0.25 是負面的、對 x1.0 是正面的 → 強 teacher 只對容量夠的學生有用。6c 與 CLIP-ReID 只差 0.4，參數約 1/40。
- CLIP-ReID 的線性 ID 頭幾乎是平的（訓練集 top-1 78.6%、最大機率 0.008），用溫度縮放 ×8.26（在乾淨訓練集上以 NLL 擬合，不碰測試集）。

## 表 4：跨域 zero-shot，Market-1501 → CUHK03-NP

| 模型（只用 Market 訓練） | 同域 mAP | → CUHK03-NP mAP | n | 保留率 |
|---|---|---|---|---|
| ResNet-50 teacher | 86.5 | 5.5 ± 0.6 | 3 | 6.4% |
| CLIP-ReID teacher（官方權重） | 89.9 | **待補（程式已就緒）** | – | |
| OSNet x0.25 B\* | 77.4 | 6.1 ± 0.1 | 3 | 7.9% |
| OSNet x0.25 D\*（R50 logit KD） | 82.6 | 8.4 ± 0.5 | 3 | 10.1% |
| **OSNet x0.25 F\*（R50 DKD）** | 85.0 | **10.2 ± 0.9** | 3 | **12.0%** |
| OSNet x0.25 ← CLIP DKD（4a） | 83.4 | 11.3 | 1 | 13.5% |
| OSNet x0.25 ← CLIP logit KD（4b） | 83.2 | 10.1 | 1 | 12.1% |
| OSNet x1.0 無 KD（6a） | **待補** | **待補** | | |
| OSNet x1.0 ← R50 DKD（6b） | 88.3 | 9.4 | 1 | 10.6% |
| **OSNet x1.0 ← CLIP DKD（6c）** | 89.5 | **16.7** | 1 | **18.7%** |
| ResNet-18 / + KD / + DKD | 80.6 / 84.8 / 86.2 | 4.4 / 8.1 / 7.6 | 1 | 5.5 / 9.6 / 8.8% |

- 文字重點：(1) 蒸餾出的 OSNet 跨域勝過自己的 R50 teacher（F\* 10.2 vs 5.5）；(2) 學生變大本身不會讓跨域變好（6b 9.4 ≈ F\* 10.2），跨域增益來自 CLIP teacher（6c 16.7）；(3) DKD 的跨域優勢只在 OSNet 上成立（R18-DKD ≤ R18-KD，Duke 內部參考也是同樣型態）。
- 跨域 seed 波動（± 0.5–0.9）比同域大，單 seed 的列只看大致排名。
- **待補**：CLIP-ReID 本身（算 6c 保留了 teacher 多少跨域能力）；6c 第二個 seed 的跨域（s1 已訓練好，只需評估）。

## 表 5：消融與負結果（Market-1501，OSNet x0.25 除非另註）

| 問題 | 設定 | mAP | 對照 | 結論 |
|---|---|---|---|---|
| DKD 的 β | β = 4 / **8** / 16 | 84.5 / **85.0** / 85.0 | – | β ≥ 8 飽和，用原論文預設 |
| 機制：teacher 的 label smoothing | 無 LS teacher + logit KD（3g） | 83.8 | D\* 82.6、F\* 85.0 | 去掉 LS 補回 DKD 優勢約一半 |
| | 無 LS teacher + DKD（3h） | 84.5 | F\* 85.0 | 不疊加 → 兩者作用在同一處（暗知識） |
| 特徵 / 關係型 KD | logit + similarity（2a） | 82.6 | D\* 82.7 | similarity 無貢獻 |
| | + attention transfer（2c） | 82.6 | 2a 82.6 | AT 無貢獻 |
| | CLIP simdist，無 logits（4c） | 78.9 | B\* 77.4 | CLIP 優勢傳不過去 |
| 助教 | R18-KD（84.8）當助教（3a） | 82.6 | D\* 82.7 | 無效 |
| | 6c（OSNet x1.0，89.5）當助教，DKD（7a） | 83.6 | F\* 85.0 | 同家族、更強的助教也無效 |
| 雙 teacher | R50 + R18-KD（3c） | 82.1 | D\* 82.7 | 反而變差 |
| CLIP 文字原型當 teacher | x0.25 ← 原型 DKD（7b） | 83.8 | 4a 83.45、F\* 85.0 | 略好於線性頭，仍輸 R50 |
| | x1.0 ← 原型 DKD（7c） | 88.4 | 6c 89.5 | 訓練集 top-1 100% 的分類器反而是較差的 teacher |

- 除了 3d 系列，消融都是單 seed。雜訊基準：DKD 設定的 seed 標準差 ≤ 0.1（Market 同域），所以 ≥ 0.5 的差距視為真實、≤ 0.1 的視為沒有差別。

## 表 6：部署成本（**待補**，固定硬體）

| 模型 | 參數（推論） | MACs | CPU ONNX Runtime 延遲（batch 1） | GPU 延遲（batch 32，每張） | Market mAP |
|---|---|---|---|---|---|
| ResNet-50 | 23.5M | 待補 | 待補 | 待補 | 86.5 |
| CLIP-ReID ViT-B/16 | ~86M | 待補 | 待補 | 待補 | 89.9 |
| ResNet-18 | 11.2M | 待補 | 待補 | 待補 | 86.2（DKD） |
| OSNet x1.0 | 2.2M | 待補 | 待補 | 待補 | 89.5（6c） |
| OSNet x0.25 | 0.2M | 待補 | 待補 | 待補 | 85.0（F\*） |

- 用 `tools/benchmark.py` 在同一台機器上一次跑完（Colab T4 + 它的 CPU）。CPU 以 ONNX Runtime 為準（PyTorch eager 會低估 OSNet）。
- 使用者系統的速度（RTX 3050，6c 3.4 ms/張 vs CLIP-ReID 11.0 ms/張）是輔助證據，不放主表。
- `benchmark.py --models clipreid_vit_b16` 已支援（隨機權重量大小與速度即可；deploy 參數不含兩個 ID 頭）。

---

## 待補清單（依優先順序）

| # | 項目 | 填哪裡 | 成本 | 狀態 |
|---|---|---|---|---|
| 1 | 6a（x1.0 無 KD） | 表 3、表 4、x1.0 補回率 | ~1.5 h GPU | **跑中** |
| 2 | 6b seed 1 | 表 3 | ~2 h GPU | **跑中** |
| 3 | CLIP-ReID 本身的跨域 | 表 4 | 幾分鐘 GPU | 程式完成（`eval_matrix --teacher-configs`） |
| 4 | 6a、6b_s1、6c_s1 的跨域 | 表 4 | 幾分鐘 GPU | 等 1、2 |
| 5 | 部署表 | 表 6 | < 1 h | 程式完成（含 CLIP-ReID） |
| 6 | CUHK03 上的 x1.0 ← R50 | 表 2 | ~1.5 h GPU | 可以排 |
| 7 | CUHK03 seeds（F\*、D\*、B\* 各 2 個） | 表 2 | ~6 h GPU | 視單位 |
| 8 | MSMT17 全套 | 新表 | 25–35 單位 | 資料申請中 |
| 9 | 確認 CLIP-ReID 是否有 CUHK03 權重 | 表 2 | – | 待查 |
