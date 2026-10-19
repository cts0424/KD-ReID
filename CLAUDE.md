# CLAUDE.md

本檔給 Claude（以及任何新加入的協作者）快速掌握這個專案。回覆使用者時請用**繁體中文**；程式碼、註解、commit message 用英文。

## 專案目標

**知識蒸餾（Knowledge Distillation）應用於行人重識別（Person ReID）**：把大型 teacher（ResNet-50、CLIP-ReID）的能力轉移到輕量 student（**兩個尺寸：OSNet x0.25（0.2M，極輕量）與 OSNet x1.0（2.2M，接近 teacher 精度）**；ResNet-18 為傳統基準），讓 student 在參數量/推論速度大幅下降的情況下，mAP / Rank-1 盡量逼近 teacher。

- 論文資料集：Market-1501、CUHK03-NP（`cuhk03np` = detected，`cuhk03np_labeled`）、MSMT17（申請中）。**不使用 DukeMTMC-reID**（已撤回；CVPR / ECCV 2026 指引不建議使用），程式仍支援，只作內部參考
- 評估指標：mAP、CMC Rank-1/5/10（Market 標準協定：排除同 ID 同相機的 gallery）
- 程式碼從零以 PyTorch 撰寫，不依賴 FastReID 等框架
- **KD-ReID 是獨立的研究**：使用者自有的跨相機系統只用來做輔助佐證（數字記在 experiments.md，不進論文主表），程式與實驗都不依賴它；外部模型（如 CLIP-ReID）一律用官方公開權重，確保可重現

### 實驗的三個基準（每個 KD 結果都要對照這三個數字）
| 設定 | config | 角色 |
|---|---|---|
| Teacher ResNet-50 | `configs/teacher_r50.yaml` | 上界 |
| Student ResNet-18 單獨訓練 | `configs/student_r18_baseline.yaml` | 下界 |
| Student ResNet-18 + KD | `configs/student_r18_kd.yaml` | 本研究方法 |

## 開發工作流程（程式碼 vs. 執行分離）

```
 寫程式（Claude / 本機 IDE）──push──▶ GitHub (cts0424/KD-ReID, 唯一真實來源)
                                          │
                                   git pull（notebook 第 3 步）
                                          ▼
                     Colab GPU：只負責訓練 / 評估，讀寫 Drive
```

- **改程式**：在 Claude 對話裡（Claude 會直接 commit + push），或本機 clone 後用 IDE 改。改完跑 pytest + ruff 再 push。
- **跑實驗**：Colab 開 `notebooks/colab_setup.ipynb`（或 Drive 上的 `KD_ReID.ipynb`），執行第 3 步就會 `git pull` 拿到最新版。
- **不要在 Colab 裡改 `kdreid/` 程式碼**：runtime 重置就會消失，也會和 GitHub 分岔。Colab 只改指令列的 config 覆寫參數。
- **實驗結果回報**：把 `log.txt` 最後幾行（或 mAP / R1）貼回 Claude，記錄到 `docs/experiments.md`。

## 執行環境：Google Colab

- **訓練一律在 Colab GPU 上**，用 `notebooks/colab_setup.ipynb` 建環境（掛 Drive → clone → `pip install -e .` → 解壓資料集 → pytest）。
- **不要在 Colab 重裝 torch**：Colab 內建 CUDA 版本，`pyproject.toml` 只寫下限。
- **資料集**：zip 放 Drive `MyDrive/KD-ReID/datasets/`（目前是 Kaggle 版 Market-1501 `archive.zip`，檔名不限），每次 runtime 複製到 `/content/data` 解壓後再讀（直接從 Drive 讀小圖極慢）。`build_dataset` 會在 `data.root` 下往下 3 層自動找 `bounding_box_train` 所在資料夾，所以不用管 zip 解出來的頂層資料夾叫什麼。
- **輸出**：寫到 Drive `MyDrive/KD-ReID/outputs/<實驗名>/`，內含 `log.txt`、`config.yaml`、`last.pth`、`best.pth`。
- **斷線續跑**：`train.resume: true`（預設），重跑同一個指令會從 `last.pth` 接續。checkpoint 以原子方式寫入，不會因斷線損壞。
- 本機 / Claude 的雲端環境沒有 GPU：只跑 `pytest`（CPU、合成資料、幾秒鐘），不要嘗試真正訓練。

## 常用指令

```bash
pip install -e ".[dev]"            # 安裝（含 pytest、ruff）
python -m pytest -q                # smoke tests（合成資料，無需下載）
ruff check . && ruff format .      # lint + 格式化（line-length 100）

# 訓練：config + 任意 key.sub=value 覆寫
python tools/train.py --config configs/teacher_r50.yaml
python tools/train.py --config configs/student_r18_kd.yaml kd.losses.feature=1.0 train.lr=3.5e-4
python tools/test.py  --config <output_dir>/config.yaml --ckpt <output_dir>/best.pth
python tools/train.py --config configs/osnet/1a_baseline.yaml   # OSNet 實驗都在 configs/osnet/

# 部署評估：參數、MACs、延遲（PyTorch / ONNX Runtime）、ONNX 匯出與數值比對
python tools/benchmark.py --models resnet50 resnet18 osnet_x1_0:512 osnet_x0_25:512 --onnx-dir onnx/
python tools/benchmark.py --runs-dir /content/drive/MyDrive/KD-ReID/outputs/osnet   # 已訓練模型
```

## 專案結構

```
kdreid/
  config.py          YAML 載入：`_base_` 繼承 + CLI `a.b=value` 覆寫（值以 YAML 解析）
  data/
    datasets.py      Market/Duke 檔名解析 → (path, pid, camid)；train pid 重新編號、camid 從 0 起
    sampler.py       RandomIdentitySampler：每個 batch = P 個 ID × K 張（triplet 需要）
    transforms.py    Resize → Flip → Pad+Crop → [ColorJitter] → Normalize → RandomErasing
    build.py         build_loaders(cfg) → (train_loader, test_loader, dataset)；test = query + gallery 串接
  models/
    backbones.py     torchvision ResNet18/34/50/101、MobileNetV3；ResNet 支援 last_stride=1
    osnet.py         OSNet x0.25/0.5/0.75/1.0，層名與 torchreid 相同（官方 ImageNet 權重可直接載入，
                     已驗證輸出與 torchreid 逐位元一致）；fc 頭對應 ReIDNet.embed
    reid_net.py      ReIDNet：backbone → GAP → [embed: Linear-BN-ReLU] → BNNeck → classifier；build_model / load_weights
    clipreid.py      CLIP-ReID ViT-B/16 影像分支（凍結 teacher 專用），直接載入官方 checkpoint；內部轉換 CLIP 正規化；
                     logit_head=text 時另載入 prompt learner + 文字編碼器，logits = 投影 CLS · 身份文字原型
  losses/
    reid.py          Label-smoothing CE、batch-hard Triplet
    kd.py            logit / dkd / feature / cosine / similarity / simdist / rkd / attention，DistillLoss 加權總和（表格見檔頭 docstring）
  engine/
    trainer.py       Trainer：AMP、warmup+step LR、多 teacher、凍結 backbone 暖身、自動續跑、依 mAP 存 best
    evaluator.py     特徵抽取（含 flip TTA）、eval_market（mAP/CMC）
  benchmark.py       部署指標：參數、MACs、PyTorch/ONNX Runtime 延遲、ONNX 匯出
  utils.py           seed、logger（每行即關檔，Drive 可同步）、原子化 checkpoint
tools/train.py, tools/test.py, tools/benchmark.py, tools/eval_matrix.py, tools/export_package.py, tools/check_teacher.py    CLI 入口
deploy/              獨立部署檔：osnet_reid.py（由 kdreid/models/osnet.py 產生的獨立模型定義）、reid_infer.py、README 範本
configs/             base.yaml + 各實驗設定（只寫與 base 不同之處）
configs/osnet/       OSNet 實驗，檔名 = plan_osnet.md 的實驗編號（1a、2c…）；`_` 開頭為共用基底
configs/clip/        階段 4：CLIP-ReID teacher（4a DKD、4b logit KD、teacher_eval 給 check_teacher）
configs/cuhk03/      階段 5：同一批主表設定換成 CUHK03-NP（`_base_` 指向 Market 版，只改 data.name / output_dir / teacher）
configs/osnet_x1/    plan 6：OSNet x1.0 學生（同 B\* 配方；6a 無 KD、6b ← R50 DKD、6c ← CLIP DKD）
configs/plan7/       plan 7 新方法：7a 同家族助教（6c → x0.25）、7b / 7c CLIP-ReID 文字原型 logits（x0.25 / x1.0）
configs/plan8/       plan 8 = 方法 4（無標註影像上的蒸餾，docs/method4_design.md）：8a 對照（Market 影像）、8b CUHK03 訓練集影像（UDA）
notebooks/colab_setup.ipynb      Colab 環境設定與訓練流程
tests/               CPU 測試：端到端訓練、續跑、評估正確性、OSNet、各 KD 損失性質、多 teacher、所有 config 可建模
```

## 關鍵介面與約定

- **`ReIDNet.forward()` 回傳 dict**，KD 依賴這個介面，不要改成回傳 tensor：
  - `feat_map` (B×C×h×w)、`feat`（GAP 後、BNNeck 前，給 triplet 與 KD）、`bn_feat`（BNNeck 後，給檢索）、`logits`（僅 train mode，eval 為 `None`）
- **Teacher 永遠是 eval mode + 凍結參數**；需要 logits 時由 `Trainer._teacher_forward` 手動呼叫 `teacher.classifier(bn_feat)`（CLIP-ReID teacher 在 eval 也直接回傳 logits）。換新 teacher 前先跑 `tools/check_teacher.py` 確認精度與類別順序對齊。
- **KD 損失權重放在 `cfg.kd.losses`**：`{logit, dkd, feature, cosine, similarity, simdist, rkd, attention}`，權重 0 = 關閉；拼錯 key 會直接報錯。新增 KD 方法時：
  1. 在 `losses/kd.py` 寫成函式或 `nn.Module`
  2. 加進 `DistillLoss.KEYS` 與 `forward`（回傳 key 以 `kd_` 開頭，會自動出現在 log）
  3. 若有可學參數（如 projector），放在 `DistillLoss` 內——Trainer 會自動加入 optimizer 並存進 checkpoint
  4. 在 `tests/test_osnet_kd.py` 補性質測試，並在多 teacher 端到端測試中打開它
- **多 teacher**：用 `teachers:`（list）取代 `teacher:`。每個 teacher 可帶 `weight` 與 `kd:`（只對該 teacher 覆寫全域 `kd`）。log key 會加 `t0_`、`t1_` 前綴。單一 `teacher:` 的舊設定完全相容。
- **無標註蒸餾（方法 4）**：`unlabeled: {enabled, dataset 或 folder, root, batch_size, weight, losses}`。每個 iteration 另取一個無標註 batch（`dataset` 只用它的訓練集影像、丟掉標籤），只算不需要標籤的 KD 項（`DistillLoss.LABEL_FREE`，不能用 dkd / CE / triplet），log key 以 `u_` 開頭；projector 存在 checkpoint 的 `kd_unl`。用目標域訓練集影像的結果是 UDA，不可和 zero-shot 混報。
- **`kd.losses` 一律整個取代、不合併**（`config.REPLACE_KEYS`）：子設定寫 `losses: {dkd: 1.0}` 就是「只用 DKD」，不會繼承父設定的 logit / similarity。每個 teacher 的 `kd.losses` 同理取代全域清單。其他 key（temperature、dkd.beta…）照常深度合併。
- Teacher 與 student 的 `num_classes` 相同（同一資料集的訓練 ID 數），logit KD 才能直接對齊。
- Teacher 與 student 特徵維度不同（2048 vs 512）：similarity / simdist / rkd 與維度無關；`FeatureKD` 透過線性 projector 對齊；attention 只看空間圖（R50 與 OSNet 在 256×128 輸入下都是 16×8）。
- **資料集分割大小會自動核對**（`datasets.STANDARD_STATS`）：讀到的人數 / 張數和官方不符就報錯（測試用 `KDREID_SKIP_DATASET_CHECK=1` 關閉）。log 第一行會印出實際讀到的資料夾。
- **評估協定**：cosine 距離、**不使用 re-ranking**（repo 裡沒有任何 re-ranking 程式）、預設 flip TTA（原圖 + 水平翻轉特徵相加）；Market 標準協定（排除同 ID 同相機，pid −1 的 junk 不進 gallery，gallery 15913 張與 torchreid 一致）。和文獻比較時務必對照「無 RK」的數字。
- **跨域（zero-shot）評估**：`tools/eval_matrix.py --runs ... --datasets market1501:ROOT dukemtmc:ROOT`，用 last.pth、同時報有 / 無 flip。看相對排名與保留率（跨域 mAP ÷ 同域 mAP），不看絕對分數。
- **報告數字一律用最後一個 epoch**（log 中 `Final (epoch N)` 那行）。`best.pth` 是在測試集上挑的，只能當參考，拿來報告等於偷看測試集。
- **打包給外部系統**：`tools/export_package.py --run <output_dir>`（OSNet 或 ResNet） → ONNX + 權重 + `reid_infer.py`（selftest/extract/search）+ README。改了 `kdreid/models/osnet.py` 的層結構時，要同步更新 `deploy/osnet_reid.py`（`tests/test_export_package.py` 會檢查兩者輸出一致、前處理與評估完全相同）。
- **CPU 延遲以 ONNX Runtime 為準**：PyTorch eager 對 OSNet 這類多個小型 depthwise 運算的網路很不利，會低估它的速度優勢。
- 新實驗：在 `configs/` 新增一個 `_base_: base.yaml` 的檔案，只寫差異，並把 `output_dir` 設成 Drive 上獨立的資料夾。不要在程式碼裡寫死超參數。

## 開發規範

- 每次改動後跑 `python -m pytest -q` 與 `ruff check .`，兩者都要通過才 commit。
- 不要把資料集、`*.pth`、`outputs/` 加進 git（已在 `.gitignore`）。
- 新增依賴寫進 `pyproject.toml`；避免引入 Colab 上需要編譯的套件。
- 實驗結果（mAP / R1、config、commit hash）記錄在 `docs/experiments.md`，方便論文整理。

## 研究路線圖

**主目標：做出最好的 OSNet 學生模型，兩個尺寸並行：x0.25（0.2M，最佳 F\* 85.0）與 x1.0（2.2M，最佳 6c 89.5）。詳細計畫、實驗編號與決策規則見 `docs/plan_osnet.md`。**
R50 → R18 是「傳統同質蒸餾」基準；OSNet x0.25（以及之後的 CLIP-ReID teacher）是目標方向，比較時用「差距補回率」與精度對成本，而不是只比絕對 mAP。

- [x] repo、Colab 環境、baseline 程式骨架、smoke tests
- [x] 傳統基準（Market-1501）
  - [x] teacher R50：mAP 86.6 / R1 95.0（`MyDrive/KD-ReID/outputs/teacher_r50/best.pth`）
  - [x] student R18 baseline：mAP 80.6 / R1 92.0
  - [x] student R18 + KD（logit + similarity）：mAP 84.8 / R1 93.6，補回 70% 差距——之後當 OSNet 的助教模型（plan 3a）
- [x] OSNet 計畫階段 0：OSNet x0.25/x1.0 實作、`tools/benchmark.py`、新 KD 損失（AT / DKD / RKD / simdist / 多 teacher）、configs/osnet
- [x] OSNet 計畫階段 1：強 baseline B* = 1e（lr 1e-3、240 ep）：mAP 77.0 / R1 90.6（論文 77.8）
- [x] OSNet 計畫階段 2：直接 KD R50 → OSNet：D* = 2d（logit only）mAP 82.7 / R1 93.1，補回 59%；similarity / attention 無貢獻
- [x] OSNet 計畫階段 3：**F\* = 3d（DKD，85.0 / 94.0，3 seeds）**。DKD 普遍有效（R18-DKD 86.2，+1.4）；β ≥ 8 飽和；LS 解釋約一半；助教 / 雙 teacher 無效。結論見 plan_osnet.md「階段 3 結論」
- [x] 跨域 zero-shot（Market → Duke，**僅內部參考**，論文改用 CUHK03-NP / MSMT17）：KD 增益完整保留、無過擬合；**F\* Duke mAP 20.6 ± 0.3（3 seeds）> teacher 19.7**；R18-DKD 跨域沒有變好（17.9 vs R18 KD 18.5），DKD 的跨域優勢只在 OSNet 成立
- [x] 打包 F\*（使用者系統：C01→C02 84.9 mAP，Panorama 68.7 mAP，仍落後 CLIP-ReID）：`MyDrive/KD-ReID/packages/kdreid-osnet025-dkd-v1.zip`（匯出時重測 84.98 / 93.94；ONNX 與 PyTorch 差 2.6e-6）。使用者在自己的系統測試中
- [ ] **OSNet 計畫階段 5：論文驗證（問題 1）**，詳見 plan_osnet.md「階段 5」
  - [x] 5a CUHK03-NP 主表（`configs/cuhk03/`）：**OSNet 部分完成（2026-10-13）**：teacher 61.4 / B\* 49.3 / D\* 55.8（補回 54%）/ **F\* 60.9（補回 96%，vs D\* +5.1）**，排名與 Market 一致。R18：baseline 50.6 / KD 60.5 / **DKD 62.4（> teacher）**。第一輪誤讀到 Duke 的結果在 `outputs/duke_internal/`（內部參考）
  - [x] 5b Market seeds（3 seeds，2026-10-14）：teacher 86.5 ± 0.1 / B\* 77.4 ± 0.4 / D\* 82.6 ± 0.1（補回 57%）/ **F\* 85.0 ± 0.0（補回 83%，vs D\* +2.4）**
  - [x] 5c 跨域 Market → CUHK03-NP（3 seeds，2026-10-14）：**F\* 10.2 ± 0.9 > D\* 8.4 ± 0.5 > B\* 6.1 ± 0.1 > teacher 5.5 ± 0.6**（保留率 12.0 / 10.1 / 7.9 / 6.4%）；R18-DKD 7.6 未勝 R18-KD 8.1（單 seed），與 Duke 型態一致
  - [x] **5a 補完（2026-10-19）**：CUHK03 3 seeds：B\* 49.5 ± 0.3 / D\* 55.6 ± 0.2（補回 51%）/ **F\* 61.0 ± 0.3（97%，vs D\* +5.4）**；x1.0 無 KD 67.0（> R50 teacher 61.4）、x1.0 ← R50 DKD 69.2。CLIP-ReID 官方沒有 CUHK03 權重 → x1.0 ← CLIP 的第二資料集改用 MSMT17
  - [ ] 5d MSMT17（申請中；到手後寫讀取程式、先定配方；CLIP-ReID 官方有 MSMT17 權重）
  - [x] 問題 2 前置診斷：**瓶頸在 teacher**（R50 在實際場景落後 CLIP-ReID 7.9–16.3 mAP；student 跨相機 ≥ teacher）。待補實際場景的 B\* / D\*
  - [x] 固定硬體部署表（見 paper_tables.md 表 6）
- [ ] 問題 2：跨相機系統的強模型 → **CLIP-ReID 當 teacher**（階段 4，`configs/clip/`）
  - [x] CLIP-ReID 的 ID 頭幾乎是平的（訓練集 top-1 78.6%、最大機率 0.008）→ teacher `logit_scale` 8.26（temperature scaling）
  - [x] x0.25：CLIP teacher 反而變差（4a DKD 83.45 ± 0.07 < F\* 85.0）；使用者系統打平；4c（只用特徵關係）78.9
  - [x] **容量對照（plan 6，`configs/osnet_x1/`）**：x1.0 ← R50 DKD 88.3、← CLIP DKD **89.5**（≈ CLIP-ReID 89.9，參數 1/40）。CLIP − R50 從 −1.55（x0.25）翻成 +1.2（x1.0）→ 容量是瓶頸
  - [x] **使用者系統（2026-10-17）**：6c 三個場景 90.91 / 93.35 / 71.73，比 6b 高 5.4 / 1.7 / 1.7、比 F\* 平均高 5.9 / 4.2 / 2.8（F\* seed 標準差 0.1–0.5），和 CLIP-ReID 的差距縮小 33–48%。RTX 3050 ONNX CUDA batch 32：6c 3.4 ms/張，CLIP-ReID 11.0 ms/張（約 3 倍快）→ **問題 2 以 x1.0 學生達成**（限制：單 seed）
  - [x] **x1.0 跨域（2026-10-17）**：Market → CUHK03-NP 6c **16.7** vs 6b 9.4、F\* 10.2 ± 0.9、R50 teacher 5.5 → 跨域增益來自 CLIP teacher，不是學生變大。問題 2 事前三項標準全部達成（單 seed）
  - [x] 6c seed 1：89.5 / 95.2 → **6c = 89.5 ± 0.0（2 seeds）**；整個管線 FPS 已測（YOLO 為瓶頸，見 experiments.md）
  - [x] **x1.0 2×2 補齊（2026-10-18）**：6a 無 KD **85.8**（論文 84.9）、6b = 88.3 ± 0.0（2 seeds）。增益 +2.5（R50）/ +3.7（CLIP）；以 CLIP 計補回率 x1.0 90% vs x0.25 48%
  - [x] **跨域參考（2026-10-18）**：**CLIP-ReID 本身 Market → CUHK03-NP 39.2**（保留率 43.6%），遠高於所有學生；6c 17.1 ± 0.6、6b 8.8 ± 0.8、6a 6.7（2 / 2 / 1 seeds）。6c 同域達 CLIP 的 99.6%、跨域只有 44% → 跨域是學生與 CLIP 最大的剩餘差距（方法 4：無標註目標域蒸餾的動機）
  - [x] **部署表（2026-10-18，T4 + 2 vCPU）**：ORT CPU batch 1：CLIP 255 ms / R50 81 / R18 40 / **x1.0 22** / **x0.25 3.2**；GPU fp16 b32 每張：CLIP 2.30 / x1.0 0.83 / x0.25 0.54（OSNet 在 GPU 受 kernel 啟動成本限制）。OSNet 優勢在 CPU / 邊緣
- [ ] **Plan 8 = 方法 4：無標註影像上的蒸餾**（`configs/plan8/`，設計與事前判斷標準見 `docs/method4_design.md`）：程式完成（2026-10-19）；8a（對照，Market 影像）、8b（CUHK03 訓練集影像，UDA）待跑；成功標準 8b 跨域 ≥ 22.6 且 Market ≥ 89.2
- [x] **Plan 7 新方法（2026-10-18，全部負結果，不採用）**（`configs/plan7/`）：7a 同家族助教 6c → x0.25 **83.6**（< F\* 85.0，≈ 4a 83.45）；7b x0.25 ← CLIP 文字原型 **83.8**；7c x1.0 ← CLIP 文字原型 **88.4**（< 6c 89.5，≈ 6b 88.3）。文字原型訓練集 top-1 100% 卻是較差的 teacher → teacher 準確度 ≠ 蒸餾品質。各 1 seed
