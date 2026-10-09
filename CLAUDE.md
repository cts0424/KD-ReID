# CLAUDE.md

本檔給 Claude（以及任何新加入的協作者）快速掌握這個專案。回覆使用者時請用**繁體中文**；程式碼、註解、commit message 用英文。

## 專案目標

**知識蒸餾（Knowledge Distillation）應用於行人重識別（Person ReID）**：把大型 teacher（預設 ResNet-50）的能力轉移到輕量 student（預設 ResNet-18，也支援 MobileNetV3），讓 student 在參數量/推論速度大幅下降的情況下，mAP / Rank-1 盡量逼近 teacher。

- 主要資料集：Market-1501（DukeMTMC-reID 已支援相同格式）
- 評估指標：mAP、CMC Rank-1/5/10（Market 標準協定：排除同 ID 同相機的 gallery）
- 程式碼從零以 PyTorch 撰寫，不依賴 FastReID 等框架

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
```

## 專案結構

```
kdreid/
  config.py          YAML 載入：`_base_` 繼承 + CLI `a.b=value` 覆寫（值以 YAML 解析）
  data/
    datasets.py      Market/Duke 檔名解析 → (path, pid, camid)；train pid 重新編號、camid 從 0 起
    sampler.py       RandomIdentitySampler：每個 batch = P 個 ID × K 張（triplet 需要）
    transforms.py    Resize → Flip → Pad+Crop → Normalize → RandomErasing
    build.py         build_loaders(cfg) → (train_loader, test_loader, dataset)；test = query + gallery 串接
  models/
    backbones.py     torchvision ResNet18/34/50/101、MobileNetV3；ResNet 支援 last_stride=1
    reid_net.py      ReIDNet：backbone → GAP → BNNeck → classifier
  losses/
    reid.py          Label-smoothing CE、batch-hard Triplet
    kd.py            logit_kd（Hinton）、FeatureKD（投影後 L2）、similarity_kd（batch 內相似度矩陣）、DistillLoss（加權總和）
  engine/
    trainer.py       Trainer：AMP、warmup+step LR、自動續跑、依 mAP 存 best
    evaluator.py     特徵抽取（含 flip TTA）、eval_market（mAP/CMC）
  utils.py           seed、logger、原子化 checkpoint
tools/train.py, tools/test.py    CLI 入口
configs/             base.yaml + 各實驗設定（只寫與 base 不同之處）
notebooks/colab_setup.ipynb      Colab 環境設定與訓練流程
tests/test_smoke.py  CPU 端到端測試（teacher → KD student、續跑、評估正確性）
```

## 關鍵介面與約定

- **`ReIDNet.forward()` 回傳 dict**，KD 依賴這個介面，不要改成回傳 tensor：
  - `feat_map` (B×C×h×w)、`feat`（GAP 後、BNNeck 前，給 triplet 與 KD）、`bn_feat`（BNNeck 後，給檢索）、`logits`（僅 train mode，eval 為 `None`）
- **Teacher 永遠是 eval mode + 凍結參數**；需要 logits 時由 `Trainer._teacher_forward` 手動呼叫 `teacher.classifier(bn_feat)`。
- **KD 損失權重放在 `cfg.kd.losses`**：`{logit, feature, similarity}`，權重 0 = 關閉。新增 KD 方法時：
  1. 在 `losses/kd.py` 寫成函式或 `nn.Module`
  2. 在 `DistillLoss.__init__/forward` 加一個權重與 key（回傳 dict 的 key 以 `kd_` 開頭，會自動出現在 log）
  3. 若有可學參數（如 projector），放在 `DistillLoss` 內——Trainer 會自動加入 optimizer 並存進 checkpoint
  4. 在 `tests/test_smoke.py` 的 KD 端到端測試中打開它
- Teacher 與 student 的 `num_classes` 相同（同一資料集的訓練 ID 數），logit KD 才能直接對齊。
- Teacher 與 student 特徵維度不同（2048 vs 512）：`similarity_kd` 與維度無關；`FeatureKD` 透過線性 projector 對齊。
- 新實驗：在 `configs/` 新增一個 `_base_: base.yaml` 的檔案，只寫差異，並把 `output_dir` 設成 Drive 上獨立的資料夾。不要在程式碼裡寫死超參數。

## 開發規範

- 每次改動後跑 `python -m pytest -q` 與 `ruff check .`，兩者都要通過才 commit。
- 不要把資料集、`*.pth`、`outputs/` 加進 git（已在 `.gitignore`）。
- 新增依賴寫進 `pyproject.toml`；避免引入 Colab 上需要編譯的套件。
- 實驗結果（mAP / R1、config、commit hash）記錄在 `docs/experiments.md`，方便論文整理。

## 研究路線圖

- [x] 階段 0：repo、Colab 環境、baseline 程式骨架、smoke tests
- [ ] 階段 1：在 Market-1501 訓練 teacher R50 與 student R18 baseline，確認數字接近文獻（BoT R50 約 mAP 85–86%、R1 約 94%）
- [ ] 階段 2：KD 消融——logit / feature / similarity 各自與組合、溫度 T、權重
- [ ] 階段 3：更強 teacher（ResNet-101、ViT/TransReID 類）與更輕 student（MobileNetV3）
- [ ] 階段 4：跨資料集驗證（DukeMTMC-reID）、推論速度與參數量對比
