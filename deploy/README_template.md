# {{name}}

行人重識別（Person ReID）模型：輸入一張行人裁切圖，輸出一個 {{dim}} 維特徵向量；兩張圖的特徵越相似（cosine 越高），越可能是同一個人。

- 架構：{{backbone}}，{{params}} 參數、{{gmacs}} GMACs（256×128 輸入）
- 訓練：Market-1501，知識蒸餾（teacher：{{teachers}}；損失：{{kd_losses}}）
- 來源：KD-ReID repo commit `{{commit}}`，實驗 `{{run}}`，打包於 {{created}}

## 準確率（Market-1501 同域，cosine 距離，**無 re-ranking**）

| 推論方式 | mAP (%) | Rank-1 (%) | Rank-5 (%) |
| --- | --- | --- | --- |
{{accuracy_rows}}

flip TTA = 原圖與水平翻轉圖的特徵相加，推論時間約 2 倍。

## 快速開始

```bash
pip install -r requirements.txt
python reid_infer.py selftest                        # 確認在這台機器上輸出正確
python reid_infer.py extract --images gallery/ --out feats        # 資料夾 -> feats.npy + feats.json
python reid_infer.py search --query q.jpg --gallery gallery/ --topk 10
python reid_infer.py --backend torch selftest        # 改用 PyTorch（需安裝 torch）
```

## 輸入與輸出規格

| 項目 | 規格 |
| --- | --- |
| 輸入 | `images`：float32，N×3×256×128（NCHW），batch 大小可變 |
| 前處理 | RGB → 雙線性縮放到 256×128（高×寬，不保持比例）→ ÷255 → 減 ImageNet mean `[0.485, 0.456, 0.406]`、除 std `[0.229, 0.224, 0.225]` |
| 輸出 | `features`：float32，N×{{dim}}，**尚未正規化** |
| 後處理 | L2 正規化後，相似度 = 內積（cosine，範圍 −1 到 1） |

前處理必須和上表一致（尤其是 RGB 順序與 normalize），否則準確率會明顯下降。`reid_infer.py` 的 `preprocess()` 是參考實作。

## 檔案

| 檔案 | 用途 |
| --- | --- |
| `model.onnx` | ONNX（opset 17），可用 ONNX Runtime、TensorRT、OpenVINO 等載入 |
| `weights.pt` + `osnet_reid.py` | PyTorch 權重與獨立模型定義（不需要訓練程式碼） |
| `reid_infer.py` | 自我檢查、抽特徵、以圖搜圖 |
| `selftest.npz` | 固定輸入與參考輸出，用來驗證新環境 |
| `metadata.json` | 前處理、輸出規格、準確率、驗證數據的機器可讀版本 |

## 使用限制（請先讀）

- **只在 Market-1501 上訓練與測試過**（6 台相機、校園場景、夏季白天）。在你的攝影機、角度、光線、解析度下，準確率可能明顯較低；跨資料集測試尚未完成。
- **沒有通用的判定門檻**：「多少相似度算同一人」必須用你自己場景中已標註的配對資料來挑選。
- 輸入應是**偵測器裁切好的單一行人**（例如 YOLO 的 person 框）；不適用於車輛或其他物體。
- 目前只跑過單一 random seed；數字是研究階段結果，不是產品保證。
- Market-1501 為學術研究資料集，商業使用前請自行確認資料集授權。
