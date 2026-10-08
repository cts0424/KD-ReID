# 實驗紀錄

每跑完一個實驗新增一列。數字取 `best.pth` 對應的評估結果（含 flip TTA）。

| 日期 | 實驗 | Teacher | Student | KD 設定 | mAP | R1 | commit | 備註 |
|---|---|---|---|---|---|---|---|---|
| | teacher_r50 | — | ResNet-50 | — | | | | 上界 |
| | student_r18_baseline | — | ResNet-18 | — | | | | 下界 |
| | student_r18_kd | ResNet-50 | ResNet-18 | logit 1.0, sim 1.0, T=4 | | | | |
