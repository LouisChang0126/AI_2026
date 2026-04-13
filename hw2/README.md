# AI HW2 — SimCLR Self-Supervised Learning on CIFAR-10

NYCU Artificial Intelligence Spring 2026 — Project #2

使用 SimCLR 對比學習框架，在 CIFAR-10 上以 ResNet-18 為 backbone 進行自監督表徵學習，並透過 kNN monitor 與 linear probing 評估表徵品質。

## 環境建置

```bash
conda create -n AI_hw2 python=3.11 -y
conda activate AI_hw2
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install numpy matplotlib tqdm
```

## 專案結構

```
hw2/
├── config.py               # 所有超參數（dataclass + argparse）
├── dataset.py              # 資料載入與資料增強
├── model.py                # Modified ResNet-18、Projector Head、各模型定義
├── utils.py                # NT-Xent Loss、kNN Monitor、MetricsLogger、checkpoint
├── train_simclr.py         # SimCLR 自監督訓練
├── train_supervised.py     # 監督式訓練 baseline
├── linear_eval.py          # Linear probing 評估
├── requirements.txt        # 套件相依
└── README.md
```

## 模組說明

| 模組 | 說明 |
|------|------|
| `config.py` | 集中管理所有超參數，任何欄位皆可透過 `--flag` 從命令列覆寫 |
| `dataset.py` | SimCLR 雙視角增強、監督式增強、測試 transform；支援 CIFAR-10/100、STL-10 |
| `model.py` | 針對 32×32 輸入修改的 ResNet-18（conv1 改 3×3, 移除 maxpool）、ProjectorHead (512→512→128)、SimCLRModel、SupervisedModel、LinearClassifier |
| `utils.py` | `NTXentLoss`（對比損失）、`knn_evaluate`（kNN 分類監控）、`MetricsLogger`（記錄曲線並輸出 plot/CSV）、checkpoint 存讀、seed 設定 |
| `train_simclr.py` | SimCLR 訓練迴圈，每 5 個 epoch 執行 kNN monitor，自動存 checkpoint 與學習曲線 |
| `train_supervised.py` | 標準監督式訓練，同樣的 backbone + Linear(512→10) |
| `linear_eval.py` | 凍結 backbone，僅訓練一層線性分類器來評估表徵品質；支援 `--resume random` 做隨機 baseline |

## 預設超參數

| 參數 | 值 | 說明 |
|------|------|------|
| Optimizer | Adam | — |
| SSL Learning Rate | 3e-4 | — |
| Weight Decay | 1e-6 | — |
| Temperature | 0.5 | NT-Xent 溫度參數 |
| Batch Size | 256 | 視 VRAM 調整 |
| SSL Epochs | 200 | — |
| Projector | 512→512→128 | 兩層 MLP |
| kNN k | 20 | 每 5 個 epoch 評估一次 |
| Linear Probing | Adam, lr=1e-3, 100 epochs | 凍結 backbone |

## 執行實驗

所有腳本皆在 `hw2/` 目錄下執行：

```bash
conda activate AI_hw2
```

### (Required) SimCLR Baseline

```bash
# 訓練 SimCLR（200 epochs），過程中自動記錄 loss 曲線與 kNN accuracy
python train_simclr.py --batch_size 256 --experiment_name baseline

# 對訓練好的 SSL 模型做 linear probing
python linear_eval.py --resume checkpoints/baseline_epoch200.pt --experiment_name ssl_probe
```

### (Required) Supervised Baseline

```bash
python train_supervised.py --batch_size 256 --experiment_name supervised
```

### (Optional) Random Backbone Baseline

```bash
python linear_eval.py --resume random --experiment_name random_probe
```

### (Optional) Temperature Ablation

```bash
python train_simclr.py --temperature 0.1 --experiment_name temp01
python train_simclr.py --temperature 5.0 --experiment_name temp50
```

### (Optional) Batch Size Ablation

```bash
python train_simclr.py --batch_size 512 --experiment_name bs512
python train_simclr.py --batch_size 128 --experiment_name bs128
python train_simclr.py --batch_size 64  --experiment_name bs64
python train_simclr.py --batch_size 32  --experiment_name bs32
```

### (Optional) No Projector Head

```bash
python train_simclr.py --use_projector False --experiment_name no_proj
```

### (Optional) Transfer to Other Datasets

```bash
# CIFAR-100
python linear_eval.py --resume checkpoints/baseline_epoch200.pt \
    --lp_dataset cifar100 --experiment_name ssl_cifar100

# STL-10
python linear_eval.py --resume checkpoints/baseline_epoch200.pt \
    --lp_dataset stl10 --experiment_name ssl_stl10
```

## 輸出結構

訓練完成後自動產生以下檔案：

```
checkpoints/
├── baseline_epoch50.pt
├── baseline_epoch100.pt
├── baseline_epoch150.pt
├── baseline_epoch200.pt
└── ...

logs/
├── baseline/
│   ├── loss_curve.png          # NT-Xent loss 曲線
│   ├── knn_curve.png           # kNN accuracy 曲線
│   └── metrics.csv             # 逐 epoch 數值紀錄
├── supervised/
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   └── metrics.csv
└── ssl_probe/
    ├── linear_probe_acc.png
    └── metrics.csv
```

## 參考資料

- [SimCLR 原始論文 (Chen et al., 2020)](https://arxiv.org/abs/2002.05709)
- [PyTorch](https://pytorch.org/) / [torchvision](https://pytorch.org/vision/)
