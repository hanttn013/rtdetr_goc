import os
import json
import matplotlib.pyplot as plt

LOG_PATH = r"model\log.txt"
SAVE_DIR = r"metrics\plots"

os.makedirs(SAVE_DIR, exist_ok=True)

epochs = []
train_loss = []
train_lr = []
map5095 = []   # AP@[.50:.95]
map50 = []     # AP@0.50
map75 = []     # AP@0.75
aps = []       # AP small
apm = []       # AP medium
apl = []       # AP large

loss_vfl = []
loss_bbox = []
loss_giou = []

with open(LOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        if "epoch" in record:
            epochs.append(record["epoch"])

        if "train_loss" in record:
            train_loss.append(record["train_loss"])

        if "train_lr" in record:
            train_lr.append(record["train_lr"])

        if "train_loss_vfl" in record:
            loss_vfl.append(record["train_loss_vfl"])

        if "train_loss_bbox" in record:
            loss_bbox.append(record["train_loss_bbox"])

        if "train_loss_giou" in record:
            loss_giou.append(record["train_loss_giou"])

        coco = record.get("test_coco_eval_bbox", None)
        if isinstance(coco, list) and len(coco) >= 6:
            map5095.append(coco[0])
            map50.append(coco[1])
            map75.append(coco[2])
            aps.append(coco[3])
            apm.append(coco[4])
            apl.append(coco[5])

print(f"epochs      : {len(epochs)}")
print(f"train_loss  : {len(train_loss)}")
print(f"train_lr    : {len(train_lr)}")
print(f"mAP         : {len(map5095)}")

# 1. Total training loss
if train_loss:
    plt.figure(figsize=(8, 5))
    plt.plot(epochs[:len(train_loss)], train_loss, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_loss.png"), dpi=200)
    plt.close()

# 2. Loss components
if loss_vfl and loss_bbox and loss_giou:
    plt.figure(figsize=(8, 5))
    plt.plot(epochs[:len(loss_vfl)], loss_vfl, label="VFL")
    plt.plot(epochs[:len(loss_bbox)], loss_bbox, label="BBox")
    plt.plot(epochs[:len(loss_giou)], loss_giou, label="GIoU")
    plt.title("Training Loss Components")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_loss_components.png"), dpi=200)
    plt.close()

# 3. Learning rate
if train_lr:
    plt.figure(figsize=(8, 5))
    plt.plot(epochs[:len(train_lr)], train_lr, marker='o')
    plt.title("Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "learning_rate.png"), dpi=200)
    plt.close()

# 4. mAP curves
if map5095:
    plt.figure(figsize=(8, 5))
    plt.plot(epochs[:len(map5095)], map5095, label="mAP@[0.50:0.95]")
    plt.plot(epochs[:len(map50)], map50, label="mAP@0.50")
    plt.plot(epochs[:len(map75)], map75, label="mAP@0.75")
    plt.title("Validation/Test mAP")
    plt.xlabel("Epoch")
    plt.ylabel("AP")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "map_curves.png"), dpi=200)
    plt.close()

# 5. AP by object size
if aps:
    plt.figure(figsize=(8, 5))
    plt.plot(epochs[:len(aps)], aps, label="AP Small")
    plt.plot(epochs[:len(apm)], apm, label="AP Medium")
    plt.plot(epochs[:len(apl)], apl, label="AP Large")
    plt.title("AP by Object Size")
    plt.xlabel("Epoch")
    plt.ylabel("AP")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "ap_by_size.png"), dpi=200)
    plt.close()

print(f"✅ Saved all plots to: {SAVE_DIR}")