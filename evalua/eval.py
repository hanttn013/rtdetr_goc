import json
import pandas as pd

LOG_PATH = r"model\log.txt"
SAVE_CSV = r"metrics\metrics_summary.csv"

rows = []
with open(LOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue

        bbox = r.get("test_coco_eval_bbox", [None] * 12)
        row = {
            "epoch": r.get("epoch"),
            "train_lr": r.get("train_lr"),
            "train_loss": r.get("train_loss"),
            "train_loss_vfl": r.get("train_loss_vfl"),
            "train_loss_bbox": r.get("train_loss_bbox"),
            "train_loss_giou": r.get("train_loss_giou"),
            "mAP_50_95": bbox[0],
            "mAP_50": bbox[1],
            "mAP_75": bbox[2],
            "AP_small": bbox[3],
            "AP_medium": bbox[4],
            "AP_large": bbox[5],
            "AR_1": bbox[6],
            "AR_10": bbox[7],
            "AR_100": bbox[8],
            "AR_small": bbox[9],
            "AR_medium": bbox[10],
            "AR_large": bbox[11],
        }
        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(SAVE_CSV, index=False, encoding="utf-8-sig")
print("Saved:", SAVE_CSV)

print("\nBest mAP row:")
print(df.loc[df["mAP_50_95"].idxmax()])