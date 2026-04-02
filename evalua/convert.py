import sys
import torch
import builtins
from pathlib import Path
from contextlib import contextmanager

ROOT = Path(r"E:\Pj\Source Code Deep Learning\rtdetr_goc")
PROJECT_DIR = ROOT / "rtdetrv2_pytorch"
sys.path.append(str(PROJECT_DIR))

from src.core import YAMLConfig

CONFIG_PATH = PROJECT_DIR / r"configs\rtdetrv2\include\rtdetrv2_r18vd_custom_vehicle.yml"
CKPT_PATH = ROOT / r"model\best.pth"
ONNX_PATH = ROOT / r"model\model.onnx"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextmanager
def patch_open_for_yaml():
    original_open = builtins.open

    def patched_open(file, mode='r', *args, **kwargs):
        file_str = str(file).lower()

        if (
            'r' in mode
            and 'b' not in mode
            and (file_str.endswith('.yml') or file_str.endswith('.yaml'))
            and 'encoding' not in kwargs
        ):
            kwargs['encoding'] = 'utf-8-sig'

        return original_open(file, mode, *args, **kwargs)

    builtins.open = patched_open
    try:
        yield
    finally:
        builtins.open = original_open


# ===== Load config =====
with patch_open_for_yaml():
    cfg = YAMLConfig(str(CONFIG_PATH))

cfg.yaml_cfg["PResNet"]["pretrained"] = False

model = cfg.model.to(device)
model.eval()

# ===== Load weights =====
ckpt = torch.load(str(CKPT_PATH), map_location=device)
state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
model.load_state_dict(state_dict, strict=True)

# ===== Dummy input =====
dummy = torch.randn(1, 3, 640, 640).to(device)

# ===== Export =====
torch.onnx.export(
    model,
    dummy,
    str(ONNX_PATH),
    input_names=["images"],
    output_names=["pred_logits", "pred_boxes"],
    opset_version=17,
    dynamic_axes={
        "images": {0: "batch"},
        "pred_logits": {0: "batch"},
        "pred_boxes": {0: "batch"},
    }
)

print("✅ Exported ONNX:", ONNX_PATH)