"""
Predict and draw bounding boxes on the test dataset images.
Usage: python output/evalute/detect_test.py -c configs/rtdetrv2/include/rtdetrv2_r18vd_custom_vehicle.yml -r output/best.pth -t 0.5
"""
import os 
import sys 

# Add the root directory of the project to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

import argparse
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from pathlib import Path

from src.core import YAMLConfig

def draw_boxes(im_pil, labels, boxes, scores, class_names, save_path, thrh=0.5):
    """
    Draw boxes on a single image and save it.
    """
    draw = ImageDraw.Draw(im_pil)

    # Filter by threshold
    mask = scores > thrh
    valid_labels = labels[mask]
    valid_boxes = boxes[mask]
    valid_scores = scores[mask]

    for j, b in enumerate(valid_boxes):
        c_id = valid_labels[j].item()
        c_name = class_names[c_id] if c_id < len(class_names) else str(c_id)
        score_val = round(valid_scores[j].item(), 2)
        
        # Draw rectangle
        draw.rectangle(list(b), outline='red', width=3)
        
        # Draw text background
        text = f"{c_name} {score_val}"
        text_bbox = draw.textbbox((b[0], max(0, b[1]-15)), text)
        draw.rectangle(text_bbox, fill='red')
        
        # Draw text
        draw.text((b[0], max(0, b[1]-15)), text=text, fill='white')

    im_pil.save(save_path)


def main(args):
    # Load configuration
    cfg = YAMLConfig(args.config, resume=args.resume)
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # Load model weights
    cfg.model.load_state_dict(state)

    class Model(torch.nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = torch.device(args.device)
    model = Model().to(device)
    model.eval()

    # Get class names from config
    class_names = cfg.yaml_cfg.get('class_names', [])
    if not class_names:
        class_names = [f'cls_{i}' for i in range(cfg.yaml_cfg.get('num_classes', 80))]

    # Get test directory
    test_dir = Path('./dataset/test')
    if 'test_dataloader' in cfg.yaml_cfg and 'dataset' in cfg.yaml_cfg['test_dataloader']:
        test_dir = Path(cfg.yaml_cfg['test_dataloader']['dataset'].get('img_folder', './dataset/test'))

    # Retrieve resize ops from training dataloader to set dynamic transform size
    size = [640, 640]
    if 'train_dataloader' in cfg.yaml_cfg:
        try:
            ops = cfg.yaml_cfg['train_dataloader']['dataset']['transforms']['ops']
            for op in ops:
                if op['type'] == 'Resize':
                    size = op['size']
                    if isinstance(size, int):
                        size = [size, size]
                    break
        except KeyError:
            pass

    transforms = T.Compose([
        T.Resize(size),
        T.ToTensor(),
    ])

    save_dir = Path('./output/evalute/images')
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading images from {test_dir}...")
    image_files = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png')) + list(test_dir.glob('*.jpeg'))
    print(f"Found {len(image_files)} images.")
    print(f"Saving detections to {save_dir} with threshold {args.thresh}...")

    with torch.no_grad():
        for i, img_path in enumerate(image_files):
            try:
                im_pil = Image.open(img_path).convert('RGB')
                w, h = im_pil.size
                orig_size = torch.tensor([w, h])[None].to(device)

                im_data = transforms(im_pil)[None].to(device)

                output = model(im_data, orig_size)
                labels, boxes, scores = output
                
                # Assuming batch size 1 for inference
                labels = labels[0]
                boxes = boxes[0]
                scores = scores[0]

                save_path = save_dir / img_path.name
                draw_boxes(im_pil, labels, boxes, scores, class_names, save_path, thrh=args.thresh)
                
                if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
                    print(f"Processed {i + 1}/{len(image_files)} images...")
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    print(f"Done! All images saved to {save_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the config file")
    parser.add_argument('-r', '--resume', type=str, required=True, help='Path to weights (best.pth)')
    parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('-t', '--thresh', type=float, default=0.5, help='Confidence threshold for plotting')
    args = parser.parse_args()
    main(args)
