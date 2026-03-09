"""
Evaluate the RT-DETR model on the test dataset.
Usage: python test_model.py -c ../../configs/dataset/custom.yml -r ../best.pth
"""
import os 
import sys 

# Add the root directory of the project to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

import argparse
import numpy as np
from src.misc import dist_utils
from src.core import YAMLConfig
from src.solver import TASKS
from src.solver.det_engine import evaluate

def main(args):
    # Setup distributed training/testing
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    # Load configuration
    cfg = YAMLConfig(args.config, resume=args.resume)
    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    # Setup model and basic solver components
    solver._setup()
    
    # Check if test_dataloader exists in config, otherwise use val_dataloader
    if hasattr(solver.cfg, 'test_dataloader'):
        print("Using test_dataloader for evaluation...")
        dataloader_cfg = solver.cfg.test_dataloader
    else:
        print("test_dataloader not found in config. Using val_dataloader instead...")
        dataloader_cfg = solver.cfg.val_dataloader

    # Wrap the loader
    solver.test_dataloader = dist_utils.warp_loader(
        dataloader_cfg, 
        shuffle=getattr(dataloader_cfg, 'shuffle', False)
    )
    solver.evaluator = solver.cfg.evaluator

    if solver.cfg.resume:
        print(f'Resume checkpoint from {solver.cfg.resume}')
        solver.load_resume_state(solver.cfg.resume)

    solver.eval()
    module = solver.ema.module if solver.ema else solver.model
    
    print("Evaluating on the test dataset...")
    test_stats, coco_evaluator = evaluate(
        module, 
        solver.criterion, 
        solver.postprocessor,
        solver.test_dataloader, 
        solver.evaluator, 
        solver.device
    )
    
    if solver.output_dir:
        save_dir = solver.output_dir / "evalute"
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Save exact COCO evaluations
        dist_utils.save_on_master(
            coco_evaluator.coco_eval["bbox"].eval, 
            save_dir / "test_eval.pth"
        )
        print(f"Evaluation results saved to {save_dir / 'test_eval.pth'}")
        
        # Extract per-class AP (Average Precision)
        if dist_utils.is_main_process():
            summary_path = save_dir / "test_metrics.txt"
            
            with open(summary_path, "w") as f:
                stats = coco_evaluator.coco_eval["bbox"].stats
                f.write("=== Model Evaluation Metrics on Test Set (COCO format) ===\n")
                f.write(f"mAP (IoU=0.50:0.95): {stats[0]:.4f}\n")
                f.write(f"mAP (IoU=0.50)     : {stats[1]:.4f}\n")
                f.write(f"mAP (IoU=0.75)     : {stats[2]:.4f}\n")
                f.write(f"mAP (small)        : {stats[3]:.4f}\n")
                f.write(f"mAP (medium)       : {stats[4]:.4f}\n")
                f.write(f"mAP (large)        : {stats[5]:.4f}\n")
                f.write(f"AR (maxDets=1)     : {stats[6]:.4f}\n")
                f.write(f"AR (maxDets=10)    : {stats[7]:.4f}\n")
                f.write(f"AR (maxDets=100)   : {stats[8]:.4f}\n")
                
                # Try to extract per-class AP metrics
                try:
                    eval_results = coco_evaluator.coco_eval["bbox"].eval
                    precisions = eval_results['precision']
                    # precisions shape is [T, R, K, A, M] 
                    # T: iou thresholds, R: recall thresholds, K: categories, A: areas, M: maxDets
                    # We want AP per class at IoU=0.50:0.95, area=all, maxDets=100
                    
                    # Compute mean precision over T and R for each class K (using area=0 (all), maxDets=2 (100))
                    # Note: precision is -1 if no GT object is found in that category.
                    cat_ids = coco_evaluator.coco_eval["bbox"].params.catIds
                    
                    f.write("\n=== Per-Class Average Precision (AP at IoU=0.50:0.95) ===\n")
                    f.write(f"{'Class ID':<10} | {'AP':<10}\n")
                    f.write("-" * 25 + "\n")
                    
                    for i, cat_id in enumerate(cat_ids):
                        # p = precisions[:, :, i, 0, 2]
                        # precision computation logic from pycocotools:
                        # AP is the average over IoU thresholds (dim 0) of the average over recall thresholds (dim 1)
                        class_precisions = precisions[:, :, i, 0, 2]
                        class_precisions = class_precisions[class_precisions > -1]
                        
                        if len(class_precisions) > 0:
                            class_ap = np.mean(class_precisions)
                            f.write(f"{cat_id:<10} | {class_ap:.4f}\n")
                        else:
                            f.write(f"{cat_id:<10} | N/A\n")
                except Exception as e:
                    f.write(f"\nCould not parse per-class AP: {str(e)}\n")
                    
            print(f"Summary metrics saved to {summary_path}")
            
            # Print content to console
            with open(summary_path, "r") as f:
                print("\n" + f.read())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the config file")
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('--print-method', type=str, default='builtin')
    parser.add_argument('--print-rank', type=int, default=0)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    main(args)
