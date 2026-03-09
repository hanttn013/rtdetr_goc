import torch
from src.zoo.rtdetr.rtdetr import RTDETR
from src.nn.backbone.presnet import PResNet
from src.zoo.rtdetr.hybrid_encoder import HybridEncoder
from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer

def build_dummy_model():
    """
    Builds a dummy RT-DETR R50 model using the exact classes from the repo.
    """
    backbone = PResNet(
        depth=50, 
        variant='d', 
        return_idx=[1, 2, 3], 
        num_stages=4, 
        freeze_norm=True, 
        pretrained=False
    )
    
    encoder = HybridEncoder(
        in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        use_encoder_idx=[2],
        num_encoder_layers=1,
        nhead=8,
        dim_feedforward=1024,
        expansion=1.0,
        depth_mult=1,
        act='silu',
        eval_spatial_size=[640, 640]
    )
    
    decoder = RTDETRTransformer(
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=[256, 256, 256],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_decoder_points=4,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=1024,
        num_denoising=100,
        eval_spatial_size=[640, 640]
    )
    
    model = RTDETR(backbone, encoder, decoder)
    return model

def print_shape_hook(module, input, output):
    """
    A forward hook to print the shapes of inputs and outputs of a module.
    """
    print(f"\n--- Output of {module.__class__.__name__} ---")
    if isinstance(output, torch.Tensor):
        print(f"Shape: {output.shape}")
    elif isinstance(output, (list, tuple)):
        for i, out in enumerate(output):
            if isinstance(out, torch.Tensor):
                print(f"Output {i} Shape: {out.shape}")
            elif isinstance(out, dict):
                print(f"Output {i} Keys: {list(out.keys())}")
                for k, v in out.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k} Shape: {v.shape}")
                    elif isinstance(v, list):
                        print(f"  {k} List Length: {len(v)}")
    elif isinstance(output, dict):
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                print(f"Key '{k}' Shape: {v.shape}")
            elif isinstance(v, list) and len(v)>0 and isinstance(v[0], torch.Tensor):
                print(f"Key '{k}' List of {len(v)} tensors, shape[0]: {v[0].shape}")


def main():
    print("Building model...")
    model = build_dummy_model()
    model.eval() # Set to eval mode to simplify output (no aux losses)

    # Register hooks to dump shapes
    model.backbone.register_forward_hook(print_shape_hook)
    model.encoder.register_forward_hook(print_shape_hook)
    model.decoder.register_forward_hook(print_shape_hook)
    model.decoder.decoder.register_forward_hook(print_shape_hook)
    
    print("\nStarting simulated forward pass...")
    # B=2, C=3, H=640, W=640
    dummy_input = torch.randn(2, 3, 640, 640)
    
    with torch.no_grad():
         output = model(dummy_input)
         
    print("\n--- Final Model Output ---")
    if isinstance(output, dict):
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape}")
            elif isinstance(v, list):
                print(f"{k}: list of length {len(v)}")

if __name__ == '__main__':
    main()
