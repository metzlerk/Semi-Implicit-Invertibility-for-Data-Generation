#!/usr/bin/env python3
"""
Inspect a ChemicalDataGeneration encoder checkpoint to determine its format.
"""

import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect encoder checkpoint contents.")
    parser.add_argument(
        "--checkpoint",
        default="/home/kjmetzler/ChemicalDataGeneration/models/trained_models/spectrum/hypertuned_encoder.pth",
        help="Path to encoder checkpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    print(f"Loaded checkpoint type: {type(ckpt)}")
    if isinstance(ckpt, dict):
        print(f"Checkpoint keys: {sorted(ckpt.keys())}")
        for key in ("model", "model_state_dict", "state_dict", "encoder_state_dict"):
            if key in ckpt:
                print(f"Key '{key}' type: {type(ckpt[key])}")
        if "config" in ckpt:
            config = ckpt["config"]
            print(f"Config type: {type(config)}")
            if isinstance(config, dict):
                print(f"Config keys: {sorted(config.keys())}")
                for field in ("n_layers", "input_size", "output_size", "init_style", "trainable"):
                    if field in config:
                        print(f"Config {field}: {config[field]}")
                print(f"Config repr: {config}")
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
            print(f"State dict keys ({len(state_dict)}):")
            for key, tensor in state_dict.items():
                if hasattr(tensor, "shape"):
                    print(f"  {key}: {tuple(tensor.shape)}")
                else:
                    print(f"  {key}: {type(tensor)}")
        elif "config" not in ckpt:
            print(f"State dict keys ({len(ckpt)}):")
            for key, tensor in ckpt.items():
                if hasattr(tensor, "shape"):
                    print(f"  {key}: {tuple(tensor.shape)}")
                else:
                    print(f"  {key}: {type(tensor)}")
    else:
        print(f"Checkpoint keys: {list(ckpt.keys())}")
        for key, tensor in ckpt.items():
            if hasattr(tensor, "shape"):
                print(f"  {key}: {tuple(tensor.shape)}")
            else:
                print(f"  {key}: {type(tensor)}")
