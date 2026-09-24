"""
Evaluate a trained flood model checkpoint on Sen1Floods11.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from flood_detection_pipeline import Sen1Floods11Dataset, FloodDetectionLoss, FloodSegmentationModel, create_data_splits
from train_flood_model import validate, visualize_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained Sen1Floods11 flood model checkpoint")
    parser.add_argument("--data-dir", default="datasets/floods/sen1floods11")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained .pth checkpoint")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="logs/flood_eval")
    parser.add_argument("--s1-only", action="store_true")
    parser.add_argument("--no-official-splits", action="store_true")
    parser.add_argument("--s2-bands", nargs="*", type=int, default=[1, 2, 3, 7, 11, 12])
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_chips, val_chips, test_chips = create_data_splits(
        args.data_dir,
        prefer_official_splits=not args.no_official_splits,
    )
    selected_chips = val_chips if args.split == "val" else test_chips

    dataset = Sen1Floods11Dataset(
        args.data_dir,
        selected_chips,
        mode=args.split,
        use_s1=True,
        use_s2=not args.s1_only,
        s2_bands=args.s2_bands,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    in_channels = 2 + (0 if args.s1_only else len(args.s2_bands))
    model = FloodSegmentationModel(in_channels=in_channels)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    criterion = FloodDetectionLoss(bce_weight=0.5, dice_weight=0.5)
    loss, metrics = validate(model, loader, criterion, args.device)

    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "loss": loss,
        "metrics": metrics,
        "dataset_size": len(dataset),
    }
    (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    visualize_predictions(
        model,
        dataset,
        args.device,
        num_samples=min(4, len(dataset)),
        save_path=output_dir / f"{args.split}_predictions.png",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
