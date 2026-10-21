import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import ResNet18


class TestImages(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.asarray(self.images[index])
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)
        return tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    images = np.load(args.images, mmap_mode="r")
    if images.shape != (10_000, 32, 32, 3):
        raise ValueError(f"Expected test images with shape (10000, 32, 32, 3), got {images.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(num_classes=10).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.eval()

    loader = DataLoader(TestImages(images), batch_size=args.batch_size, shuffle=False, num_workers=2)
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            predictions.extend(model(batch.to(device)).argmax(dim=1).cpu().tolist())

    if len(predictions) != 10_000 or any(pred < 0 or pred > 9 for pred in predictions):
        raise RuntimeError("Prediction count or class range is invalid")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "pred"])
        writer.writerows(enumerate(predictions))
    print(f"Saved {len(predictions)} predictions to {args.output} on {device}")


if __name__ == "__main__":
    main()
