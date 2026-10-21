from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DATA_DIR = Path(os.environ.get("TASK3_DATA_DIR", "/content/drive/MyDrive/InterIIT_CV/Task3/data"))
DEFAULT_OUTPUT_DIR = (
    Path("/kaggle/working/task3_5fold_oof")
    if Path("/kaggle/working").is_dir()
    else DATA_DIR.parent / "task3_5fold_oof"
)
OUTPUT_DIR = Path(os.environ.get("TASK3_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
NUM_FOLDS = 5
NUM_EPOCHS = 60
BATCH_SIZE = 128
NUM_WORKERS = 2
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
SEED = 2026
NUM_CLASSES = 10

CSV_COLUMNS = [
    "index", "noisy_label", "predicted_label", "probability_noisy_label",
    "probability_predicted_label",
    *[f"probability_class_{class_id}" for class_id in range(NUM_CLASSES)],
    "cross_entropy_loss",
]


def atomic_torch_save(value, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, destination)


def atomic_json_save(value, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def atomic_csv_save(frame, destination):
    temporary = destination.with_name(destination.name + ".tmp")
    frame.to_csv(temporary, index=False, columns=CSV_COLUMNS, float_format="%.9g")
    os.replace(temporary, destination)


def load_ids(path):
    ids = np.loadtxt(path, dtype=np.int64, ndmin=1)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{path.name} must contain unique integer IDs")
    return ids


def load_data():
    required = ["model.py", "train_images.npy", "labels.csv", "train_idx.txt", "val_idx.txt"]
    missing = [name for name in required if not (DATA_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {DATA_DIR}")
    if str(DATA_DIR) not in sys.path:
        sys.path.insert(0, str(DATA_DIR))
    from model import ResNet18

    images = np.load(DATA_DIR / "train_images.npy", mmap_mode="r")
    if images.ndim != 4 or images.shape[1:] not in ((32, 32, 3), (3, 32, 32)):
        raise ValueError(f"Expected 32x32 RGB image array, got {images.shape}")
    channels_last = images.shape[1:] == (32, 32, 3)
    if not np.issubdtype(images.dtype, np.number) or not np.isfinite(images).all():
        raise ValueError("Image array must contain finite numeric values")

    # Deliberately load only noisy labels; clean-label columns cannot enter this run.
    label_frame = pd.read_csv(DATA_DIR / "labels.csv", usecols=["index", "noisy_label"])
    if label_frame["index"].duplicated().any():
        raise ValueError("labels.csv has duplicate index values")
    label_frame["index"] = label_frame["index"].astype(np.int64)
    label_frame["noisy_label"] = label_frame["noisy_label"].astype(np.int64)
    train_ids = load_ids(DATA_DIR / "train_idx.txt")
    val_ids = load_ids(DATA_DIR / "val_idx.txt")
    if len(train_ids) != 45_000 or len(val_ids) != 5_000:
        raise ValueError(f"Expected 45,000 train and 5,000 validation IDs; got {len(train_ids)} and {len(val_ids)}")
    if np.intersect1d(train_ids, val_ids).size:
        raise ValueError("Train and validation IDs overlap")
    if min(train_ids.min(), val_ids.min()) < 0 or max(train_ids.max(), val_ids.max()) >= len(images):
        raise ValueError("An ID is outside train_images.npy")
    if not set(np.concatenate([train_ids, val_ids])).issubset(set(label_frame["index"])):
        raise ValueError("An ID from train_idx.txt or val_idx.txt is missing in labels.csv")
    if not label_frame["noisy_label"].between(0, NUM_CLASSES - 1).all():
        raise ValueError("noisy_label values must be integers from 0 to 9")

    noisy_by_id = np.full(len(images), -1, dtype=np.int64)
    noisy_by_id[label_frame["index"].to_numpy()] = label_frame["noisy_label"].to_numpy()
    image_tensor = torch.from_numpy(np.array(images, copy=True))
    if channels_last:
        image_tensor = image_tensor.permute(0, 3, 1, 2)
    image_tensor = image_tensor.to(torch.float32)
    if image_tensor.max().item() > 1.0:
        image_tensor.div_(255.0)
    return ResNet18, image_tensor, noisy_by_id, train_ids


class IndexedImages(Dataset):
    def __init__(self, images, noisy_by_id, ids, augment=False):
        self.images = images
        self.noisy_by_id = noisy_by_id
        self.ids = np.asarray(ids, dtype=np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, position):
        image_id = int(self.ids[position])
        image = self.images[image_id]
        if self.augment:
            image = F.pad(image, (4, 4, 4, 4), mode="reflect")
            top = int(torch.randint(0, 9, ()).item())
            left = int(torch.randint(0, 9, ()).item())
            image = image[:, top:top + 32, left:left + 32]
            if torch.rand(()) < 0.5:
                image = image.flip(-1)
        return image, int(self.noisy_by_id[image_id]), image_id


def make_loader(dataset, shuffle, seed=None):
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(), persistent_workers=NUM_WORKERS > 0,
        generator=generator, drop_last=False,
    )


@torch.inference_mode()
def export_fold_scores(model, images, noisy_by_id, heldout_ids, device, fold, epoch):
    model.eval()
    loader = make_loader(IndexedImages(images, noisy_by_id, heldout_ids), shuffle=False)
    ids_out, labels_out, predictions_out, probabilities_out, losses_out = [], [], [], [], []
    for batch_images, batch_labels, batch_ids in loader:
        logits = model(batch_images.to(device, non_blocking=True))
        labels_device = batch_labels.to(device, non_blocking=True)
        probabilities = torch.softmax(logits, dim=1)
        losses = F.cross_entropy(logits, labels_device, reduction="none")
        ids_out.append(batch_ids.numpy())
        labels_out.append(batch_labels.numpy())
        predictions_out.append(probabilities.argmax(1).cpu().numpy())
        probabilities_out.append(probabilities.cpu().numpy())
        losses_out.append(losses.cpu().numpy())
    ids = np.concatenate(ids_out)
    labels = np.concatenate(labels_out)
    predictions = np.concatenate(predictions_out)
    probabilities = np.concatenate(probabilities_out)
    losses = np.concatenate(losses_out)
    if len(ids) != len(heldout_ids) or len(np.unique(ids)) != len(heldout_ids) or set(ids) != set(heldout_ids):
        raise RuntimeError(f"Fold {fold} epoch {epoch} did not score each held-out ID exactly once")
    rows = {
        "index": ids,
        "noisy_label": labels,
        "predicted_label": predictions,
        "probability_noisy_label": probabilities[np.arange(len(ids)), labels],
        "probability_predicted_label": probabilities[np.arange(len(ids)), predictions],
        "cross_entropy_loss": losses,
    }
    rows.update({f"probability_class_{class_id}": probabilities[:, class_id] for class_id in range(NUM_CLASSES)})
    scores = pd.DataFrame(rows, columns=CSV_COLUMNS).sort_values("index", kind="stable")
    path = OUTPUT_DIR / f"fold_{fold}" / f"epoch_{epoch:02d}_heldout.csv"
    atomic_csv_save(scores, path)
    return path, float(losses.mean())


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def train_fold(ResNet18, images, noisy_by_id, train_ids, heldout_ids, fold, device):
    fold_dir = OUTPUT_DIR / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    fold_seed = SEED + fold * 10_000
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fold_seed)
    model = ResNet18(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    train_dataset = IndexedImages(images, noisy_by_id, train_ids, augment=True)
    state_path = fold_dir / "resume_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"completed_epoch": 0}
    completed_epoch = int(state.get("completed_epoch", 0))
    if not 0 <= completed_epoch <= NUM_EPOCHS:
        raise ValueError(f"Invalid completed_epoch in {state_path}")

    history_path = fold_dir / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    history_by_epoch = {int(row["epoch"]): row for row in history}
    latest_checkpoint_path = fold_dir / "latest_checkpoint.pt"
    final_checkpoint_path = fold_dir / f"checkpoint_epoch_{NUM_EPOCHS:02d}.pt"
    resume_checkpoint_path = latest_checkpoint_path if latest_checkpoint_path.exists() else final_checkpoint_path
    if resume_checkpoint_path.exists():
        checkpoint = load_checkpoint(resume_checkpoint_path, model, optimizer, scheduler, device)
        checkpoint_epoch = int(checkpoint["epoch"])
        if checkpoint_epoch < completed_epoch or checkpoint_epoch > completed_epoch + 1:
            if not (resume_checkpoint_path == final_checkpoint_path and checkpoint_epoch == NUM_EPOCHS == completed_epoch):
                raise RuntimeError(f"Checkpoint epoch {checkpoint_epoch} conflicts with resume state {completed_epoch}")
        if checkpoint_epoch == completed_epoch + 1:
            scores_path = fold_dir / f"epoch_{checkpoint_epoch:02d}_heldout.csv"
            if not scores_path.exists():
                scores_path, heldout_loss = export_fold_scores(model, images, noisy_by_id, heldout_ids, device, fold, checkpoint_epoch)
            else:
                heldout_loss = float(pd.read_csv(scores_path)["cross_entropy_loss"].mean())
            history_by_epoch[checkpoint_epoch] = {
                "epoch": checkpoint_epoch,
                "train_cross_entropy_loss": checkpoint.get("train_cross_entropy_loss"),
                "heldout_cross_entropy_loss": heldout_loss,
                "learning_rate_next_epoch": checkpoint.get("learning_rate_next_epoch"),
                "heldout_csv": scores_path.name,
            }
            atomic_json_save([history_by_epoch[key] for key in sorted(history_by_epoch)], history_path)
            completed_epoch = checkpoint_epoch
            atomic_json_save({"completed_epoch": completed_epoch}, state_path)
    elif completed_epoch:
        raise FileNotFoundError(f"Resume state says epoch {completed_epoch} is complete, but its checkpoint is missing")
    print(f"Fold {fold}: {len(train_ids)} train, {len(heldout_ids)} held out; resuming at epoch {completed_epoch + 1}")

    for epoch in range(completed_epoch + 1, NUM_EPOCHS + 1):
        model.train()
        loader = make_loader(train_dataset, shuffle=True, seed=fold_seed + epoch)
        train_loss_sum = 0.0
        seen = 0
        for batch_images, batch_labels, _ in loader:
            batch_images = batch_images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch_images), batch_labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(batch_labels)
            seen += len(batch_labels)
        scheduler.step()
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "seed": fold_seed,
            "fold": fold,
            "train_cross_entropy_loss": train_loss_sum / seen,
            "learning_rate_next_epoch": scheduler.get_last_lr()[0],
        }
        atomic_torch_save(checkpoint, latest_checkpoint_path)
        score_path, heldout_loss = export_fold_scores(model, images, noisy_by_id, heldout_ids, device, fold, epoch)
        history_by_epoch[epoch] = {
            "epoch": epoch,
            "train_cross_entropy_loss": train_loss_sum / seen,
            "heldout_cross_entropy_loss": heldout_loss,
            "learning_rate_next_epoch": scheduler.get_last_lr()[0],
            "heldout_csv": score_path.name,
        }
        history = [history_by_epoch[key] for key in sorted(history_by_epoch)]
        atomic_json_save(history, history_path)
        atomic_json_save({"completed_epoch": epoch}, state_path)
        completed_epoch = epoch
        print(f"Fold {fold} | epoch {epoch:02d}/{NUM_EPOCHS} | train CE {train_loss_sum / seen:.5f} | heldout CE {heldout_loss:.5f}")

    if completed_epoch == NUM_EPOCHS and latest_checkpoint_path.exists():
        os.replace(latest_checkpoint_path, final_checkpoint_path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ResNet18, images, noisy_by_id, train_ids = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shuffled = np.random.default_rng(SEED).permutation(train_ids)
    folds = [part.astype(np.int64) for part in np.array_split(shuffled, NUM_FOLDS)]
    flattened = np.concatenate(folds)
    if any(len(fold_ids) != 9_000 for fold_ids in folds):
        raise RuntimeError("Each of the five folds must contain exactly 9,000 IDs")
    if len(np.unique(flattened)) != 45_000 or set(flattened) != set(train_ids):
        raise RuntimeError("Folds must be disjoint and cover train_idx.txt exactly")
    fold_plan = {
        "seed": SEED,
        "num_folds": NUM_FOLDS,
        "train_ids": train_ids.tolist(),
        "heldout_ids": [fold_ids.tolist() for fold_ids in folds],
    }
    plan_path = OUTPUT_DIR / "fold_plan.json"
    if plan_path.exists():
        old_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if old_plan != fold_plan:
            raise RuntimeError("Saved fold plan differs from current inputs; refusing unsafe resume")
    else:
        atomic_json_save(fold_plan, plan_path)

    print(f"Device: {device}. Five fresh ResNet-18 folds, up to {NUM_EPOCHS} epochs each.")
    for fold_index, heldout_ids in enumerate(folds):
        train_fold_ids = np.concatenate([folds[index] for index in range(NUM_FOLDS) if index != fold_index])
        if len(train_fold_ids) != 36_000 or np.intersect1d(train_fold_ids, heldout_ids).size:
            raise RuntimeError(f"Fold {fold_index} split is not disjoint 36,000/9,000")
        train_fold(ResNet18, images, noisy_by_id, train_fold_ids, heldout_ids, fold_index, device)

    frames = []
    for fold_index in range(NUM_FOLDS):
        fold_dir = OUTPUT_DIR / f"fold_{fold_index}"
        state = json.loads((fold_dir / "resume_state.json").read_text(encoding="utf-8"))
        epoch = int(state["completed_epoch"])
        if epoch < 1:
            raise RuntimeError(f"Fold {fold_index} has no completed scoring epoch")
        frames.append(pd.read_csv(fold_dir / f"epoch_{epoch:02d}_heldout.csv"))
    oof = pd.concat(frames, ignore_index=True).sort_values("index", kind="stable")
    if list(oof.columns) != CSV_COLUMNS or len(oof) != 45_000:
        raise RuntimeError("Combined OOF CSV has an invalid schema or row count")
    if oof["index"].duplicated().any() or set(oof["index"].to_numpy()) != set(train_ids):
        raise RuntimeError("Combined OOF CSV must contain every training ID exactly once")
    atomic_csv_save(oof, OUTPUT_DIR / "out_of_fold_scores.csv")
    print(f"Complete: {OUTPUT_DIR / 'out_of_fold_scores.csv'}")


if __name__ == "__main__":
    main()
