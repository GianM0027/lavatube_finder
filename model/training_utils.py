import time
from collections import Counter
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.metrics import precision_recall_fscore_support


def print_model_structure(model):
    """Stampa la struttura di un modello PyTorch in stile Keras/TensorFlow."""
    header = f"{'Layer (type)':<30} {'Output Shape':<22} {'Param #':<12}"
    divider = "-" * 66

    print(divider)
    print(header)
    print("=" * 66)

    total_params = 0
    trainable_params = 0

    for name, module in model.named_modules():
        # Ignoriamo il modulo radice (il modello stesso)
        if name == "":
            continue

        # Calcoliamo i parametri associati al singolo layer
        layer_params = sum(p.numel() for p in module.parameters(recurse=False))
        layer_trainable = sum(
            p.numel() for p in module.parameters(recurse=False) if p.requires_grad
        )

        total_params += layer_params
        trainable_params += layer_trainable

        # Formattazione nome e tipo del layer
        layer_type = f"{name} ({module.__class__.__name__})"
        if len(layer_type) > 28:
            layer_type = layer_type[:25] + "..."

        # Nota: PyTorch non mantiene la forma dell'output nei layer senza un passaggio forward
        output_shape = "Multiple / Dynamic"

        print(f"{layer_type:<30} {output_shape:<22} {layer_params:<12,}")

    non_trainable_params = total_params - trainable_params

    print("=" * 66)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")
    print(divider)

def _active_modalities(model: nn.Module):
    """Which input streams this model actually consumes."""
    inner = model.module if hasattr(model, "module") else model
    return (
        getattr(inner, "use_optical", True),
        getattr(inner, "use_thermal", True),
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    max_norm: float = 1.0,
    epoch: Optional[int] = None,
    num_epochs: Optional[int] = None,
) -> Dict[str, float]:
    """Runs one training epoch over the dataloader."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    desc = f"Epoch {epoch}/{num_epochs} [train]" if epoch is not None else "Train"
    progress_bar = tqdm(dataloader, desc=desc, leave=False)

    use_optical, use_thermal = _active_modalities(model)

    for batch_idx, (static_img, thermal_seq, targets) in enumerate(progress_bar):
        # Move tensors to GPU/device -- an unused modality is never transferred,
        # which matters most for the large optical crops.
        static_img = static_img.to(device, non_blocking=True) if use_optical else None
        thermal_seq = thermal_seq.to(device, non_blocking=True) if use_thermal else None
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Automatic Mixed Precision (AMP) for faster GPU training & reduced VRAM
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(static_img, thermal_seq)
            loss = criterion(logits, targets)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Unscale gradients before clipping
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        scaler.step(optimizer)
        scaler.update()

        # Metrics accumulation
        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=1)
        correct_predictions += (preds == targets).sum().item()
        total_samples += batch_size
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())

        progress_bar.set_postfix(
            loss=f"{running_loss / total_samples:.4f}",
            acc=f"{correct_predictions / total_samples * 100:.2f}%",
        )

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    return {
        "loss": epoch_loss,
        "acc": epoch_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: Optional[int] = None,
    num_epochs: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluates the model on the validation dataset."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    desc = f"Epoch {epoch}/{num_epochs} [val]" if epoch is not None else "Validate"
    progress_bar = tqdm(dataloader, desc=desc, leave=False)

    use_optical, use_thermal = _active_modalities(model)

    for static_img, thermal_seq, targets in progress_bar:
        static_img = static_img.to(device, non_blocking=True) if use_optical else None
        thermal_seq = thermal_seq.to(device, non_blocking=True) if use_thermal else None
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(static_img, thermal_seq)
            loss = criterion(logits, targets)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=1)
        correct_predictions += (preds == targets).sum().item()
        total_samples += batch_size
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())

        progress_bar.set_postfix(
            loss=f"{running_loss / total_samples:.4f}",
            acc=f"{correct_predictions / total_samples * 100:.2f}%",
        )

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    return {
        "loss": epoch_loss,
        "acc": epoch_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    device: torch.device,
    save_path: str = "best_model.pt",
) -> Dict[str, list]:
    """Full training loop with validation, learning rate scheduling, and best model checkpointing."""
    model.to(device)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    history = {
        "train_loss": [], "train_acc": [], "train_precision": [], "train_recall": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_precision": [], "val_recall": [], "val_f1": [],
    }

    modality = getattr(model, "modality", "both")
    print(f"Starting training on device: {device} | modality: {modality}")
    print("=" * 65)

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Train & Validate
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            epoch=epoch, num_epochs=num_epochs,
        )
        val_metrics = validate_one_epoch(
            model, val_loader, criterion, device,
            epoch=epoch, num_epochs=num_epochs,
        )

        # Update Learning Rate Scheduler
        if scheduler is not None:
            if isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        # Save metric history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["train_precision"].append(train_metrics["precision"])
        history["train_recall"].append(train_metrics["recall"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        # Checkpoint best model weights. save_path=None skips saving, which is
        # what k-fold wants: a checkpoint per fold has no meaning.
        saved_flag = ""
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            if save_path is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_acc": best_val_acc,
                        "best_val_precision": val_metrics["precision"],
                        "best_val_recall": val_metrics["recall"],
                        "best_val_f1": val_metrics["f1"],
                    },
                    save_path,
                )
                saved_flag = " [BEST MODEL SAVED]"

        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} [{elapsed:.1f}s] - "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_metrics['loss']:.4f} - Train Acc: {train_metrics['acc']*100:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} - Val Acc: {val_metrics['acc']*100:.2f}% | "
            f"Val Precision: {val_metrics['precision']:.4f} - Val Recall: {val_metrics['recall']:.4f} - "
            f"Val F1: {val_metrics['f1']:.4f}{saved_flag}"
        )

    print("=" * 65)
    print(f"Training finished. Best Validation Accuracy: {best_val_acc*100:.2f}%")
    return history


def pad_collate_fn(batch):
    """
    Pads the HiRISE images' spatial dimensions (H, W) across a batch on CPU.

    Thermal frames are NOT padded to the image size: they are THEMIS windows at
    native ~100 m/pixel resolution with their own fixed, much smaller shape, so
    they are stacked as-is.
    """
    images, thermals, labels = zip(*batch)

    max_h = max(img.shape[-2] for img in images)
    max_w = max(img.shape[-1] for img in images)

    padded_images = []

    for img in images:
        pad_h = max_h - img.shape[-2]
        pad_w = max_w - img.shape[-1]

        padded_images.append(F.pad(img, (0, pad_w, 0, pad_h), value=0))

    # Return on CPU! train_one_epoch will move them to GPU cleanly.
    batched_images = torch.stack(padded_images)
    batched_thermals = torch.stack(thermals)
    batched_labels = torch.tensor(labels, dtype=torch.long)

    return batched_images, batched_thermals, batched_labels


def group_aware_split(groups, labels, val_split=0.2, seed=42):
    """
    Split indices so that every sample sharing a group stays on one side.

    Groups are assigned per class, so the class balance of the validation set
    still matches the dataset even though group sizes vary a lot (a
    DeepLandforms landform carries ~4.7 images, a plain-terrain crop exactly 1).
    Within a class, groups are shuffled and taken until the target share of that
    class's samples is reached.
    """
    rng = np.random.default_rng(seed)

    members = {}
    for idx, group in enumerate(groups):
        members.setdefault(group, []).append(idx)

    # A group must stay intact even when it spans classes -- a handful of sites
    # carry different labels across repeat observations. Such a group is
    # stratified under its majority class, but never split.
    by_class = {}
    for group, idxs in members.items():
        counts = Counter(labels[i] for i in idxs)
        majority = counts.most_common(1)[0][0]
        by_class.setdefault(majority, []).append(idxs)

    train_idx, val_idx = [], []

    for label, group_list in by_class.items():
        order = rng.permutation(len(group_list))
        n_samples = sum(len(group_list[i]) for i in order)
        target = n_samples * val_split

        taken = 0
        for position, i in enumerate(order):
            idxs = group_list[i]
            # Keep filling validation until the target share is reached, but
            # never take every group of a class.
            if taken < target and position < len(order) - 1:
                val_idx.extend(idxs)
                taken += len(idxs)
            else:
                train_idx.extend(idxs)

    return sorted(train_idx), sorted(val_idx)


def loaders_from_indices(
    image_dataset, train_idx, val_idx, batch_size=2, num_workers=2
):
    """Build train/val loaders for an explicit index split."""
    train_loader = DataLoader(
        Subset(image_dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(image_dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def group_kfold_indices(groups, labels, n_splits=5, seed=42):
    """
    Stratified, group-aware folds.

    Every sample sharing a group stays in one fold, so resolution replicas and
    the two halves of a matched crop pair cannot straddle a split, while class
    balance is held across folds. Worth preferring over a single hold-out here:
    the matched dataset has only ~229 independent groups, so one 20% split
    leaves ~45 validation groups and the noise can swamp the effect being
    measured.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    return list(splitter.split(np.zeros(len(labels)), labels, groups))


def cross_validate(
    image_dataset,
    model_factory,
    criterion,
    optimizer_factory,
    num_epochs,
    device,
    n_splits=5,
    batch_size=4,
    num_workers=2,
    seed=42,
    select_by="val_f1",
    verbose=True,
):
    """
    Run group-aware stratified k-fold and report per-fold and aggregate metrics.

    :param model_factory: zero-argument callable returning a fresh model. A new
        model per fold is essential -- reusing one would carry weights (and the
        previous fold's validation data) across folds.
    :param optimizer_factory: callable taking the model, returning an optimizer.
    :param select_by: history key whose best epoch is reported for each fold.
    :return: (per_fold DataFrame, history list)
    """
    import pandas as pd

    folds = group_kfold_indices(
        image_dataset.group_keys(), image_dataset.img_labels, n_splits, seed
    )

    metrics = ["val_loss", "val_acc", "val_precision", "val_recall", "val_f1"]
    rows, histories = [], []

    for fold, (train_idx, val_idx) in enumerate(folds, start=1):
        if verbose:
            print(f"\n--- fold {fold}/{n_splits} "
                  f"({len(train_idx)} train / {len(val_idx)} val) ---")

        train_loader, val_loader = loaders_from_indices(
            image_dataset, train_idx, val_idx, batch_size, num_workers
        )

        model = model_factory()
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer_factory(model),
            scheduler=None,
            num_epochs=num_epochs,
            device=device,
            save_path=None,  # checkpoints are meaningless across folds
        )

        best = int(np.argmax(history[select_by]))
        rows.append({"fold": fold, "best_epoch": best + 1,
                     **{m: history[m][best] for m in metrics}})
        histories.append(history)

    per_fold = pd.DataFrame(rows).set_index("fold")

    if verbose:
        print("\n" + "=" * 65)
        print(per_fold.round(4).to_string())
        summary = per_fold[metrics].agg(["mean", "std"])
        print("\n" + summary.round(4).to_string())

    return per_fold, histories


def create_dataloaders(
    image_dataset, batch_size=2, val_split=0.2, num_workers=2, group_aware=True
):
    if group_aware and hasattr(image_dataset, "group_keys"):
        train_idx, val_idx = group_aware_split(
            image_dataset.group_keys(), image_dataset.img_labels, val_split
        )
        train_dataset = Subset(image_dataset, train_idx)
        val_dataset = Subset(image_dataset, val_idx)
    else:
        # Random row split: leaks rescaled copies of the same landform across
        # the split. Kept only for comparison against earlier results.
        val_size = int(len(image_dataset) * val_split)
        train_size = len(image_dataset) - val_size

        train_dataset, val_dataset = random_split(
            image_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
        num_workers=num_workers,
        pin_memory=True  # Enables fast CPU->GPU transfer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


