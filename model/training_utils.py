import gc
import time
from collections import Counter
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
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
    progress: bool = True,
) -> Dict[str, float]:
    """
    Runs one training epoch over the dataloader.

    :param progress: draw a per-batch tqdm bar. Clear it for long unattended
        runs -- see :func:`train_model` for why it matters more than it looks.
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    desc = f"Epoch {epoch}/{num_epochs} [train]" if epoch is not None else "Train"
    progress_bar = tqdm(dataloader, desc=desc, leave=False, disable=not progress)

    use_optical, use_thermal = _active_modalities(model)

    for batch_idx, (static_img, thermal_seq, thermal_time, targets) in enumerate(
        progress_bar
    ):
        # Move tensors to GPU/device -- an unused modality is never transferred,
        # which matters most for the large optical crops.
        static_img = static_img.to(device, non_blocking=True) if use_optical else None
        thermal_seq = thermal_seq.to(device, non_blocking=True) if use_thermal else None
        thermal_time = thermal_time.to(device, non_blocking=True) if use_thermal else None
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Automatic Mixed Precision (AMP) for faster GPU training & reduced VRAM
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(static_img, thermal_seq, thermal_time)
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
    progress: bool = True,
) -> Dict[str, float]:
    """Evaluates the model on the validation dataset."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    desc = f"Epoch {epoch}/{num_epochs} [val]" if epoch is not None else "Validate"
    progress_bar = tqdm(dataloader, desc=desc, leave=False, disable=not progress)

    use_optical, use_thermal = _active_modalities(model)

    for static_img, thermal_seq, thermal_time, targets in progress_bar:
        static_img = static_img.to(device, non_blocking=True) if use_optical else None
        thermal_seq = thermal_seq.to(device, non_blocking=True) if use_thermal else None
        thermal_time = thermal_time.to(device, non_blocking=True) if use_thermal else None
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(static_img, thermal_seq, thermal_time)
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
    checkpoint_metric: str = "f1",
    keep_best_state: bool = False,
    progress: bool = True,
):
    """
    Full training loop with validation, LR scheduling and best-model checkpointing.

    :param checkpoint_metric: validation metric the checkpoint tracks, one of
        ``"f1"`` (default), ``"acc"``, ``"precision"``, ``"recall"``. **Macro F1
        rather than accuracy**, because both tasks here are imbalanced -- 34%
        skylights on binary, and on 3-class the smallest class is 30% -- and
        accuracy is dominated by the majority class. A model that drifts towards
        predicting "other pit" more often can gain accuracy while getting worse
        at the thing being detected; F1 averages over the classes and does not
        reward that.
    :param save_path: where to write the checkpoint. ``None`` skips saving,
        which is what k-fold wants -- a checkpoint per fold has no meaning on its
        own, so :func:`cross_validate` collects the best fold instead.
    :param keep_best_state: also return the best epoch's weights, on the CPU, so
        a caller can keep them without a round trip through the filesystem.
    :return: the history dict, or ``(history, best_state, best_score)`` when
        ``keep_best_state``.
    """
    if checkpoint_metric not in ("f1", "acc", "precision", "recall"):
        raise ValueError(
            "checkpoint_metric must be one of f1/acc/precision/recall, "
            f"got {checkpoint_metric!r}"
        )

    model.to(device)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_score = -1.0
    best_state = None
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
            epoch=epoch, num_epochs=num_epochs, progress=progress,
        )
        val_metrics = validate_one_epoch(
            model, val_loader, criterion, device,
            epoch=epoch, num_epochs=num_epochs, progress=progress,
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

        # Checkpoint on macro F1, not accuracy -- see the docstring.
        saved_flag = ""
        if val_metrics[checkpoint_metric] > best_score:
            best_score = val_metrics[checkpoint_metric]
            best_epoch = epoch

            if keep_best_state:
                # Detached CPU copy, so the caller keeps it while training
                # continues to overwrite the live weights.
                best_state = {
                    key: value.detach().to("cpu", copy=True)
                    for key, value in model.state_dict().items()
                }

            if save_path is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "checkpoint_metric": checkpoint_metric,
                        "best_val_acc": val_metrics["acc"],
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
    print(f"Training finished. Best validation {checkpoint_metric}: "
          f"{best_score * 100:.2f}% (epoch {best_epoch})")

    if keep_best_state:
        return history, best_state, best_score
    return history


def pad_collate_fn(batch):
    """
    Pads the HiRISE images' spatial dimensions (H, W) across a batch on CPU.

    LandformDataset resamples every crop to a fixed ``out_px``, so in normal use
    there is nothing to pad and this is a plain stack. It is kept because it
    still handles the 1x1 placeholder returned under ``load_optical=False``, and
    because it costs nothing.

    Thermal frames are NOT padded to the image size: they are THEMIS windows at
    native ~100 m/pixel resolution with their own fixed, much smaller shape, so
    they are stacked as-is. The same goes for the per-frame time vectors, which
    are ``(T, THERMAL_TIME_DIM)`` and carry no spatial extent at all.
    """
    images, thermals, thermal_times, labels = zip(*batch)

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
    batched_times = torch.stack(thermal_times)
    batched_labels = torch.tensor(labels, dtype=torch.long)

    return batched_images, batched_thermals, batched_times, batched_labels


def group_aware_split(groups, labels, val_split=0.2, seed=42):
    """
    Split indices so that every sample sharing a group stays on one side.

    Groups are assigned per class, so the class balance of the validation set
    still matches the dataset even though group sizes vary (a landform carries
    its 0.5/1/2/3/5 m/pixel replicas, a product carries several landforms).
    Within a class, groups are shuffled and taken until the target share of that
    class's samples is reached.

    Prefer :func:`cross_validate` for anything reported: with 159 product-level
    groups a single 20% hold-out leaves ~32 groups, and the fold-to-fold spread
    is wider than most effects worth measuring.
    """
    rng = np.random.default_rng(seed)

    members = {}
    for idx, group in enumerate(groups):
        members.setdefault(group, []).append(idx)

    # A group must stay intact even when it spans classes -- with product-level
    # grouping most groups do. Such a group is stratified under its majority
    # class, but never split. Ties are broken by the lowest class id rather than
    # by dict order, so the split does not depend on row ordering.
    by_class = {}
    for group, idxs in members.items():
        counts = Counter(labels[i] for i in idxs)
        top = max(counts.values())
        majority = min(label for label, n in counts.items() if n == top)
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
    train_dataset, val_dataset, train_idx, val_idx, batch_size=2, num_workers=2
):
    """
    Build train/val loaders for an explicit index split.

    Two dataset objects, not one, because augmentation has to differ between
    them: the training copy redraws its crop offset and dihedral transform on
    every access, the validation copy does not. Both must be built over the
    *same* annotation table so the indices mean the same thing on each side.
    """
    if len(train_dataset) != len(val_dataset):
        raise ValueError(
            "train and val datasets must cover the same annotation table "
            f"({len(train_dataset)} vs {len(val_dataset)} rows)"
        )

    # Workers are worth having, but they are expensive in a way that is easy to
    # miss. The bottleneck is decoding crops, not the GPU: a fixed-footprint
    # window is up to 1920 px on a side before it is resampled to 384, and GDAL
    # has to decompress all of it (these tiles carry no overviews, so asking
    # rasterio for a decimated read buys nothing -- measured, 68 ms/crop either
    # way). Measured on one training epoch, fixed_gsd, optical:
    #
    #     num_workers=0  62 ms/sample
    #     num_workers=4  19 ms/sample
    #     num_workers=8  14 ms/sample
    #
    # The cost: Windows spawns a fresh interpreter per worker, each of which
    # imports torch and lands at **about 0.7 GB resident**. Giving both loaders
    # `num_workers` therefore costs 2 x 0.7 GB x num_workers, and at 8 that is
    # 11 GB before the training process itself is counted. It has already filled
    # a 31 GB machine and taken the IDE down with it.
    #
    # So: workers on the training loader only. The validation loader gets none,
    # because its crops are deterministic under augment=False and
    # ``LandformDataset(cache_crops=True)`` decodes them once and keeps them --
    # 270 MB for the whole table, against 0.7 GB for a single worker.
    common = dict(
        batch_size=batch_size,
        collate_fn=pad_collate_fn,
        pin_memory=True,
    )

    train_kwargs = dict(common, num_workers=num_workers)
    if num_workers > 0:
        # persistent_workers keeps them alive for the life of the loader, which
        # is one fold, rather than respawning every epoch.
        train_kwargs["persistent_workers"] = True
        train_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        Subset(train_dataset, train_idx), shuffle=True, **train_kwargs
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx), shuffle=False, num_workers=0, **common
    )
    return train_loader, val_loader


def shutdown_loader(loader) -> None:
    """
    Stop a loader's worker processes now, rather than whenever it is collected.

    Persistent workers live until the DataLoader is finalised, and CPython does
    not promise when that happens -- a lingering reference from a traceback or
    an exception context is enough to keep a fold's workers alive while the next
    fold spawns its own. Left alone across a five-fold run that accumulates:
    measured, 47 live worker processes holding 14.2 GB, with the run itself
    stalled.

    Called at the end of every fold in :func:`cross_validate`. Safe on a loader
    with no workers, and safe to call twice.
    """
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            try:
                shutdown()
            except Exception:
                # A worker that is already gone is exactly the outcome wanted.
                pass
        loader._iterator = None


def group_kfold_indices(groups, labels, n_splits=5, seed=42):
    """
    Stratified, group-aware folds.

    Every sample sharing a group stays in one fold, so resolution replicas,
    repeat observations and (at product level) everything sharing an
    acquisition cannot straddle a split, while class balance is held across
    folds. Worth preferring over a single hold-out here: there are only ~312
    landform-level groups and ~159 product-level ones, so one 20% split leaves
    32-62 validation groups and the noise can swamp the effect being measured.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    return list(splitter.split(np.zeros(len(labels)), labels, groups))


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    """
    Class probabilities over a loader, for a model that is already trained.

    :return: ``(targets, probabilities)`` as numpy arrays, shapes ``(N,)`` and
        ``(N, n_classes)``. Row order follows the loader, so pass a loader built
        with ``shuffle=False``.
    """
    model.eval()
    model.to(device)

    use_optical, use_thermal = _active_modalities(model)
    all_targets, all_probs = [], []

    for static_img, thermal_seq, thermal_time, targets in dataloader:
        static_img = static_img.to(device, non_blocking=True) if use_optical else None
        thermal_seq = thermal_seq.to(device, non_blocking=True) if use_thermal else None
        thermal_time = thermal_time.to(device, non_blocking=True) if use_thermal else None

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(static_img, thermal_seq, thermal_time)

        all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        all_targets.append(targets.numpy())

    return np.concatenate(all_targets), np.concatenate(all_probs)


def binary_report(targets, probabilities, positive=1, label="skylight", verbose=True):
    """
    Detector-style readout for the Type-1-versus-rest task.

    Macro F1 averages over both classes and so is flattered by the majority
    class. What matters for a detector is what happens to the class being
    detected, so this reports precision, recall and F1 for the positive class
    alone, plus average precision -- the area under the precision/recall curve,
    which is threshold-free and is the right summary for an imbalanced
    detection problem (here 625 skylights against 1221 other pits).

    :param probabilities: ``(N, 2)`` from :func:`collect_predictions`, or a
        ``(N,)`` vector of positive-class scores.
    :return: dict of metrics.
    """
    from sklearn.metrics import (average_precision_score, confusion_matrix,
                                 precision_recall_fscore_support)

    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    scores = probabilities if probabilities.ndim == 1 else probabilities[:, positive]
    predictions = (scores >= 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average="binary", pos_label=positive, zero_division=0
    )
    result = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "average_precision": float(average_precision_score(targets == positive, scores)),
        "accuracy": float((predictions == targets).mean()),
        "positive_rate": float((targets == positive).mean()),
    }

    if verbose:
        print(f"{label}: precision {result['precision']:.3f}  "
              f"recall {result['recall']:.3f}  F1 {result['f1']:.3f}")
        print(f"average precision {result['average_precision']:.3f}  "
              f"(a constant 'always {label}' scores {result['positive_rate']:.3f})")
        print("confusion (rows=true, cols=pred):")
        print(confusion_matrix(targets, predictions))

    return result


def cross_validate(
    dataset_factory,
    model_factory,
    criterion,
    optimizer_factory,
    num_epochs,
    device,
    n_splits=5,
    batch_size=4,
    num_workers=0,
    seed=42,
    select_by="val_f1",
    group_level="product",
    collect_oof=False,
    checkpoint_path=None,
    checkpoint_metric="f1",
    progress=False,
    verbose=True,
):
    """
    Run group-aware stratified k-fold and report per-fold and aggregate metrics.

    Which epoch is reported
    -----------------------
    The ``val_*`` columns come from each fold's **final** epoch, at a fixed
    budget of ``num_epochs``. The ``best_*`` columns come from the epoch that
    maximised ``select_by`` *on the validation fold itself*, which is a model
    selection performed on the data being scored and is optimistic by
    construction. On the binary task the gap was 2.4 accuracy points for
    optical (95.9 best-epoch against 93.5 final) and 4.5 for thermal (73.5
    against 69.0), always in the same direction, with ``best_epoch`` landing on
    epoch 1 of 8 for one fold.

    That matters here beyond the usual, because the number these models are
    compared against -- the ``histogram`` baseline -- has no epochs and so gets
    no such selection. Reporting best-epoch CNNs against a fixed baseline
    charges the whole bias to the CNN's side of the comparison.

    **Report ``val_*``. Quote ``best_*`` only as a diagnostic**, next to
    ``best_epoch``: a fold whose best epoch is far from the last one is a fold
    that is still moving, which is an argument for a different budget rather
    than a better score.

    :param dataset_factory: callable taking ``augment: bool`` and returning a
        dataset over the full annotation table. Called twice -- once augmented
        for training, once not for validation -- so the two sides differ only in
        augmentation, never in content.
    :param model_factory: zero-argument callable returning a fresh model. A new
        model per fold is essential -- reusing one would carry weights (and the
        previous fold's validation data) across folds.
    :param optimizer_factory: callable taking the model, returning an optimizer.
    :param select_by: history key whose best epoch is *also* recorded, as the
        ``best_*`` columns. It is not the headline number -- see below.
    :param group_level: ``"product"`` (honest) or ``"landform"``. See
        ``LandformDataset.group_keys``.
    :param seed: drives the fold partition **and** the weight initialisation, so
        repeating a run with a different seed re-partitions and re-initialises.
        That is what makes repeated cross-validation an estimate of variability
        rather than five correlated redraws of the same split.
    :param collect_oof: also return out-of-fold predictions. Every sample is
        predicted exactly once, by the fold that did not train on it, giving one
        honest confusion matrix over the whole data set rather than five partial
        ones. Taken from each fold's **final** epoch, not its best, so it does
        not inherit the optimism noted above.
    :param checkpoint_path: where to write **one** checkpoint per run: the
        weights of whichever fold reached the highest validation
        ``checkpoint_metric``. ``None`` writes nothing.

        One file per run, not one per fold, because the folds are not
        independently useful and 5 x 3 modalities x 3 seeds x 2 tasks x 2 crop
        policies of ResNet18 weights is 180 files. The optimiser state is
        deliberately not stored: this is a model to publish or fine-tune from,
        not a run to resume.
    :param checkpoint_metric: what that checkpoint tracks. Macro F1 by default
        rather than accuracy -- see :func:`train_model`.
    :param progress: per-batch tqdm bars. **Off by default here**, unlike
        :func:`train_model`, because a full grid emits enough of them to take an
        IDE down. One run is 5 folds x 8 epochs x (140 train + 44 val) batches =
        about 7,400 bar updates; 24 runs is 177,000. In a Jupyter front-end each
        one is a widget message the client must hold, and PyCharm's notebook
        view runs in a JVM capped at 2 GB of heap -- measured, it filled that
        heap, crashed once with a native OOM, and afterwards sat spinning at 11
        cores of garbage collection while stealing them from the training
        process. The per-epoch summary line that ``train_model`` prints is 40
        lines per run and tells you the same thing.
    :return: ``(per_fold DataFrame, history list)``, or
        ``(per_fold, histories, targets, probabilities)`` when ``collect_oof``.
    """
    import pandas as pd

    train_dataset = dataset_factory(True)
    val_dataset = dataset_factory(False)

    groups = val_dataset.group_keys(level=group_level)
    folds = group_kfold_indices(
        groups, val_dataset.img_labels, n_splits, seed
    )

    if verbose:
        print(f"{len(val_dataset)} samples in {len(set(groups))} "
              f"{group_level}-level groups")

    metrics = ["val_loss", "val_acc", "val_precision", "val_recall", "val_f1"]
    rows, histories = [], []

    oof_targets = np.zeros(len(val_dataset), dtype=int)
    oof_probs = None
    best_run = {"score": -1.0, "state": None, "fold": None}

    for fold, (train_idx, val_idx) in enumerate(folds, start=1):
        # The seed has to reach the weights too, not only the fold partition,
        # or repeated runs differ in how the data was cut but start from the
        # same initialisation and understate the true spread.
        torch.manual_seed(seed * 1000 + fold)
        if verbose:
            print(f"\n--- fold {fold}/{n_splits} "
                  f"({len(train_idx)} train / {len(val_idx)} val) ---")

        train_loader, val_loader = loaders_from_indices(
            train_dataset, val_dataset, train_idx, val_idx, batch_size, num_workers
        )

        model = model_factory()
        # train_model returns a bare history unless asked to keep the best
        # weights, so the arity of what comes back depends on this flag.
        # Unpacking three values unconditionally breaks every call that does not
        # write a checkpoint -- which is the default.
        keep_state = checkpoint_path is not None
        result = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer_factory(model),
            scheduler=None,
            num_epochs=num_epochs,
            device=device,
            save_path=None,   # one checkpoint per run, written below
            checkpoint_metric=checkpoint_metric,
            keep_best_state=keep_state,
            progress=progress,
        )
        if keep_state:
            history, fold_state, fold_score = result
        else:
            history, fold_state, fold_score = result, None, -1.0

        if checkpoint_path is not None and fold_score > best_run["score"]:
            best_run = {"score": fold_score, "state": fold_state, "fold": fold}

        # Headline: the final epoch, at a fixed budget, chosen without looking
        # at the fold it is scored on. Best-epoch is kept beside it as a
        # diagnostic -- see the docstring for why it is not the headline.
        best = int(np.argmax(history[select_by]))
        rows.append({
            "fold": fold,
            "epochs": num_epochs,
            **{m: history[m][-1] for m in metrics},
            "best_epoch": best + 1,
            **{f"best_{m}": history[m][best] for m in metrics},
        })
        histories.append(history)

        if collect_oof:
            # val_loader does not shuffle, so its row order is val_idx order.
            fold_targets, fold_probs = collect_predictions(model, val_loader, device)
            if oof_probs is None:
                oof_probs = np.zeros((len(val_dataset), fold_probs.shape[1]))
            oof_targets[val_idx] = fold_targets
            oof_probs[val_idx] = fold_probs

        # Reclaim the fold before the next one starts. Without this the worker
        # processes of every fold stay alive at once -- see shutdown_loader.
        shutdown_loader(train_loader)
        shutdown_loader(val_loader)
        del train_loader, val_loader, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    per_fold = pd.DataFrame(rows).set_index("fold")

    if checkpoint_path is not None and best_run["state"] is not None:
        import os

        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_run["state"],
                "checkpoint_metric": checkpoint_metric,
                "score": best_run["score"],
                "fold": best_run["fold"],
                "n_splits": n_splits,
                "seed": seed,
                "group_level": group_level,
                "num_epochs": num_epochs,
            },
            checkpoint_path,
        )
        if verbose:
            print(f"\ncheckpoint: fold {best_run['fold']}, "
                  f"val {checkpoint_metric} {best_run['score'] * 100:.1f}% "
                  f"-> {checkpoint_path}")

    if verbose:
        print("\n" + "=" * 65)
        print(per_fold.round(4).to_string())
        summary = per_fold[metrics].agg(["mean", "std"])
        print("\nfinal epoch (report this):")
        print(summary.round(4).to_string())
        optimism = (per_fold[f"best_{select_by}"] - per_fold[select_by]).mean()
        print(f"\nbest-epoch selection would add {optimism * 100:+.1f} points of "
              f"{select_by}; it is a diagnostic, not the headline.")

    if collect_oof:
        return per_fold, histories, oof_targets, oof_probs

    return per_fold, histories


def create_dataloaders(
    dataset_factory,
    batch_size=2,
    val_split=0.2,
    num_workers=2,
    group_aware=True,
    group_level="product",
    seed=42,
):
    """
    Single hold-out split, for quick looks only.

    :param dataset_factory: callable taking ``augment: bool``, as in
        :func:`cross_validate`.
    :param group_aware: when cleared, splits rows at random. That leaks
        resolution replicas of the same landform across the split (measured
        previously: 73.8% of validation images had a copy in training) and is
        kept only to reproduce that number, never to report a result.
    """
    train_dataset = dataset_factory(True)
    val_dataset = dataset_factory(False)

    if group_aware and hasattr(val_dataset, "group_keys"):
        train_idx, val_idx = group_aware_split(
            val_dataset.group_keys(level=group_level),
            val_dataset.img_labels,
            val_split,
            seed,
        )
    else:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(val_dataset))
        cut = int(len(val_dataset) * val_split)
        val_idx, train_idx = sorted(order[:cut]), sorted(order[cut:])

    return loaders_from_indices(
        train_dataset, val_dataset, train_idx, val_idx, batch_size, num_workers
    )




def repeated_cross_validate(
    seeds=(42, 43, 44),
    checkpoint_path=None,
    verbose=True,
    **kwargs,
):
    """
    Run :func:`cross_validate` once per seed and stack the results.

    Why more than one seed
    ----------------------
    With 127 product-level groups a 5-fold split leaves about 25 groups per
    fold, and the fold-to-fold spread is comparable to every effect the ablation
    is trying to measure. A single 5-fold run therefore estimates the *mean*
    reasonably and the *uncertainty on that mean* badly, because the five folds
    share one partition: they are not five independent samples of "how well does
    this do on unseen products", they are five slices of one particular way of
    cutting the data.

    Changing the seed re-partitions the groups and re-initialises the weights,
    so the spread across seeds captures both sources of variability. Report the
    mean over all seeds x folds, and the standard deviation **of the per-seed
    means** as the uncertainty on it.

    Cost scales linearly. Three seeds is three times the compute; it is the
    smallest number that gives any read on between-partition variability at all,
    which is why it is the default rather than a larger one.

    :param seeds: one cross-validation run per entry.
    :param checkpoint_path: if given, treated as a template and formatted with
        ``seed=`` so the runs do not overwrite each other. Pass something like
        ``"model/checkpoints/binary_both_seed{seed}.pt"``.
    :param kwargs: everything else is forwarded to :func:`cross_validate`.
    :return: ``(per_run DataFrame indexed by (seed, fold), {seed: history list},
        {seed: (targets, probabilities)})``. The third is empty unless
        ``collect_oof=True`` was forwarded.
    """
    import pandas as pd

    collect_oof = kwargs.get("collect_oof", False)

    frames, histories, oof = [], {}, {}

    for seed in seeds:
        if verbose:
            print()
            print("-" * 65)
            print(f"seed {seed}")
            print("-" * 65)

        path = (checkpoint_path.format(seed=seed)
                if checkpoint_path is not None else None)

        result = cross_validate(
            seed=seed, checkpoint_path=path, verbose=verbose, **kwargs
        )

        if collect_oof:
            per_fold, history, targets, probabilities = result
            oof[seed] = (targets, probabilities)
        else:
            per_fold, history = result

        histories[seed] = history
        frames.append(per_fold.reset_index().assign(seed=seed))

    per_run = (pd.concat(frames, ignore_index=True)
               .set_index(["seed", "fold"])
               .sort_index())

    if verbose:
        metrics = [c for c in ("val_acc", "val_f1") if c in per_run.columns]
        per_seed = per_run.groupby("seed")[metrics].mean()
        print()
        print("=" * 65)
        print("mean per seed:")
        print((per_seed * 100).round(2).to_string())
        for metric in metrics:
            print(f"  {metric}: {per_run[metric].mean() * 100:.1f}% overall, "
                  f"sd of per-seed means {per_seed[metric].std() * 100:.2f} pts, "
                  f"sd across all folds {per_run[metric].std() * 100:.2f} pts")

    return per_run, histories, oof


# ---------------------------------------------------------------------------
# Run cache
# ---------------------------------------------------------------------------
#
# A grid of 36 runs is far longer than one sitting, so each is written to disk
# as it finishes and skipped on the next pass. Two things make that dangerous
# if done naively, and both have bitten this project:
#
# 1. The result is two files. ``folds.json`` is written first, so an
#    interruption between the two writes leaves a run that *looks* finished and
#    then fails on ``np.load``. A third file, written last, is the completion
#    marker: if it is there, all three are.
#
# 2. The cache key is task/policy/modality/seed and nothing else. Change the
#    epoch budget, the crop footprint, the thermal window, whether the model is
#    told the time -- and the old runs are silently reused beside the new ones,
#    producing a table that averages incomparable things. So the settings are
#    stored next to the result and compared on load.


def save_run(stem, per_fold, oof_targets, oof_probs, config=None):
    """
    Write one finished run: per-fold metrics, out-of-fold predictions, config.

    The config file is written **last** and is what :func:`load_run` treats as
    proof that the run completed.

    :param stem: path prefix, e.g. ``model/cv_cache/binary_fixed_gsd_both_seed42``.
    :param config: settings this run was produced under. Anything that changes
        what the numbers mean belongs here.
    """
    import json
    import os

    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    per_fold.to_json(stem + "_folds.json")
    np.savez(stem + "_oof.npz", targets=oof_targets, probs=oof_probs)
    with open(stem + "_config.json", "w") as handle:
        json.dump(config or {}, handle, indent=1, sort_keys=True)


def load_run(stem, config=None, verbose=True):
    """
    Read a cached run back, or ``None`` if it is absent or incomplete.

    :param config: the settings the caller is about to run under. When the
        cached run records different ones, the differing keys are reported and
        ``None`` is returned, so the run is recomputed rather than mixed in.
        Runs written before configs were recorded have no config file; those are
        accepted, with a note, since refusing them would throw away good work.
    :return: ``(per_fold DataFrame, (targets, probabilities))`` or ``None``.
    """
    import json
    import os

    import pandas as pd

    folds_path, oof_path = stem + "_folds.json", stem + "_oof.npz"
    if not (os.path.exists(folds_path) and os.path.exists(oof_path)):
        return None

    config_path = stem + "_config.json"
    if os.path.exists(config_path) and config:
        with open(config_path) as handle:
            stored = json.load(handle)
        differing = sorted(
            key for key in set(stored) | set(config)
            if stored.get(key) != config.get(key)
        )
        if differing:
            if verbose:
                print(f"  {os.path.basename(stem)}: cached under different "
                      f"settings ({', '.join(differing)}) -- recomputing")
            return None
    elif not os.path.exists(config_path) and verbose:
        print(f"  {os.path.basename(stem)}: cached before settings were "
              "recorded; assuming it matches")

    # to_json drops the index NAME, so the fold column comes back as "index".
    per_fold = pd.read_json(folds_path).rename_axis("fold")
    cached = np.load(oof_path)
    return per_fold, (cached["targets"], cached["probs"])
