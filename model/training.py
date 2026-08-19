import time
from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    max_norm: float = 1.0,
) -> Tuple[float, float]:
    """Runs one training epoch over the dataloader."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (static_img, thermal_seq, targets) in enumerate(dataloader):
        # Move tensors to GPU/device
        static_img = static_img.to(device, non_blocking=True)
        thermal_seq = thermal_seq.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Automatic Mixed Precision (AMP) for faster GPU training & reduced VRAM
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
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

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluates the model on the validation dataset."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for static_img, thermal_seq, targets in dataloader:
        static_img = static_img.to(device, non_blocking=True)
        thermal_seq = thermal_seq.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(static_img, thermal_seq)
            loss = criterion(logits, targets)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=1)
        correct_predictions += (preds == targets).sum().item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


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
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"Starting training on device: {device}")
    print("=" * 65)

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Train & Validate
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        # Update Learning Rate Scheduler
        if scheduler is not None:
            if isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Save metric history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        # Checkpoint best model weights
        saved_flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                },
                save_path,
            )
            saved_flag = " [BEST MODEL SAVED]"

        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} [{elapsed:.1f}s] - "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}%{saved_flag}"
        )

    print("=" * 65)
    print(f"Training finished. Best Validation Accuracy: {best_val_acc*100:.2f}%")
    return history

def create_dataloaders(training_data, validation_data, batch_size=4, device="cuda"):

    pass