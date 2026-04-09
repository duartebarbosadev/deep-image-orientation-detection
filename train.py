import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import argparse
import logging
import time

import config
from src.caching import cache_dataset
from src.dataset import (
    ImageOrientationDataset,
    ImageOrientationDatasetFromCache,
    build_source_id,
    discover_upright_image_files,
    split_image_files,
)
from src.model import get_orientation_model
from src.utils import (
    get_autocast_context,
    get_grad_scaler,
    get_device,
    move_batch_to_device,
    should_use_channels_last,
    setup_logging,
    get_data_transforms,
)
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter


SPLIT_STATE_FILENAME = "train_val_split.json"


def get_split_state_path(model_dir: str) -> str:
    return os.path.join(model_dir, SPLIT_STATE_FILENAME)


def load_saved_split_state(model_dir: str):
    split_state_path = get_split_state_path(model_dir)
    if not os.path.exists(split_state_path):
        return None

    with open(split_state_path, "r", encoding="utf-8") as handle:
        split_state = json.load(handle)

    if not isinstance(split_state, dict):
        raise ValueError(f"Invalid split state file: '{split_state_path}'.")

    train_image_files = split_state.get("train_image_files")
    val_image_files = split_state.get("val_image_files")
    if not isinstance(train_image_files, list) or not isinstance(val_image_files, list):
        raise ValueError(f"Invalid split state file: '{split_state_path}'.")
    if not train_image_files or not val_image_files:
        raise ValueError(f"Split state file is empty: '{split_state_path}'.")

    return split_state


def save_split_state(model_dir: str, split_seed: int, train_image_files, val_image_files):
    split_state_path = get_split_state_path(model_dir)
    split_state = {
        "split_seed": split_seed,
        "train_image_files": list(train_image_files),
        "val_image_files": list(val_image_files),
    }

    with open(split_state_path, "w", encoding="utf-8") as handle:
        json.dump(split_state, handle, indent=2)


def build_train_val_datasets(args, data_transforms, split_state=None, split_seed=None):
    """Builds train and validation datasets from disjoint source image splits."""
    if split_state is None:
        effective_seed = args.seed if split_seed is None else split_seed
        source_image_files = discover_upright_image_files(args.data_dir)
        train_image_files, val_image_files = split_image_files(
            source_image_files, train_ratio=0.8, seed=effective_seed
        )

        logging.info(f"Discovered {len(source_image_files)} source images.")
        logging.info(
            f"Split into Training: {len(train_image_files)} source images, "
            f"Validation: {len(val_image_files)} source images."
        )
    else:
        train_image_files = [os.path.abspath(path) for path in split_state["train_image_files"]]
        val_image_files = [os.path.abspath(path) for path in split_state["val_image_files"]]
        split_seed = split_state.get("split_seed")

        if set(train_image_files) & set(val_image_files):
            raise ValueError("Saved train/validation split contains overlapping source images.")

        missing_files = [
            path
            for path in train_image_files + val_image_files
            if not os.path.exists(path)
        ]
        if missing_files:
            raise FileNotFoundError(
                f"Saved split references {len(missing_files)} missing source images."
            )

        logging.info(
            "Using train/validation split restored from saved state"
            + (f" (seed {split_seed})." if split_seed is not None else ".")
        )
        logging.info(
            f"Restored Training: {len(train_image_files)} source images, "
            f"Validation: {len(val_image_files)} source images."
        )

    if config.USE_CACHE:
        cache_num_workers = None if args.workers == 0 else args.workers
        cache_dataset(
            upright_dir=args.data_dir,
            num_workers=cache_num_workers,
            force_rebuild=args.force_rebuild_cache,
        )
        train_dataset = ImageOrientationDatasetFromCache(
            cache_dir=config.CACHE_DIR,
            source_ids=[build_source_id(path) for path in train_image_files],
            transform=data_transforms["train"],
        )
        val_dataset = ImageOrientationDatasetFromCache(
            cache_dir=config.CACHE_DIR,
            source_ids=[build_source_id(path) for path in val_image_files],
            transform=data_transforms["val"],
        )
        logging.info("Successfully loaded training and validation datasets from cache.")
    else:
        logging.info("Using ON-THE-FLY image processing (caching is disabled).")
        train_dataset = ImageOrientationDataset(
            image_files=train_image_files, transform=data_transforms["train"]
        )
        val_dataset = ImageOrientationDataset(
            image_files=val_image_files, transform=data_transforms["val"]
        )
        logging.info("Successfully loaded dataset for on-the-fly processing.")

    return train_dataset, val_dataset, train_image_files, val_image_files


def train(args):
    """Main training routine."""
    setup_logging()
    training_start_time = time.time()

    logging.info("=================================================")
    logging.info("      STARTING MODEL TRAINING SCRIPT")
    logging.info("=================================================")
    logging.info("Configuration:")
    logging.info(f"  - Using Cache: {config.USE_CACHE}")
    if config.USE_CACHE:
        logging.info(f"  - Cache Directory: {config.CACHE_DIR}")
        logging.info(f"  - Force Rebuild Cache: {args.force_rebuild_cache}")
    logging.info(f"  - Resume from checkpoint: {args.resume}")
    logging.info(f"  - Source Data Directory: {args.data_dir}")
    logging.info(f"  - Model Save Directory: {args.model_dir}")
    logging.info(f"  - Number of Epochs: {args.epochs}")
    logging.info(f"  - Batch Size: {args.batch_size}")
    logging.info(f"  - Learning Rate: {args.lr}")
    logging.info(f"  - Dataloader Workers: {args.workers}")
    logging.info(f"  - Train/Validation Split Seed: {args.seed}")

    writer = SummaryWriter(f"runs/{config.MODEL_NAME}")

    # Ensure model save directory exists
    os.makedirs(args.model_dir, exist_ok=True)

    device = get_device()
    checkpoint_path = os.path.join(args.model_dir, "checkpoint.pth")

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
        logging.info("Float32 matmul precision set to 'high'.")

    # Determine if pin_memory should be used
    pin_memory_enabled = device.type == "cuda"
    if pin_memory_enabled:
        logging.info("CUDA detected, pin_memory will be enabled for DataLoaders.")
    else:
        logging.info("CUDA not detected, pin_memory will be disabled.")

    if device.type == "cuda":
        logging.info("CUDA detected, automatic mixed precision will use bfloat16.")
    elif device.type == "mps":
        logging.info(
            "MPS detected, automatic mixed precision will use float16 with GradScaler."
        )
    else:
        logging.info(
            f"Automatic mixed precision is disabled on {device.type.upper()}; "
            "training will run in FP32."
        )

    channels_last_enabled = should_use_channels_last(device)
    if channels_last_enabled:
        logging.info(
            f"{device.type.upper()} detected, channels-last tensors will be used for image batches."
        )

    ### Dataset and Dataloader logic
    logging.info("\n--- Initializing Dataset and Dataloaders ---")
    data_transforms = get_data_transforms()
    saved_split_state = None
    effective_split_seed = args.seed
    resume_checkpoint = None

    if args.resume:
        try:
            saved_split_state = load_saved_split_state(args.model_dir)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            logging.error(f"Failed to load saved train/validation split: {e}")
            return

        if saved_split_state is not None:
            saved_seed = saved_split_state.get("split_seed")
            if saved_seed is not None:
                effective_split_seed = saved_seed
                if saved_seed != args.seed:
                    logging.warning(
                        f"Saved split was created with seed {saved_seed}. "
                        f"Ignoring CLI seed {args.seed} and reusing the saved split."
                    )
        else:
            logging.warning(
                "Resume was requested but no saved split state was found. "
                "Falling back to checkpoint metadata if available, otherwise the current CLI seed."
            )
            if os.path.exists(checkpoint_path):
                try:
                    resume_checkpoint = torch.load(checkpoint_path, map_location=device)
                    saved_seed = resume_checkpoint.get("split_seed")
                    if saved_seed is not None:
                        effective_split_seed = saved_seed
                        if saved_seed != args.seed:
                            logging.warning(
                                f"Checkpoint was created with split seed {saved_seed}. "
                                f"Ignoring CLI seed {args.seed} and reusing that seed."
                            )
                except Exception as e:
                    logging.warning(
                        f"Could not read split seed from checkpoint metadata: {e}"
                    )

    logging.info(f"  - Effective Train/Validation Split Seed: {effective_split_seed}")

    try:
        train_dataset, val_dataset, train_image_files, val_image_files = (
            build_train_val_datasets(
                args,
                data_transforms,
                split_state=saved_split_state,
                split_seed=effective_split_seed,
            )
        )
    except (ValueError, FileNotFoundError) as e:
        logging.error(f"Failed to initialize dataset: {e}")
        return

    if saved_split_state is None:
        save_split_state(
            args.model_dir,
            split_seed=effective_split_seed,
            train_image_files=train_image_files,
            val_image_files=val_image_files,
        )
        logging.info(
            f"Saved train/validation split state to {get_split_state_path(args.model_dir)}"
        )

    logging.info(
        f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}."
    )

    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": pin_memory_enabled,
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        dataloader_kwargs["prefetch_factor"] = config.DATALOADER_PREFETCH_FACTOR

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_kwargs,
    )
    logging.info("Dataloaders created successfully.")

    logging.info("\n--- Setting up Model ---")
    # Store the original model instance
    original_model = get_orientation_model()
    if channels_last_enabled:
        original_model = original_model.to(
            device=device, memory_format=torch.channels_last
        )
    else:
        original_model = original_model.to(device)

    # This will be the model instance used for training/inference during the loop
    model_for_training = original_model

    # Compile the model for performance if PyTorch 2.0+ is used
    if hasattr(torch, "compile"):
        logging.info("PyTorch 2.0+ detected. Compiling the model for performance...")
        model_for_training = torch.compile(original_model, mode="reduce-overhead")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Add label_smoothing

    # Initialize optimizer with parameters of the model used for training
    optimizer = optim.AdamW(
        model_for_training.parameters(), lr=args.lr, weight_decay=1e-3
    )
    scaler = get_grad_scaler(device)
    logging.info(
        f"Using pre-trained {config.MODEL_NAME} model. Final layers is trainable."
    )
    logging.info(f"Optimizer configured with AdamW, LR={args.lr}, Weight Decay=1e-3")

    # Add scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # --- Checkpoint Loading ---
    start_epoch = 0
    best_val_acc = 0.0
    epochs_no_improve = 0

    if args.resume and os.path.exists(checkpoint_path):
        logging.info(f"\n--- Resuming training from checkpoint: {checkpoint_path} ---")
        try:
            checkpoint = resume_checkpoint
            if checkpoint is None:
                checkpoint = torch.load(checkpoint_path, map_location=device)

            # Load model state
            original_model.load_state_dict(checkpoint["model_state_dict"])

            # Load optimizer and scheduler states
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            scaler_state_dict = checkpoint.get("scaler_state_dict")
            if scaler.is_enabled():
                if scaler_state_dict:
                    scaler.load_state_dict(scaler_state_dict)
                elif scaler_state_dict == {}:
                    logging.info(
                        "Checkpoint has no active GradScaler state; using a fresh scaler."
                    )

            # Load training progress
            start_epoch = checkpoint["epoch"] + 1
            best_val_acc = checkpoint.get("best_val_acc", 0.0)
            epochs_no_improve = checkpoint.get("epochs_no_improve", 0)

            logging.info(
                f"Resumed from epoch {start_epoch}. Best Val Acc: {best_val_acc:.4f}"
            )
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}. Starting from scratch.")
            start_epoch = 0
            best_val_acc = 0.0
    else:
        logging.info("\n--- Starting Training Loop from scratch ---")

    # --- Training Loop ---
    early_stop_patience = 7  # Stop after 7 epochs of no improvement

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        # --- Training Phase ---
        model_for_training.train()
        running_loss, running_corrects = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = move_batch_to_device(
                inputs,
                labels,
                device,
                non_blocking=pin_memory_enabled,
            )
            optimizer.zero_grad(set_to_none=True)

            with get_autocast_context(device):
                outputs = model_for_training(inputs)
                loss = criterion(outputs, labels)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.float() / len(train_dataset)

        # --- Validation Phase ---
        model_for_training.eval()
        val_loss, val_corrects = 0.0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = move_batch_to_device(
                    inputs,
                    labels,
                    device,
                    non_blocking=pin_memory_enabled,
                )

                with get_autocast_context(device):
                    outputs = model_for_training(inputs)
                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.float() / len(val_dataset)

        scheduler.step()

        epoch_duration = time.time() - epoch_start_time

        logging.info(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
            f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Duration: {epoch_duration:.2f}s"
        )

        # --- TensorBoard Logging ---
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", epoch_acc, epoch)
        writer.add_scalar("Loss/validation", val_epoch_loss, epoch)
        writer.add_scalar("Accuracy/validation", val_epoch_acc, epoch)
        writer.add_scalar(
            "Hyperparameters/learning_rate", optimizer.param_groups[0]["lr"], epoch
        )

        # --- MODEL AND CHECKPOINT SAVING LOGIC ---
        current_acc = val_epoch_acc.item()
        if current_acc > best_val_acc:
            best_val_acc = current_acc
            epochs_no_improve = 0  # Reset counter

            # Save the best model (the original, un-compiled version)
            static_save_path = os.path.join(args.model_dir, "best_model.pth")
            torch.save(original_model.state_dict(), static_save_path)

            # Also save a versioned name including the model name and accuracy
            versioned_model_name = f"{config.MODEL_NAME}_{best_val_acc:.4f}.pth"
            versioned_save_path = os.path.join(args.model_dir, versioned_model_name)
            torch.save(original_model.state_dict(), versioned_save_path)

            logging.info(f"   New best model saved! Val Acc: {best_val_acc:.4f}")
            logging.info(
                f"   Model saved as '{static_save_path}' and '{versioned_save_path}'"
            )

        else:
            epochs_no_improve += 1

        # Save checkpoint at the end of every epoch
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": original_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_acc": best_val_acc,
            "epochs_no_improve": epochs_no_improve,
            "split_seed": effective_split_seed,
        }
        torch.save(checkpoint, checkpoint_path)
        logging.debug(f"Checkpoint saved to {checkpoint_path}")

        # --- Check for early stopping ---
        if epochs_no_improve >= early_stop_patience:
            logging.info(
                f"\n--- Early stopping triggered after {early_stop_patience} epochs with no improvement. ---"
            )
            logging.info(
                f"Best validation accuracy was {best_val_acc:.4f} at epoch {epoch - early_stop_patience + 1}."
            )
            break

    # SUMMARY
    total_duration = time.time() - training_start_time
    total_minutes = total_duration / 60
    logging.info("\n=================================================")
    logging.info("              TRAINING COMPLETE")
    logging.info("=================================================")
    logging.info(
        f"Total Training Time: {total_duration:.2f} seconds ({total_minutes:.2f} minutes)"
    )
    if os.path.exists(os.path.join(args.model_dir, "best_model.pth")):
        final_model_name = f"{config.MODEL_NAME}_{best_val_acc:.4f}.pth"
        logging.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
        logging.info(
            f"Final best model saved as 'best_model.pth' and '{final_model_name}'"
        )
    else:
        logging.warning(
            "No model was saved as validation accuracy did not improve from its initial state."
        )
    logging.info("=================================================")

    writer.close()  # Close the TensorBoard writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an image orientation detection model."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=config.DATA_DIR,
        help="Directory with upright images.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=config.MODEL_SAVE_DIR,
        help="Directory to save trained models.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.NUM_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=config.BATCH_SIZE, help="Training batch size."
    )
    parser.add_argument(
        "--lr", type=float, default=config.LEARNING_RATE, help="Learning rate."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=config.NUM_WORKERS,
        help="Number of data loading workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the train/validation split.",
    )
    parser.add_argument(
        "--force-rebuild-cache",
        action="store_true",
        help="If set, clears and rebuilds the image cache.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint.",
    )

    args = parser.parse_args()
    train(args)
