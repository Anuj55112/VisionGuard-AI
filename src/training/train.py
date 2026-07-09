import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pathlib import Path

# Package level imports
from src.config import load_config
from src.data.dataset import MedicalSegmentationDataset
from src.models.unet import UNet
from src.models.segformer import SegFormerWrapper
from src.training.eval import evaluate
from src.utils.dice_score import dice_loss
from src.utils.logger import setup_logger

logger = setup_logger("train", "train.log")

def train_model(
    model: nn.Module,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    val_percent: float = 0.2,
    save_checkpoint: bool = True,
    amp: bool = True,
    weight_decay: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    wandb_logging: bool = False,
    wandb_project: str = "visionguard-ai"
):
    # Create dataset
    # We use synthetic data generator if no real images are in standard directories
    dataset = MedicalSegmentationDataset(
        image_dir="./data/imgs/",
        mask_dir="./data/masks/",
        image_size=256,
        is_train=True
    )
    
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    experiment = None
    if wandb_logging:
        try:
            import wandb
            experiment = wandb.init(project=wandb_project, resume='allow')
            experiment.config.update({
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "val_percent": val_percent,
                "amp": amp,
                "weight_decay": weight_decay
            })
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}. Running training without WandB logging.")
            wandb_logging = False

    logger.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
    num_classes = getattr(model, 'n_classes', getattr(model, 'num_classes', 1))
    criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
    
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image'].to(device=device, dtype=torch.float32)
                true_masks = batch['mask'].to(device=device, dtype=torch.float32 if num_classes == 1 else torch.long)

                # Mixed precision context
                with torch.autocast(device.type if device.type != 'mps' and device.type != 'cpu' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if num_classes == 1:
                        # BCE + Dice loss
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(F.sigmoid(masks_pred), true_masks, multiclass=False)
                    else:
                        # CrossEntropy + Dice loss
                        loss = criterion(masks_pred, true_masks.squeeze(1))
                        # one-hot representation for dice calculation
                        one_hot_masks = F.one_hot(true_masks.squeeze(1), num_classes).permute(0, 3, 1, 2).float()
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            one_hot_masks,
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                
                if wandb_logging and experiment:
                    experiment.log({
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # Epoch completed, run validation
        val_score = evaluate(model, val_loader, device, amp)
        scheduler.step(val_score)
        logger.info(f'Epoch {epoch}: Validation Dice Score: {val_score:.4f}')

        if wandb_logging and experiment:
            experiment.log({
                'validation Dice': val_score,
                'epoch': epoch
            })

        if save_checkpoint:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}.pth')
            state_dict = model.state_dict()
            # Include config metadata
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_score,
                'num_classes': num_classes
            }, checkpoint_path)
            logger.info(f'Checkpoint saved at {checkpoint_path}')

    if wandb_logging and experiment:
        experiment.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VisionGuard AI Segmentation Models')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'segformer'], help='Model type to train')
    parser.add_argument('--epochs', type=int, default=None, help='Override training epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Override training batch size')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    
    args = parser.parse_args()
    
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")
    
    epochs = args.epochs if args.epochs is not None else config.epochs
    batch_size = args.batch_size if args.batch_size is not None else config.batch_size
    lr = args.lr if args.lr is not None else config.learning_rate
    
    if args.model == 'unet':
        model = UNet(
            in_channels=config.unet_in_channels,
            num_classes=config.unet_num_classes,
            bilinear=config.unet_bilinear
        )
    else:
        model = SegFormerWrapper(
            pretrained_model_name=config.segformer_name,
            num_classes=config.segformer_num_classes,
            from_pretrained=False
        )
        
    model.to(device=device)
    
    train_model(
        model=model,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        amp=config.mixed_precision,
        checkpoint_dir=config.checkpoint_dir,
        wandb_logging=config.wandb_logging,
        wandb_project=config.wandb_project
    )
