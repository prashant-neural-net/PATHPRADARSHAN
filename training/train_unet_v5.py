import os
import random
import numpy as np
import rasterio
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from model import UNet


# ============================================================
# CONFIG
# ============================================================

IMAGE_DIR = "dataset_v3_original/images"
MASK_DIR = "dataset_v3_original/masks"

BATCH_SIZE = 8
EPOCHS = 60
LR = 1e-4
SEED = 42

PATIENCE = 12
MODEL_PATH = "road_unet_v5_best.pth"


DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DATASET
# ============================================================

class RoadDataset(Dataset):

    def __init__(self, image_dir, mask_dir, augment=False):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment = augment

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".tif")
        ])

        self.masks = sorted([
            f for f in os.listdir(mask_dir)
            if f.endswith(".tif")
        ])

        assert len(self.images) == len(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_path = os.path.join(
            self.image_dir,
            self.images[idx]
        )

        mask_path = os.path.join(
            self.mask_dir,
            self.masks[idx]
        )

        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)

        image /= 10000.0

        if self.augment:

            if random.random() < 0.5:
                image = np.flip(
                    image, axis=2
                ).copy()
                mask = np.flip(
                    mask, axis=1
                ).copy()

            if random.random() < 0.5:
                image = np.flip(
                    image, axis=1
                ).copy()
                mask = np.flip(
                    mask, axis=0
                ).copy()

            if random.random() < 0.5:

                k = random.randint(1, 3)

                image = np.rot90(
                    image,
                    k=k,
                    axes=(1, 2)
                ).copy()

                mask = np.rot90(
                    mask,
                    k=k,
                    axes=(0, 1)
                ).copy()

        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        return (
            torch.from_numpy(image).float(),
            torch.from_numpy(mask).float().unsqueeze(0)
        )


# ============================================================
# DATA SPLIT
# EXACT SAME SEED / SPLIT AS V3
# ============================================================

full_dataset = RoadDataset(
    IMAGE_DIR,
    MASK_DIR,
    augment=False
)

train_size = int(
    0.8 * len(full_dataset)
)

val_size = (
    len(full_dataset)
    - train_size
)

train_base, val_dataset = torch.utils.data.random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

aug_dataset = RoadDataset(
    IMAGE_DIR,
    MASK_DIR,
    augment=True
)

train_dataset = Subset(
    aug_dataset,
    train_base.indices
)

print("Total samples:", len(full_dataset))
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = UNet().to(DEVICE)

print("Model: U-Net")
print("Input channels: 4")


# ============================================================
# FOCAL LOSS
# ============================================================

class BinaryFocalLoss(nn.Module):

    def __init__(
        self,
        alpha=0.75,
        gamma=2.0
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits,
        target
    ):

        probs = torch.sigmoid(
            logits
        )

        eps = 1e-6

        probs = torch.clamp(
            probs,
            eps,
            1.0 - eps
        )

        pt = (
            target * probs
            +
            (1.0 - target)
            * (1.0 - probs)
        )

        alpha_t = (
            target * self.alpha
            +
            (1.0 - target)
            * (1.0 - self.alpha)
        )

        focal = (
            -alpha_t
            * (1.0 - pt) ** self.gamma
            * torch.log(pt)
        )

        return focal.mean()


# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(
    logits,
    target
):

    probs = torch.sigmoid(
        logits
    )

    probs = probs.reshape(
        probs.shape[0],
        -1
    )

    target = target.reshape(
        target.shape[0],
        -1
    )

    smooth = 1e-6

    intersection = (
        probs * target
    ).sum(dim=1)

    denominator = (
        probs.sum(dim=1)
        +
        target.sum(dim=1)
    )

    dice = (
        2.0 * intersection
        + smooth
    ) / (
        denominator
        + smooth
    )

    return 1.0 - dice.mean()


focal_loss = BinaryFocalLoss(
    alpha=0.75,
    gamma=2.0
)


def combined_loss(
    logits,
    target
):

    return (
        0.5 * focal_loss(
            logits,
            target
        )
        +
        0.5 * dice_loss(
            logits,
            target
        )
    )


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)


# ============================================================
# TRAINING
# ============================================================

best_dice = -1.0
epochs_without_improvement = 0

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0

    for images, masks in train_loader:

        images = images.to(
            DEVICE
        ).contiguous()

        masks = masks.to(
            DEVICE
        ).contiguous()

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            images
        ).contiguous()

        loss = combined_loss(
            outputs,
            masks
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(
        train_loader
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    intersection = 0.0
    prediction_sum = 0.0
    target_sum = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(
                DEVICE
            ).contiguous()

            masks = masks.to(
                DEVICE
            ).contiguous()

            outputs = model(
                images
            ).contiguous()

            loss = combined_loss(
                outputs,
                masks
            )

            val_loss += loss.item()

            probs = torch.sigmoid(
                outputs
            )

            predictions = (
                probs >= 0.6
            ).float()

            intersection += (
                predictions * masks
            ).sum().item()

            prediction_sum += (
                predictions.sum().item()
            )

            target_sum += (
                masks.sum().item()
            )

    val_loss /= len(
        val_loader
    )

    val_dice = (
        2.0 * intersection
    ) / (
        prediction_sum
        +
        target_sum
        +
        1e-6
    )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if val_dice > best_dice:

        best_dice = val_dice
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        marker = " ← BEST"

    else:

        epochs_without_improvement += 1
        marker = ""


    print(
        f"Epoch [{epoch+1:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Dice: {val_dice:.4f}"
        f"{marker}"
    )


    if epochs_without_improvement >= PATIENCE:

        print(
            f"Early stopping at epoch {epoch+1}"
        )

        break


print()
print("==============================")
print("U-NET V5 COMPLETE")
print("==============================")
print(
    "Best validation Dice:",
    round(best_dice, 4)
)
print(
    "Saved:",
    MODEL_PATH
)
