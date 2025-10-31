import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from BasicCNN import BasicCNN

# --- constants ---
NUM_EPOCHS = 5   
BATCH_SIZE = 64
LEARNING_RATE = 0.001


# AUG PARAMS
ROTATION_DEG = 14.77
TRANSLATE_X = 0.0771
TRANSLATE_Y = 0.1117
SCALE_MIN = 0.892
SCALE_MAX = 1.084
SHEAR_DEG = 14.81
ERASE_PROB = 0.236
ERASE_SCALE = 0.108
ERASE_RATIO = 0.688


def get_transform(
    rotation_deg=ROTATION_DEG,
    translate_x=TRANSLATE_X,
    translate_y=TRANSLATE_Y,
    scale_min=SCALE_MIN,
    scale_max=SCALE_MAX,
    shear_deg=SHEAR_DEG,
    erase_prob=ERASE_PROB,
    erase_scale=ERASE_SCALE,
    erase_ratio=ERASE_RATIO
):
    """Define augmentation pipeline with Optuna-sampled hyperparameters."""
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomAffine(
            degrees=rotation_deg,
            translate=(translate_x, translate_y),
            scale=(scale_min, scale_max),
            shear=shear_deg
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomErasing(
            p=erase_prob,
            scale=(erase_scale, erase_scale * 2),
            ratio=(erase_ratio / 2, erase_ratio)
        ),
    ])
    return transform


def get_test_transform():
    """Simpler transform for test time."""
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def train_and_evaluate(model, train_loader, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]", leave=False)
        running_loss = 0.0
        for data in train_pbar:
            images, labels = data['image'].to(device), data['label'].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Avg Training Loss = {avg_loss:.4f}")

        # --- Evaluate ---
        acc = test_with_tta(model, test_loader, device)
        print(f"Validation Accuracy after epoch {epoch+1}: {acc:.2f}%")

    return acc


def test_with_tta(model, test_loader, device, n_augmentations=5):
    """Apply test-time augmentation (TTA) by averaging predictions across random transforms."""
    model.eval()
    correct = 0
    total = 0

    # Create augmentation for TTA
    tta_transform = transforms.RandomAffine(
        degrees=10, translate=(0.03, 0.03), scale=(0.97, 1.03)
    )

    with torch.no_grad():
        val_pbar = tqdm(test_loader, desc="Evaluating (TTA)", leave=False)
        for data in val_pbar:
            images, labels = data['image'].to(device), data['label'].to(device)

            # Average predictions across multiple augmented versions
            outputs_sum = 0
            for _ in range(n_augmentations):
                augmented = tta_transform(images)
                outputs_sum += model(augmented)

            outputs_mean = outputs_sum / n_augmentations
            _, predicted = torch.max(outputs_mean.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_pbar.set_postfix(acc=100 * correct / total)

    return 100 * correct / total


def main():
    mnist = load_dataset("mnist")

    transform = get_transform()
    test_transform = get_transform() 

    def transform_batch(batch):
        batch['image'] = [transform(img) for img in batch['image']]
        return batch

    def transform_batch_test(batch):
        batch['image'] = [test_transform(img) for img in batch['image']]
        return batch

    train_dataset = mnist["train"].with_transform(transform_batch)
    test_dataset = mnist["test"].with_transform(transform_batch_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BasicCNN(
        num_conv_layers=4,
        initial_channels=32,
        channel_multiplier=2.0,
        num_classes=10
    ).to(device)

    print("Starting training with enhanced augmentation...")
    final_acc = train_and_evaluate(model, train_loader, test_loader, device)

    print(f"\n✅ Final Model Accuracy with Augmentation + TTA: {final_acc:.2f}%")

    torch.save(model.state_dict(), "best_cnn_mnist_tta.pth")
    print("Model saved as best_cnn_mnist_tta.pth")


if __name__ == "__main__":
    main()
