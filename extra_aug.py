import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from BasicCNN import BasicCNN

# --- constants ---
NUM_EPOCHS = 10   # slightly longer training
BATCH_SIZE = 64
LEARNING_RATE = 0.001


def get_best_transform():
    """Return improved augmentation transform based on best Optuna parameters + new helpful aug."""
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomAffine(
            degrees=20.6,
            translate=(0.05, 0.05),
            scale=(0.96, 1.05),
            shear=4.4
        ),
        transforms.ColorJitter(contrast=0.2),  # slight contrast jitter helps robustness
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomErasing(
            p=0.32,
            scale=(0.04, 0.08),
            ratio=(0.7, 1.3)
        ),
    ])


def get_test_transform():
    """Simpler transform for test time."""
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def train_and_evaluate(model, train_loader, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

    transform = get_best_transform()
    test_transform = get_test_transform()

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BasicCNN(
        num_conv_layers=3,
        initial_channels=8,
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
