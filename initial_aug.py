import torch
import numpy as np
from tqdm import tqdm
import optuna

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

from BasicCNN import BasicCNN

NUM_EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001


def get_transform(trial):
    """Define augmentation pipeline with Optuna-sampled hyperparameters."""
    rotation_deg = trial.suggest_float("rotation_deg", 0, 30)
    translate_x = trial.suggest_float("translate_x", 0.0, 0.2)
    translate_y = trial.suggest_float("translate_y", 0.0, 0.2)
    scale_min = trial.suggest_float("scale_min", 0.8, 1.0)
    scale_max = trial.suggest_float("scale_max", 1.0, 1.2)
    shear_deg = trial.suggest_float("shear_deg", 0.0, 15.0)
    erase_prob = trial.suggest_float("erase_prob", 0.0, 0.5)
    erase_scale = trial.suggest_float("erase_scale", 0.02, 0.15)
    erase_ratio = trial.suggest_float("erase_ratio", 0.3, 3.3)

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


def train_and_evaluate(model, train_loader, test_loader, device):
    """Train the CNN and return final test accuracy."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]", leave=False)
        for data in train_pbar:
            images, labels = data['image'].to(device), data['label'].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_pbar.set_postfix(loss=loss.item())

    # --- Evaluate accuracy ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        val_pbar = tqdm(test_loader, desc="Validating", leave=False)
        for data in val_pbar:
            images, labels = data['image'].to(device), data['label'].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_pbar.set_postfix(correct=correct, total=total)

    accuracy = 100 * correct / total
    return accuracy


def objective(trial):
    """Objective function for Optuna — maximizes validation accuracy."""
    mnist = load_dataset("mnist")

    transform = get_transform(trial)

    def transform_batch(batch):
        batch['image'] = [transform(img) for img in batch['image']]
        return batch

    train_dataset = mnist["train"].with_transform(transform_batch)

    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    def transform_batch_test(batch):
        batch['image'] = [test_transform(img) for img in batch['image']]
        return batch

    test_dataset = mnist["test"].with_transform(transform_batch_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicCNN(
        num_conv_layers=3,
        initial_channels=8,
        channel_multiplier=2.0,
        num_classes=10
    ).to(device)

    accuracy = train_and_evaluate(model, train_loader, test_loader, device)
    return accuracy


def main():
    n_trials = 15
    print(f"Starting Optuna optimization for {n_trials} trials...\n")

    # Create progress bar for all Optuna trials
    with tqdm(total=n_trials, desc="Optuna Trials") as optuna_pbar:
        def callback(study, trial):
            optuna_pbar.update(1)
            optuna_pbar.set_postfix(best_acc=study.best_value)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=3600, callbacks=[callback])

    print("\nOptimization Complete!")
    print("Best Trial:")
    trial = study.best_trial
    print(f"  Accuracy: {trial.value:.2f}%")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Optional: retrain final model on best augmentation settings
    print("\nRetraining final model with best augmentation...")
    best_transform = get_transform(trial)
    mnist = load_dataset("mnist")

    def transform_batch(batch):
        batch['image'] = [best_transform(img) for img in batch['image']]
        return batch

    train_dataset = mnist["train"].with_transform(transform_batch)
    test_dataset = mnist["test"].with_transform(transform_batch)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicCNN(num_conv_layers=3, initial_channels=8,
                     channel_multiplier=2.0, num_classes=10).to(device)

    final_acc = train_and_evaluate(model, train_loader, test_loader, device)
    print(f"\nFinal retrained model accuracy: {final_acc:.2f}%")

    # Save the final model
    torch.save(model.state_dict(), "best_cnn_mnist.pth")
    print("Model saved as best_cnn_mnist.pth")


if __name__ == "__main__":
    main()
