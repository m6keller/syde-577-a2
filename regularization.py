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

def train_on_mnist(
    model, 
    num_epochs=NUM_EPOCHS, 
    batch_size=BATCH_SIZE, 
    learning_rate=LEARNING_RATE, 
    trial: optuna.Trial = None,
    weight_decay: float = 0.0
    ):
    mnist = load_dataset("mnist")

    transform = transforms.Compose([
        transforms.Resize((64, 64)), # Ensure images are 64x64
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # MNIST is grayscale, so only one channel
    ])


    def transform_batch(batch):
        batch['image'] = [transform(img) for img in batch['image']]
        return batch

    train_dataset = mnist["train"].with_transform(transform_batch)
    test_dataset = mnist["test"].with_transform(transform_batch)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay if trial else 0.0)
    
    print("Starting training...")
    for epoch in range(num_epochs):
        print("-" * 30)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        model.train()
        running_loss = 0.0
        
        for data in tqdm(train_loader, desc="Training"):
            images, labels = data['image'].to(device), data['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        model.eval() 
        correct = 0
        total = 0
        with torch.no_grad(): 
            for data in tqdm(test_loader, desc="Validating"):
                images, labels = data['image'].to(device), data['label'].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        if trial is not None:
            trial.report(accuracy, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        print(f"Test Accuracy: {accuracy:.2f}%")
        
    print("-" * 30)
    print("Finished Training!")
    return model, accuracy  

def objective(trial):
    """
    This defines the objective function for Optuna hyperparameter for regularization optimization.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dropout_conv = trial.suggest_float("dropout_conv", 0.0, 0.5, step=0.1)
    dropout_fc = trial.suggest_float("dropout_fc", 0.0, 0.5, step=0.1)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    # Fixed model architecture + dropout
    model = BasicCNN(
        num_conv_layers=3,
        initial_channels=16,
        channel_multiplier=2.0,
        num_classes=10,
        dropout_conv=dropout_conv,
        dropout_fc=dropout_fc
    ).to(device)

    # optimizer weight decay is set in train_on_mnist
    _, final_accuracy = train_on_mnist(model=model, trial=trial, weight_decay=weight_decay)
    return final_accuracy

def main():
    n_trials = 10
    print(f"Starting Optuna regularization optimization for {n_trials} trials...\n")

    with tqdm(total=n_trials, desc="Optuna Trials") as optuna_pbar:
        def callback(study, trial):
            optuna_pbar.update(1)
            optuna_pbar.set_postfix(best_acc=study.best_value)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    print("\nOptimization Complete!")
    best_trial = study.best_trial
    print(f"Best Accuracy: {best_trial.value:.2f}%")
    print("Best Parameters:")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()