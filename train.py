import torch
import matplotlib.pyplot as plt
import optuna

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm
from optuna.trial import Trial

from BasicCNN import BasicCNN

# TRAINING PARAMS
NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# MODEL PARAMS
NUM_CONVOLUTIONAL_LAYERS = 4
INITIAL_CHANNELS = 32
CHANNEL_MULTIPLIER = 2.0
NUM_CLASSES = 10

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

def train_on_mnist(
    model, 
    num_epochs=NUM_EPOCHS, 
    batch_size=BATCH_SIZE, 
    learning_rate=LEARNING_RATE, 
    transform = get_transform(),
    trial: Trial = None
    ):
    mnist = load_dataset("mnist")

    def transform_batch(batch):
        batch['image'] = [transform(img) for img in batch['image']]
        return batch

    train_dataset = mnist["train"].with_transform(transform_batch)
    test_dataset = mnist["test"].with_transform(transform_batch)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)

    train_loss_history = []
    accuracy_history = []
    
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

        train_loss_history.append(avg_train_loss)

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
    
    plt.plot(range(1, num_epochs + 1), accuracy_history)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy over Epochs')
    plt.savefig('accuracy_over_epochs.png')

    plt.plot(range(1, num_epochs + 1), train_loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss over Epochs')
    plt.savefig('training_loss_over_epochs.png')
        
    print("-" * 30)
    print("Finished Training!")
    return model, accuracy  


def main():
    model = BasicCNN(
        num_conv_layers=NUM_CONVOLUTIONAL_LAYERS,
        initial_channels=INITIAL_CHANNELS,
        channel_multiplier=CHANNEL_MULTIPLIER,
        num_classes=NUM_CLASSES
    )

    train_on_mnist(model)

if __name__ == "__main__":
    main()