import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class BasicCNN(nn.Module):
    def __init__(self, num_conv_layers=3, initial_channels=16, channel_multiplier=2, num_classes=10):
        super(BasicCNN, self).__init__()
        
        layers = []
        in_channels = 1 # MNIST is grayscale

        current_channels = initial_channels
        for _ in range(num_conv_layers):
            # Add a convolutional block: Conv -> ReLU -> MaxPool
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=current_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            in_channels = current_channels
            current_channels = int(current_channels * channel_multiplier)
            
        self.features = nn.Sequential(*layers)
        
        self.flatten = nn.Flatten()
        self.classifier = nn.LazyLinear(num_classes)


    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits


if __name__ == '__main__':
    # --- Configuration ---
    NUM_CONV_LAYERS = 3       # Depth: How many conv blocks
    INITIAL_CHANNELS = 8      # Width: Channels in the first layer
    CHANNEL_MULTIPLIER = 2.0  # Width: How much to increase channels by

    # Training parameters
    NUM_EPOCHS = 5
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=-1
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    model = BasicCNN(
        num_conv_layers=NUM_CONV_LAYERS,
        initial_channels=INITIAL_CHANNELS,
        channel_multiplier=CHANNEL_MULTIPLIER,
        num_classes=10
    ).to(device)
    
    # Generate a dummy input tensor with the correct shape: (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 1, 64, 64).to(device)
    model(dummy_input) # This call initializes the lazy layer