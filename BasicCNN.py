import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class BasicCNN(nn.Module):
    def __init__(self, num_conv_layers=3, initial_channels=16, channel_multiplier=2, 
                 num_classes=10, dropout_fc=0.0, dropout_conv=0.0):
        
        """
        A basic CNN architecture for MNIST classification.
        Args:
            num_conv_layers (int): Number of convolutional layers.
            initial_channels (int): Number of channels in the first convolutional layer.
            channel_multiplier (float): Factor to multiply channels after each conv layer.
            num_classes (int): Number of output classes.
            dropout_fc (float): Dropout rate for fully connected layers.
            dropout_conv (float): Dropout rate for convolutional layers.
        """
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

            if dropout_conv > 0:
                layers.append(nn.Dropout2d(p=dropout_conv))

            in_channels = current_channels
            current_channels = int(current_channels * channel_multiplier)
            
        self.features = nn.Sequential(*layers)
        
        self.flatten = nn.Flatten()
        if dropout_fc > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_fc),
                nn.LazyLinear(num_classes)
            )
        else:
            self.classifier = nn.LazyLinear(num_classes)


    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits

NUM_EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001

if __name__ == '__main__':
    NUM_CONV_LAYERS = 3
    INITIAL_CHANNELS = 8
    CHANNEL_MULTIPLIER = 2.0


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=-1
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True,
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