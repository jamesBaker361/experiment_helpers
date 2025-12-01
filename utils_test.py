
import unittest
import torch
from torch.utils.data import DataLoader,random_split,TensorDataset
from torchvision import datasets, transforms
import torch.nn.functional as F

def get_mnist_data(batch_size:int):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # --- 1. Load full training dataset ---
    full_train = datasets.MNIST(
        root="./data",
        train=True,
        transform=transform,
        download=True
    )

    # --- 2. Split into train + val ---
    train_size = int(0.8 * len(full_train))   # 80% train
    val_size   = int(0.2 * len(full_train))   # 20% val

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size], generator=generator)

    # --- 3. Load official test set ---
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        transform=transform,
        download=True
    )

    # --- 4. DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader,val_loader,test_loader

def get_mnist_model():
    model= torch.nn.Sequential(*[
            torch.nn.Conv2d(1,1,4,2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(13*13,32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,10)
        ])
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.weight)
            m.bias.data.fill_(0.0)
        elif isinstance(m,torch.nn.Conv2d):
            torch.nn.init.zeros_(m.weight)
            m.bias.data.fill_(0.0)
            
    #model.apply(init_weights)
    return model

def get_mnist_autoencoder_model():
    model= torch.nn.Sequential(*[
            torch.nn.Conv2d(1,4,4,2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4,8,4,2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8,16,4,2,),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16,8,4,2,),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8,4,4,2,output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(4,1,4,2,output_padding=0),
        ])
    
    return model
    

def get_regression_data(batch_size):
    N = 1000      # number of samples
    D_in = 5      # input features
    D_out = 1     # target dimension

    torch.manual_seed(42)

    X = torch.randn(N, D_in)
    true_w = torch.tensor([[2.0], [-1.0], [0.5], [3.0], [-2.0]])
    true_b = 1.0

    y = X @ true_w + true_b + 0.1*torch.randn(N, D_out)

    # --- 2. Wrap in TensorDataset ---
    dataset = TensorDataset(X, y)

    # --- 3. Split into train/val/test ---
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # --- 4. Create DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader,val_loader,test_loader

def get_regression_model():
    return torch.nn.Linear(5,1)