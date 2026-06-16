import torch
import torch.nn as nn
import torch.optim as optim

from config import *
from data.dataloader import get_dataloaders
from data.transform import(get_test_transform, get_train_transform)

from models.mlp import MLP
from utils.trainer import fit, evaluate
from utils.plot import plot_history

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir = DATA_DIR,
        classes = CLASSES,
        samples_per_class = IMAGES_PER_CLASS,
        batch_size = BATCH_SIZE,
        train_transform = get_train_transform(),
        test_transform = get_test_transform(),
    )

    model = MLP(
        input_dim = 784, 
        hidden_dims = [512, 256, 128],
        num_classes = len(CLASSES),
        activation = ACTIVATION,
        use_batchnorm = BATCH_NORM,
        dropout_p = DROPOUT,
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(), lr = LEARNING_RATE,
        weight_decay = 1e-4
    )

    history = fit(
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,
        optimizer = optimizer,
        criterion = criterion,
        device = device,
        epochs = EPOCHS,
    )

    plot_history(history)

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc * 100:.4f}")


if __name__ == "__main__":
    main()
