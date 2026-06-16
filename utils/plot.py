import matplotlib.pyplot as plt


def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    
#Loss Plot
    plt.figure(figsize = (8,5))

    plt.plot(epochs, history["train_loss"], label = "Train Loss")
    plt.plot(epochs, history["val_loss"], label = "Validation loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#Accuracy Plot
    plt.figure(figsize = (8,5))

    plt.plot(epochs, history["train_acc"], label = "Train Accuracy")
    plt.plot(epochs, history["val_acc"], label = "Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

