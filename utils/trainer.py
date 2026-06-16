import torch


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim = 1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim = 1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def fit(model, train_loader, val_loader, optimizer, criterion, device, epochs,):
    
    history = {
        "train_loss" : [],
        "train_acc" : [],
        "val_loss" : [],
        "val_acc" : [],
    }

    for epoch in range(epochs):
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_loss , val_acc = evaluate(
            model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f'Epoch[{epoch + 1}/{epochs}] | '
            f"Train Loss : {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.4f}"
        )
    return history
