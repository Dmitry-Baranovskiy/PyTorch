import torch

def train_model(model, dataloader, loss_fn, optimizer, device, epochs=10):
    model = model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            #Accuracy (поддержка бинарной и многоклассовой классификации)

            if pred.shape[1] == 1:
                predicted = (pred > 0.5).float()
            else:
                predicted = pred.argmax(1)

            correct += (predicted == yb).sum().item()
            total += yb.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss:.4f} | Accuracy: {acc: .4f}")
