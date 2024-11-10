import numpy as np

# {TRAINING FUNCTION}
# training the model over epochs storing also the loss_history
def train(model, X_train, Y_train, optimizer, loss_criterion, epochs):
    model.train()
    loss_history = np.array([])
    for epoch in range(epochs):
        optimizer.zero_grad()

        preds = model.forward(X_train)
        loss = loss_criterion(preds, Y_train)

        loss.backward()
        optimizer.step()

        loss_history = np.append(loss_history, loss.item())

    return loss_history
