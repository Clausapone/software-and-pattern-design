import numpy as np

# {TRAINING FUNCTION}
# training the model over epochs storing also the loss_history
def train(model, X, Y, train_mask, optimizer, loss_criterion, epochs):
    model.train()
    loss_history = np.array([])
    for epoch in range(epochs):
        preds = model.forward(X)
        loss = loss_criterion(preds[train_mask], Y[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history = np.append(loss_history, loss.item())

    return loss_history
