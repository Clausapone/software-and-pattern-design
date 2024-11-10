import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix


# {TEST FUNCTION}
# computing and exporting the results as evaluation metrics
def test(model, X_test, Y_test, criterion):

    accuracy = Accuracy(task="binary")
    precision = Precision(task="binary")
    recall = Recall(task="binary")
    f1_score = F1Score(task="binary")
    conf_mat = ConfusionMatrix(task="binary")

    # computing the predictions in evaluating mode of pytorch
    model.eval()
    with torch.no_grad():
        preds = model.forward(X_test)
        loss = criterion(preds, Y_test)
        preds = torch.round(preds)

        accuracy(preds, Y_test)
        precision(preds, Y_test)
        recall(preds, Y_test)
        f1_score(preds, Y_test)
        conf_mat(preds, Y_test)

        # Ottenere il valore delle metriche
        accuracy = accuracy(preds, Y_test)
        precision = precision(preds, Y_test)
        recall = recall(preds, Y_test)
        f1_score = f1_score(preds, Y_test)
        conf_mat = conf_mat(preds, Y_test)

    return loss.item(), accuracy.item(), precision.item(), recall.item(), f1_score.item(), conf_mat
