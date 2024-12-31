from sklearn.metrics import f1_score

def calculate_f1_score(preds, targets):
    """
    Calculate the F1 score for the predicted and ground truth masks.
    
    Args:
        preds (torch.Tensor): Predicted binary masks (N, H, W).
        targets (torch.Tensor): Ground truth binary masks (N, H, W).

    Returns:
        float: F1 score.
    """
    preds = preds.cpu().numpy().astype(int).flatten()
    targets = targets.cpu().numpy().astype(int).flatten()
    return f1_score(targets, preds)