from f1_score_calculation import calculate_f1_score
import torch

def test_model(model, dataloader, criterion, checkpoint_path="best_unet_model.pth"):
    """
    Test the model on the test dataset and compute the F1 score.
    
    Args:
        model (nn.Module): Trained U-Net model.
        dataloader (DataLoader): DataLoader for the test dataset.
        criterion (callable): Loss function (optional for tracking loss).
        checkpoint_path (str): Path to the saved model checkpoint.

    Returns:
        float: Average F1 score on the test dataset.
    """
    # Load the best model
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    model.to("cuda")

    test_loss = 0
    f1_scores = []

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to("cuda"), masks.to("cuda")
            
            # Predict masks
            preds = model(images)
            
            # Compute loss (optional)
            loss = criterion(preds, masks)
            test_loss += loss.item()
            
            # Convert predictions to binary masks
            preds = (preds > 0.5).float()
            
            # Calculate F1 score
            f1 = calculate_f1_score(preds, masks)
            f1_scores.append(f1)

    avg_loss = test_loss / len(dataloader)
    avg_f1_score = sum(f1_scores) / len(f1_scores)
    print(f"Test Loss: {avg_loss:.4f}, Average F1 Score: {avg_f1_score:.4f}")
    return avg_f1_score
