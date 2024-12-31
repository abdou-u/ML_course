import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def validate_model(model, dataloader, device, criterion, patch_size=16):
    """
    Validate the model on the validation dataset.

    Args:
        model: The UNet model to validate.
        dataloader: The DataLoader for validation data.
        device: The device to validate on (CPU or CUDA).
        criterion: The loss function.
        patch_size: Size of patches (e.g., 16x16).

    Returns:
        avg_loss: Average loss over the validation set.
        f1: The F1 score for patch-level predictions.
        accuracy: The accuracy for patch-level predictions.
    """
    model.eval()
    all_preds, all_labels = [], []
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass: model output is pixel-level logits
            outputs = model(images)  # Shape: [batch_size, num_classes, height, width]

            # Aggregate logits over patches to obtain patch-level predictions
            patch_logits = outputs.view(outputs.size(0), outputs.size(1), -1).mean(dim=-1)  # [batch_size, num_classes]
            val_loss += criterion(patch_logits, labels).item()

            # Predictions for each patch
            preds = patch_logits.argmax(dim=1).cpu().numpy()  # [batch_size]
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())  # [batch_size]

    # Concatenate all batches for metric calculation
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate F1 score and accuracy for patch-level predictions
    f1 = f1_score(all_labels, all_preds, average="binary")
    accuracy = accuracy_score(all_labels, all_preds)

    avg_loss = val_loss / len(dataloader)
    return avg_loss, f1, accuracy