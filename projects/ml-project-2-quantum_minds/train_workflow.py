import torch
import time
from train_model import train_model
from validate_model import validate_model


# Define the training workflow
def train_workflow(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler,
    device,
    num_epochs,
    patience,
    save_path,
):
    """
    Train and validate the model, saving the best model based on F1 score.
    
    Args:
        model: The UNet model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer for training.
        criterion: Loss function.
        scheduler: Learning rate scheduler.
        device: Device to train on (CPU or CUDA).
        num_epochs: Number of training epochs.
        patience: Early stopping patience (number of epochs to wait for improvement).
        save_path: Path to save the best model.
    
    Returns:
        None
    """
    best_f1 = 0.0
    epochs_no_improve = 0  # Counter for early stopping
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Training phase
        train_loss = train_model(model, train_loader, optimizer, criterion, device)

        # Validation phase
        val_loss, val_f1, val_acc = validate_model(model, val_loader, device, criterion)

        # Update the scheduler with validation f1 score
        scheduler.step(val_f1)

        # Save the best model based on validation F1 score
        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch}: New best F1 score: {val_f1:.4f} - Model saved!")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs. Best F1: {best_f1:.4f}")
            break

        # epoch details
        elapsed_time = time.time() - start_time
        print(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Val Accuracy: {val_acc:.4f} | "
            f"Time: {elapsed_time:.2f}s"
        )

        # Append metrics to history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)

    print(f"Training completed. Best F1 Score: {best_f1:.4f}")
    return history