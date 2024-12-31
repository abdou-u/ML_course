# Define the training loop
def train_model(model, train_loader, optimizer, criterion, device): 
    """
    Train the model for one epoch
    Args:
        model: the model to train
        train_loader: the DataLoader for training data
        optimizer: the optimizer to use for training
        criterion: the loss function
        device: the device to train on
    Returns:
        train_loss: the average loss over the training set
    """ 
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images) # Outputs: [batch_size, num_classes, height, width]
        
        # Aggregate logits over each patch (e.g., 16x16)
        patch_logits = outputs.view(outputs.size(0), outputs.size(1), -1).mean(dim=-1)
        # Shape: [batch_size, num_classes]

        # Compute loss
        loss = criterion(patch_logits, labels)  # Labels: [batch_size]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Average training loss for this epoch
    return train_loss / len(train_loader)
