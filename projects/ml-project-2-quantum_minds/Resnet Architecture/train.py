import os
import json
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loss import DiceBCELoss
from model import UnetLikeSegmentatorModel
from dataset import MRDDataset, JointTransform
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score


def train_model(model, train_loader, val_loader, num_epochs=25, lr=1e-4, checkpoint_path='saved_model/best_model.pth', metrics_path='training_metrics.json'):
    """
    Trains a given model using the specified data loaders, optimizer, and loss function, while tracking the best validation score
    and saving the best model.
    
    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        num_epochs (int, optional): Number of epochs to train the model. Defaults to 25.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
        checkpoint_path (str, optional): Path to save the best model. Defaults to 'saved_model/best_model.pth'.
    """

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define loss function and optimizer
    criterion = DiceBCELoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-7)

    # TensorBoard writer
    writer = SummaryWriter()

    # Track metrics
    train_losses = []
    val_losses = []
    val_f1_scores = []

    # Track the best validation score
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Training phase
        for batch_i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Normalize labels to [0, 1] if necessary
            labels = labels.float() / labels.max()

            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track training loss
            running_loss += loss.item() * inputs.size(0)

            print(f"Epoch {epoch+1}, Batch {batch_i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        writer.add_scalar('Training Loss', epoch_loss, epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Normalize labels to [0, 1] if necessary
                labels = labels.float() / labels.max()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                # Convert predictions to binary (threshold 0.5)
                predictions = (outputs > 0.5).float()

                # Collect for F1 score
                all_labels.append(labels.cpu().numpy().flatten())
                all_predictions.append(predictions.cpu().numpy().flatten())

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        writer.add_scalar('Validation Loss', val_loss, epoch)

        # Compute F1 score
        all_labels = np.concatenate(all_labels)
        all_predictions = np.concatenate(all_predictions)

        # Convert labels to binary if they are not already
        all_labels = (all_labels > 0.5).astype(int)
        f1 = f1_score(all_labels, all_predictions, average='binary')
        val_f1_scores.append(f1)
        writer.add_scalar('Validation F1 Score', f1, epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation F1 Score: {f1:.4f}")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved with Validation Loss: {best_val_loss:.4f}")

        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)

    print("Training complete. Best model saved at:", checkpoint_path)
    writer.close()

    # Save metrics to a JSON file
    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_f1_scores": val_f1_scores
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    print(f"Training metrics saved to {metrics_path}")

    return train_losses, val_losses, val_f1_scores


if __name__ == '__main__':
    # Define the path to the JSON configuration file
    config_file_path = 'config/config.json'

    # Open and read the JSON configuration file
    with open(config_file_path, 'r') as file:
        config = json.load(file)

    # Define joint transformations for training (augmentations)
    #joint_transform_train = transforms.Compose([
    #    transforms.RandomRotation(degrees=30),
    #    transforms.RandomHorizontalFlip(),
    #    transforms.ToTensor()
    #])

    # Define joint transformations for validation/testing (no augmentations)
    #joint_transform_eval = transforms.Compose([
    #    transforms.ToTensor()
    #])

    # Define image-specific transformations (e.g., normalization)
    #image_transform = transforms.Compose([
    #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #])

    # Combine joint and image-specific transformations
    #train_transformations = JointTransform(joint_transform=joint_transform_train, image_transform=image_transform)
    #eval_transformations = JointTransform(joint_transform=joint_transform_eval, image_transform=image_transform)

    # Load training dataset (No transformations applied as the dataset is already preprocessed)
    train_ds = MRDDataset(
        image_dir=os.path.join(config['data_dir'], 'train/images'),
        label_dir=os.path.join(config['data_dir'], 'train/groundtruth'),
        images_wh=tuple(config['dataset_image_size']),  # Ensure the size matches the dataset
        transforms=None  # No transformations
    )
    dataloader_train = DataLoader(dataset=train_ds, batch_size=config["train_batch_size"], shuffle=True, num_workers=2)
    print(f"Training dataset loaded: {len(train_ds)} samples, {len(dataloader_train)} batches.")

    # Load validation dataset (No transformations applied as the dataset is already preprocessed)
    val_ds = MRDDataset(
        image_dir=os.path.join(config['data_dir'], 'validation/images'),
        label_dir=os.path.join(config['data_dir'], 'validation/groundtruth'),
        images_wh=tuple(config['dataset_image_size']),  # Ensure the size matches the dataset
        transforms=None  # No transformations
    )
    dataloader_val = DataLoader(dataset=val_ds, batch_size=config["validation_batch_size"], shuffle=False, num_workers=2)
    print(f"Validation dataset loaded: {len(val_ds)} samples, {len(dataloader_val)} batches.")

    # Initialize the model
    model = UnetLikeSegmentatorModel()

    # Train the model (No test dataset, only train and validation)
    print("Starting training...")
    train_losses, val_losses, val_f1_scores = train_model(
        model=model,
        train_loader=dataloader_train,
        val_loader=dataloader_val,
        num_epochs=config['train_max_epoch'],
        lr=config['train_init_lr'],
        checkpoint_path=config['train_save_dir']
    )
    print("Training and validation complete. Best model saved to:", config['train_save_dir'])
