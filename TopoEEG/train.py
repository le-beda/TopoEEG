import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, features_dict, labels):
        self.features = features_dict
        self.labels = labels
        
        first_feature = next(iter(features_dict.values()))
        assert len(first_feature) == len(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = {
            key: value[idx] for key, value in self.features.items()
        }
        return sample, self.labels[idx]

def dict_collate_fn(batch):
    """Custom collate function for dictionary samples"""
    collated = {
        key: torch.stack( [item[key] for item in batch] ) for key in batch[0][0].keys()
    }
    return collated

def train(model, 
          X_train, y_train,
          X_val, y_val,
          num_epochs = 100,
          batch_size = 32
          ):
    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()

    # Define the optimizer
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr)


    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        # collate_fn=dict_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=True,
        # collate_fn=dict_collate_fn
    )


    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() # * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation loop
        model.eval()
        val_loss = 0.0
        predictions_np = []
        y_2_val = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item() # * inputs.size(0)

                # Collect predictions and true labels for metrics
                probabilities = torch.sigmoid(outputs)
                predictions_np.extend((probabilities > 0.5).float().cpu().numpy())
                y_2_val.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)

        # Calculate metrics
        accuracy = accuracy_score(y_2_val, predictions_np)
        f1 = f1_score(y_2_val, predictions_np)

        if epoch in set(np.arange(0, num_epochs, num_epochs-1)) | set([num_epochs-1]):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'\tTrain Loss: {epoch_loss:.4f}')
            
            print(f'\tValidation Loss: {val_loss:.4f}')
            print(f'\tValidation Accuracy: {accuracy:.4f}')
            print(f'\tValidation F1 Score: {f1:.4f}')

    return model
