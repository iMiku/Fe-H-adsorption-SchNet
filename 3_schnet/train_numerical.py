from schnet_model import AtomData2Img64
from schnet_dataloader import CustomDataset
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
import numpy as np

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    excluded_keys = ["file_name"]
    for batch_data in train_loader:
        batch_targets_list = []
        batch_outputs_list = []
        for idx, datainfo in enumerate(batch_data):
            atom_info = datainfo[0]
            # Move the atom_info data to the GPU
            atom_info = {key: val.to(device) if key not in excluded_keys else torch.tensor(0).to(device) for key, val in atom_info.items()}
            
            # Append model outputs to the list (ensure it runs on GPU)
            batch_outputs_list.append(model(atom_info))
            batch_targets_list.append(torch.tensor(datainfo[1],dtype=torch.float32))
        
        # Stack the list into a tensor (along the first dimension)
        batch_outputs = torch.stack(batch_outputs_list).to(device)
        batch_targets = torch.stack(batch_targets_list).to(device)
        # Ensure that batch_outputs requires gradients if needed
        batch_outputs.requires_grad_()

        # Compute loss
        loss = criterion(batch_outputs, batch_targets)
        running_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters
    return running_loss/len(train_loader)
        

# Function to validate the model
def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    excluded_keys = ["file_name"]
    with torch.no_grad():
        for batch_data in val_loader:
            batch_targets_list = []
            batch_outputs_list = []
            for idx, datainfo in enumerate(batch_data):
                atom_info = datainfo[0]
                # Move the atom_info data to the GPU
                atom_info = {key: val.to(device) if key not in excluded_keys else torch.tensor(0).to(device) for key, val in atom_info.items()}
                
                # Append model outputs to the list (ensure it runs on GPU)
                batch_outputs_list.append(1.0 * model(atom_info))
                batch_targets_list.append(torch.tensor(datainfo[1],dtype=torch.float32))
            
            # Stack the list into a tensor (along the first dimension)
            batch_outputs = torch.stack(batch_outputs_list).to(device)
            batch_targets = torch.stack(batch_targets_list).to(device)
            loss = criterion(batch_outputs, batch_targets)
            running_loss += loss.item()
    return running_loss / len(val_loader)

# Function to test the model
def test_model(model, test_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    all_outputs = []  # To store all model outputs
    all_targets = []  # To store all true target values
    running_loss = 0.0
    excluded_keys = ["file_name"]
    with torch.no_grad():  # Disable gradient calculation during testing
        for batch_data in test_loader:
            batch_targets_list = []  # List to collect targets for this batch
            batch_outputs_list = []  # List to collect model outputs for this batch
            
            for datainfo in batch_data:
                atom_info = datainfo[0]
                
                # Move atom_info and target data to GPU (or device)
                atom_info = {key: val.to(device) if key not in excluded_keys else torch.tensor(0).to(device) for key, val in atom_info.items()}
                target = torch.tensor(datainfo[1], dtype=torch.float32).to(device)
                
                # Get model output (forward pass) and append to list
                output = model(atom_info)
                batch_outputs_list.append(output)

                # Convert target values (log-transformation) and append
                batch_targets_list.append(target)

            # Convert lists to tensors and accumulate for all batches
            all_outputs.append(torch.cat(batch_outputs_list))
            all_targets.append(torch.cat(batch_targets_list))
            
            # Calculate loss for this batch (optional, modify as needed)
            loss = criterion(torch.cat(batch_outputs_list), torch.cat(batch_targets_list))
            running_loss += loss.item()

    # Concatenate outputs and targets across all batches
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    
    return all_outputs, all_targets, running_loss / len(test_loader)


def value_to_one_hot_batch(values, bins=[1e-3, 1e-2, 1e-1, 0, 1e1]):
    # Convert the bin edges into a sorted tensor
    sorted_bins = torch.tensor(sorted(bins), dtype=torch.float32)
    
    # Ensure values is a tensor
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values, dtype=torch.float32)
    
    # Compute the bin index for each value
    bin_index = torch.sum(values.unsqueeze(1) > sorted_bins, dim=1)
    
    # Create one-hot tensor for the batch
    one_hot_vectors = torch.zeros(values.size(0), len(bins) + 1)  # +1 for values outside the range of bins
    
    # Set the corresponding index to 1
    one_hot_vectors[torch.arange(values.size(0)), bin_index] = 1
    
    return one_hot_vectors

path = "../results/"
data_set = CustomDataset(path)

# Create DataLoader
#data_loader = DataLoader(data_set, batch_size=32, shuffle=True, collate_fn=lambda x: x)
train_size = 64
val_size = 64
test_size = len(data_set) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(data_set, [train_size, val_size, test_size])

for i in range(train_size):
    atom_dict, atom_label = train_dataset[i]
    file_name = atom_dict["file_name"]
    print(f"for training: {file_name}")

for i in range(val_size):
    atom_dict, atom_label = val_dataset[i]
    file_name = atom_dict["file_name"]
    print(f"for validation: {file_name}")

for i in range(test_size):
    atom_dict, atom_label = test_dataset[i]
    file_name = atom_dict["file_name"]
    print(f"for testing: {file_name}")

# Create data loaders for train, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: x)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: x)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Create Model
model = AtomData2Img64().to(device)
model.train()
# Define the loss function
criterion = nn.MSELoss()

# Define the Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Learning rate can be adjusted

# Training loop
num_epochs = 512
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    vali_loss = validate_model(model, val_loader, criterion, device)
    print(f'Epoch {epoch + 1}/{num_epochs + 512}, Train Loss: {train_loss}, Vali Loss: {vali_loss}')
     # Save the model
    if((epoch+1)%128==0):
        model_path = f'AtomData2Img64_train64_ep{epoch+1}.pth'
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}')

optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Learning rate can be adjusted
num_epochs = 512
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    vali_loss = validate_model(model, val_loader, criterion, device)
    print(f'Epoch {epoch + 513}/{num_epochs + 512}, Train Loss: {train_loss}, Vali Loss: {vali_loss}')
    # Save the model
    if((epoch+1)%128==0):
        model_path = f'AtomData2Img64_train64_ep{epoch+513}.pth'
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}')

