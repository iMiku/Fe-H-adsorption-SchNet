from schnet_model import AtomData2Img64
from schnet_dataloader import CustomDataset
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the non-GUI backend
import os

path = "../results/np_box4_cv_xy/"
data_set = CustomDataset(path)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create Model
model = AtomData2Img64().to(device)
model_path = 'AtomData2Img64_train64_ep1024.pth'
# Load the saved state dictionary
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

y_gt = []
y_pred = []
criterion = nn.MSELoss()

contour_x = np.linspace(-31.5, 31.5, 64)
contour_y = np.linspace(-31.5, 31.5, 64)

#plt.figure()
excluded_keys = ["file_name"]
path = "./inferred_img64_train64_ep512/"
for datainfo in data_set:
    atom_info = datainfo[0]
    atom_info = {key: val.to(device) if key not in excluded_keys else val for key, val in atom_info.items()}
    file_name = atom_info['file_name']
    pred_info = model(atom_info).cpu().detach()
    y_gt.append(datainfo[1].numpy())
    y_pred.append(pred_info.numpy())
    loss = criterion(torch.tensor(y_gt[-1]), torch.tensor(y_pred[-1]))
    print(file_name, "loss: ", loss.item())
    #contour_z = torch.permute(pred_info.view(64,64),(1,0)).numpy()

    # plot groundtruth
    Z = y_gt[-1]
    levels = np.linspace(0, np.max(Z), 21)
    contour_z = torch.permute(torch.tensor(Z).view(64,64),(1,0)).numpy()
    contour = plt.contourf(contour_x*0.1, contour_y*0.1, contour_z, cmap='viridis', levels=levels)
    plt.colorbar(contour, label='PMF Difference (eV)')
    plt.title(f'MSE: {loss.item()}')
    plt.xlabel('X distance (Å)')
    plt.ylabel('Y distance (Å)')
    save_path = os.path.join(path, f"{file_name}_gt.svg")
    plt.savefig(save_path, format='svg')
    plt.clf()
    
    # plot predictions
    Z = y_pred[-1]
    levels = np.linspace(0, np.max(Z), 21)
    contour_z = torch.permute(torch.tensor(Z).view(64,64),(1,0)).numpy()
    contour = plt.contourf(contour_x*0.1, contour_y*0.1, contour_z, cmap='viridis', levels=levels)
    plt.colorbar(contour, label='PMF Difference (eV)')
    plt.title(f'MSE: {loss.item()}')
    plt.xlabel('X distance (Å)')
    plt.ylabel('Y distance (Å)')
    save_path = os.path.join(path, f"{file_name}_pred.svg")
    plt.savefig(save_path, format='svg')
    plt.clf()

    # plot difference
    Z = np.abs(y_pred[-1] - y_gt[-1])
    levels = np.linspace(0, np.max(Z), 21)
    contour_z = torch.permute(torch.tensor(Z).view(64,64),(1,0)).numpy()
    contour = plt.contourf(contour_x*0.1, contour_y*0.1, contour_z, cmap='viridis', levels=levels)
    plt.colorbar(contour, label='PMF Difference (eV)')
    plt.title(f'MSE: {loss.item()}')
    plt.xlabel('X distance (Å)')
    plt.ylabel('Y distance (Å)')
    save_path = os.path.join(path, f"{file_name}_diff.svg")
    plt.savefig(save_path, format='svg')
    plt.clf()
    #break
