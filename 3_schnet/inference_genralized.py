from ovito_read_data_add_sfedl_build_neigh_list import ovito_read_data_add_3d_mask_build_neigh_list
from schnet_model import AtomData2Img64
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the non-GUI backend
import os
import pandas as pd

# Function to read text file, ignoring headers/comments marked by '#'
def load_data(file_path):
    # Read the file, skipping lines that start with '#'
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, comment='#', names=['x', 'y', 'z']).dropna()
    return data

path = "../results/"
name = ""
name = "np_box4r1x0.887r1y0.115r1z0.446r1d176.equ.min.data"
pmf_data = "box4r1x0.887r1y0.115r1z0.446r1d176.200000000.pmf"
#name = "np_box4r1x0.721r1y-0.674r1z-0.156r1d336.equ.min.data"
#name = "np_box4r1x-0.257r1y-0.212r1z0.942r1d266.equ.min.data"
atom_numbers, r_ij, idx_i, idx_j = ovito_read_data_add_3d_mask_build_neigh_list(path+name, x_start=-5.65, y_start=-5.65)
#atom_numbers, r_ij, idx_i, idx_j = ovito_read_data_add_3d_mask_build_neigh_list(path+name)
atom_dict = {
    "structure_types": torch.tensor(atom_numbers),
    "r_ij": torch.tensor(r_ij, dtype=torch.float32),
    "idx_i": torch.tensor(idx_i),
    "idx_j": torch.tensor(idx_j),
    "file_name": name	
}
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"
# Create Model
model = AtomData2Img64().to(device)
model_path = 'AtomData2Img64_train64_ep1024.pth'
# Load the saved state dictionary
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

pred_info = model(atom_dict).cpu().detach()
contour_x = np.linspace(-31.5, 31.5, 64)
contour_y = np.linspace(-31.5, 31.5, 64)
Z_vals = torch.permute(pred_info.view(64,64),(1,0)).numpy()
contour_z = Z_vals
#contour_z = torch.permute(pred_info.view(64,64),(0,1)).numpy()
#contour_z = pred_info.view(64,64).numpy()
data = load_data(path+pmf_data)
Z_meta = data.pivot_table(index='y', columns='x', values='z').values
levels = np.linspace(0, np.max(Z_meta), 21)
contour = plt.contourf(contour_x*0.1, contour_y*0.1, contour_z, cmap='viridis', levels=levels)
plt.colorbar(contour, label='PMF (eV)')  # Add a color bar    
# Add labels and title
plt.xlabel('X distance (Å)')
plt.ylabel('Y distance (Å)')
save_path = os.path.join("./", f"{name}.svg")
plt.savefig(save_path, format='svg')
plt.clf()
