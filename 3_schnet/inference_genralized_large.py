from ovito_read_data_add_sfedl_build_neigh_list import ovito_read_data_add_3d_mask_build_neigh_list
from schnet_model import AtomData2Img64
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the non-GUI backend
import os, time

path = "./"
name = "iron_surface_center64.data"
#name = "np_box4r1x0.721r1y-0.674r1z-0.156r1d336.equ.min.data"
#name = "np_box4r1x-0.257r1y-0.212r1z0.942r1d266.equ.min.data"

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"
# Create Model
model = AtomData2Img64().to(device)
model_path = 'AtomData2Img64_train64_ep1024.pth'
# Load the saved state dictionary
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

pred_info = torch.zeros(512,512)

for i in range(8):
    for j in range(8):
        x_start = j*6.4 + (-25.6) + 0.05
        y_start = i*6.4 + (-25.6) + 0.05
        atom_numbers, r_ij, idx_i, idx_j = ovito_read_data_add_3d_mask_build_neigh_list(path+name, x_start=x_start, y_start=y_start)
        atom_dict = {
            "structure_types": torch.tensor(atom_numbers).to(device),
            "r_ij": torch.tensor(r_ij, dtype=torch.float32).to(device),
            "idx_i": torch.tensor(idx_i).to(device),
            "idx_j": torch.tensor(idx_j).to(device),
            "file_name": name   
        }
        start_time = time.time()
        local_pred = model(atom_dict).cpu().detach()
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Time taken for execution: {time_taken:.6f} seconds")
        pred_info[i*64:(i+1)*64, j*64:(j+1)*64] = torch.permute(local_pred.view(64,64),(1,0))
        print(i*8+j)


#pred_info = model(atom_dict).cpu().detach()
contour_x = np.linspace(-255.5, 255.5, 512)
contour_y = np.linspace(-255.5, 255.5, 512)
contour_z = pred_info.numpy()
levels = np.linspace(np.min(contour_z), np.max(contour_z), 21)
contour = plt.contourf(contour_x*0.1, contour_y*0.1, contour_z, cmap='viridis', levels=levels)
plt.colorbar(contour, label='PMF (eV)')  # Add a color bar    
# Add labels and title
plt.xlabel('X distance (Å)')
plt.ylabel('Y distance (Å)')
save_path = os.path.join("./", f"{name}.svg")
plt.savefig(save_path, format='svg')
plt.clf()
