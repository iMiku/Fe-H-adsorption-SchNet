from ovito_read_data_add_sfedl_build_neigh_list import ovito_read_data_add_3d_mask_build_neigh_list
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json

def list_label_files(directory, suffix):
    # List all files in the specified directory
    all_files = os.listdir(directory)
    # Filter files that end with ".label"
    label_files = [file for file in all_files if file.endswith(suffix)]
    return label_files

def read_pmf_vals(file_name):
    data = pd.read_csv(file_name, sep='\s+', header=None, comment='#', names=['x', 'y', 'z']).dropna()
    z_vals = np.array(data['z'])
    z_vals = z_vals - np.min(z_vals)
    return z_vals

def read_pmf_vals_64x64(file_name):
    data = pd.read_csv(file_name, sep='\s+', header=None, comment='#', names=['x', 'y', 'z']).dropna()
    z_vals = torch.tensor(data['z'])
    z_vals = z_vals.view(114,114)
    z_vals = z_vals[25:89, 25:89]
    z_vals = z_vals.reshape(64*64,1).squeeze(-1)
    z_vals = z_vals - torch.min(z_vals)
    return z_vals

def save_all_atom_dict(file_name, data_dict_list):
    # Convert each dictionary in the list to a NumPy array
    np_data_dict = {}
    for i, data_dict in enumerate(data_dict_list):
        for key, value in data_dict.items():
            try:
                np_data_dict[f'{key}_{i}'] = value.numpy()
            except AttributeError:
                np_data_dict[f'{key}_{i}'] = value
    
    # Save as NPZ
    np.savez(file_name, **np_data_dict)

def load_all_atom_dict(file_name):
    loaded_data = np.load(file_name)
    
    # Reconstruct the list of dictionaries
    data_dict_list = []
    i = 0
    while True:
        data_dict = {}
        for key in ['structure_types', 'r_ij', 'idx_i', 'idx_j', 'file_name']:
            try:
                data_dict[key] = torch.from_numpy(loaded_data[f'{key}_{i}'])
            except KeyError:
                # No more dictionaries to load
                return data_dict_list
            except TypeError:
                data_dict[key] = loaded_data[f'{key}_{i}']
        data_dict_list.append(data_dict)
        i += 1

class CustomDataset(Dataset):
    def __init__(self, path):
        """
        Args:
            path: path to folder containing lammps data file
        """
        self.all_file_names = os.listdir(path)
        self.file_names = list_label_files(path, ".equ.min.data")
        self.data_dicts = []
        self.labels = []
        self.has_all_data_dict = False
        self.need_to_rebuild = False
        
        all_atom_dict_file = os.path.join(path, "all_atom_dict_plain.npz")
        if os.path.exists(all_atom_dict_file):
            self.has_all_data_dict = True
            self.data_dicts = load_all_atom_dict(all_atom_dict_file)
        
        if not self.has_all_data_dict:
            for idx, name in enumerate(self.file_names):
                structure_types, r_ij, idx_i, idx_j = ovito_read_data_add_3d_mask_build_neigh_list(os.path.join(path, name))
                data_dict = {
                    "structure_types": torch.tensor(structure_types),
                    "r_ij": torch.tensor(r_ij, dtype=torch.float32),
                    "idx_i": torch.tensor(idx_i),
                    "idx_j": torch.tensor(idx_j),
                    "file_name": name
                }
                self.data_dicts.append(data_dict)
                
                pmf_file = name[3:-13] + ".200000000.pmf"
                pmf_vals = read_pmf_vals_64x64(os.path.join(path, pmf_file))
                self.labels.append(pmf_vals)
                
                if (idx + 1) % 10 == 0 or (idx + 1) == len(self.file_names):
                    print(f"loading {(idx+1)}/{len(self.file_names)}")
            
            print("building all_atom_dict_plain dictionary")
            save_all_atom_dict(all_atom_dict_file, self.data_dicts)
        else:
            for idx, atom_data in enumerate(self.data_dicts):
                name = str(atom_data['file_name'])
                #print(name)
                pmf_file = name[3:-13] + ".200000000.pmf"
                pmf_vals = read_pmf_vals_64x64(os.path.join(path, pmf_file))
                self.labels.append(pmf_vals)
                
                if (idx + 1) % 10 == 0 or (idx + 1) == len(self.file_names):
                    print(f"loading {(idx+1)}/{len(self.file_names)}")

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.file_names)

    def __getitem__(self, idx):
        """Return dict at idx"""
        return (self.data_dicts[idx], self.labels[idx])

if __name__ == "__main__":
    dl = CustomDataset("../results/np_box4_cv_xy/")
