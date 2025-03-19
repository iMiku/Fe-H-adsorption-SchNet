from ovito.io import import_file, export_file
from ovito.data import *
from ovito.modifiers import *
import numpy as np

def ovito_read_data_add_3d_mask_build_neigh_list(fileName, neigh_num=16, x_start=-3.15, y_start=-3.15):
    pipeline = import_file(fileName)
    pipeline.modifiers.append(ExpressionSelectionModifier(expression = 'ParticleType==2'))
    pipeline.modifiers.append(DeleteSelectedModifier())
    
    data = pipeline.compute()
    finder = NearestNeighborFinder(neigh_num, data)

    # neigh_idx: (M, N) indices of neighbor particles, 
    # M equal to the total number of particles 
    # N refers to the number of nearest neighbors 
    # neigh_vec: (M, N, 3) for x-, y- and z- components of delta
    neigh_idx, neigh_vec = finder.find_all()

    # M is the total number of particles, N is the number of nearest neighbors
    M, N = neigh_idx.shape   
    # idx_i: (M*N) array of all i- (centered) atoms
    # Repeat each particle index N times (since each particle has N neighbors)
    idx_i = np.repeat(np.arange(M), N)   
    # idx_j: (M*N) array of all j- (neighbor) atoms
    # Flatten the neighbor indices
    idx_j = neigh_idx.flatten()
    # r_ij: (M*N, 3) array of x-, y- and z- component of delta of each pair
    # Reshape the neighbor vectors into a (M*N, 3) array
    r_ij = neigh_vec.reshape(-1, 3)
    # Now, idx_i contains the indices of the center particles,
    # idx_j contains the indices of the neighbors,
    # r_ij contains the delta vectors for each particle-neighbor pair.
    structure_types = np.array(data.particles['Particle Type'])

    #start    = -3.15
    interval =  0.1
    steps    =  64
    z_start  =  3.625
    z_interval = 0.25
    z_steps  = 16
    comX = np.mean(data.particles.positions[:, 0])
    comY = np.mean(data.particles.positions[:, 1])
    comZ = np.mean(data.particles.positions[:, 2])
    id_start = data.particles.count
    # Initialize lists for appending results
    idx_i_list = idx_i.tolist()
    idx_j_list = idx_j.tolist()
    r_ij_list = r_ij.tolist()
    structure_types_list = structure_types.tolist()

    # Find neighbors for grid points
    for i in range(steps):
        for j in range(steps):
            for k in range(z_steps):
                coordX = comX + x_start + i*interval
                coordY = comY + y_start + j*interval
                coordZ = comZ + z_start + k*z_interval
                coord = [coordX, coordY, coordZ]
            
                for neigh in finder.find_at(coord):
                    #print(neigh.index, neigh.delta)
                    # Append new neighbors' data
                    idx_i_list.append(i*steps*z_steps + j*z_steps + id_start + k)
                    idx_j_list.append(neigh.index)
                    r_ij_list.append(neigh.delta)
                structure_types_list.append(0)  # append '0' to structure_types each loop

    # Convert lists back to arrays
    idx_i_final = np.array(idx_i_list)
    idx_j_final = np.array(idx_j_list)
    r_ij_final = np.array(r_ij_list)
    structure_types_final = np.array(structure_types_list)

    return structure_types_final, r_ij_final, idx_i_final, idx_j_final

if __name__ == "__main__":
    path = "../results/np_box4_cv_xy/"
    name = "np_box4r1x-0.010r1y0.876r1z0.482r1d115.equ.min.data"
    atom_numbers, r_ij, idx_i, idx_j = ovito_read_data_add_sfedl_build_neigh_list(path+name)
    print(len(atom_numbers))
    print(len(r_ij))
    print(len(idx_i))
    print(len(idx_j))
    print(np.unique(idx_i))
    print(np.max(r_ij))
