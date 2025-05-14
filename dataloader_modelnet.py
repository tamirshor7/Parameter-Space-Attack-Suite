'''
Adapted from https://www.kaggle.com/code/balraj98/pointnet-for-3d-object-classification-pytorch
'''
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import joblib
from PIL import Image
import json
import os
import trimesh
from scipy.ndimage import zoom
import numpy as np

def mesh_to_point_cloud(mesh, num_points=1024):
    
    
    # Sample points on the mesh surface
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    

    return points

def resample_voxel_grid(voxel_grid, target_shape):
    """
    Resamples a voxel grid to the target shape using interpolation.
    
    Parameters:
    voxel_grid (numpy.ndarray): The voxel grid to be resampled, shape (D, H, W).
    target_shape (tuple): The target shape to resample to (target_D, target_H, target_W).
    
    Returns:
    numpy.ndarray: The resampled voxel grid with shape `target_shape`.
    """
    # Calculate the zoom factors for each dimension
    zoom_factors = [target_shape[i] / voxel_grid.shape[i] for i in range(3)]
    
    # Resample the voxel grid
    resampled_voxel_grid = zoom(voxel_grid, zoom_factors, order=1)
    
    return resampled_voxel_grid
    
def mesh_to_voxels(mesh, voxel_resolution=64, resample_shape = (15,15,15)):
    """
    Converts a 3D mesh into a voxel grid.
    
    Parameters:
    mesh_file (str): Path to the mesh file (.obj, .off, etc.)
    voxel_resolution (int): Number of voxels per side for the grid.
    
    Returns:
    numpy.ndarray: Voxel grid of shape (voxel_resolution, voxel_resolution, voxel_resolution).
    """
    
    
    # Get the bounds of the mesh
    bounds = mesh.bounds
    scale = np.max(bounds[1] - bounds[0])  # Scaling factor to fit the mesh in the voxel grid
    
    # Voxelize the mesh
    voxel_grid = mesh.voxelized(pitch=scale/voxel_resolution)
    
    # Convert the voxel grid into a dense binary representation
    voxel_matrix = resample_voxel_grid(voxel_grid.matrix.astype(np.uint8),resample_shape)
    
    return voxel_matrix

class ModelNet(Dataset):
    def __init__(self, root, train="train", transform=None, resample_shape = (20,20,20)):
        self.root = root
        folders = [dir for dir in sorted(os.listdir(self.root)) if os.path.isdir(f"{self.root}/{dir}")]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transform = transform
        self.files = []
        self.resample_shape = resample_shape
        for category in self.classes.keys():
            new_dir = f"{self.root}/{category}/{'train' if train else 'test'}"
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    sample['pcd_path'] = new_dir + "/" + file
                    sample['category'] = category
                    self.files.append(sample)
    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):
    
        mesh = trimesh.load(file, file_type='off')
        points = mesh_to_voxels(mesh, 64, resample_shape = self.resample_shape)
       
        
        return torch.Tensor(points)
        #if self.transform:
           
        #    points = self.transform(points)
            

        return points

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)
        return pointcloud, self.classes[category]



def get_modelnet_loader(
        root='root',
        train=True,
        batch_size=64,
        num_workers=2,
        pin_memory=True,
        resample_shape=(20,20,20)
):
    """
    :param root:
    :param train:
    :param batch_size:
    :param transform:
    :param num_workers:
    :param pin_memory:
    :return: Dataloader.
    """
    transform =T.Compose([
                    T.ToTensor(),
                    
                    ])


    dataset = ModelNet(root=root, train=train, transform=transform, resample_shape=resample_shape)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)
    return dataloader
    




# Basic Functaset.
class Functaset(Dataset):
    def __init__(self, pkl_file):
        super(Functaset, self).__init__()
        self.functaset = joblib.load(pkl_file)

    def __getitem__(self, item):
        pair = self.functaset[item]
        modul = torch.tensor(pair['modul'])
        label = torch.tensor(pair['label'])
        return modul, label

    def __len__(self):
        return len(self.functaset)


def collate_fn(data):
    """
    :param data: is a list of tuples with (modul, label).
    :return: data batch.
    """
    moduls, labels = zip(*data)
    return torch.stack(moduls), torch.stack(labels)




# Get the functa version of ModelNet10.
def get_modelnet_functa(
        data_dir=None,
        batch_size=256,
        mode='train',
        num_workers=4,
        pin_memory=True,
        outer_test = True
):
    """
    :param data_dir:
    :param batch_size:
    :param mode:
    :param num_workers:
    :param pin_memory:
    :return:
    """
    assert mode in ['train', 'val', 'test']
    if data_dir is None:
        data_dir = f'./functaset/modelnet_{mode}.pkl'
    
    functaset = Functaset(data_dir)
    shuffle = mode == 'train'
    return DataLoader(
        functaset, batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=pin_memory, collate_fn=collate_fn,
    )

