# -*- coding: utf-8 -*-
"""
3D Point Cloud Shape Detection for Indoor Modelling

Created by Florent Poux, (c) 2023 Licence MIT
To reuse in your project, please cite the most appropriate article accessible on my Google Scholar page

Have fun with this script!
"""

#%% 1. Library setup
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

#%% 2. Point Cloud Import

DATANAME = 
pcd = 

#%% 3. Data Pre-Processing
pcd_center = 



# 3.1. Sampling
#%% 3.1. Random Sampling Test
retained_ratio = 
o3d.visualization.draw_geometries()

#%% 3.2. Statistical outlier filter 
nn = 
std_multiplier = 

filtered_pcd = 

outliers = 
o3d.visualization.draw_geometries()


filtered_pcd = 

#%% 3.3. Voxel downsampling
voxel_size = 

pcd_downsampled = 
o3d.visualization.draw_geometries()

#%% 3.4. Estimating normals
nn_distance = 
print(nn_distance)

radius_normals = 

pcd_downsampled.estimate_normals()

pcd_downsampled.paint_uniform_color()
o3d.visualization.draw_geometries()

#%% 4. Extracting and Setting Parameters

front = 
lookat = 
up = 
zoom = 

pcd = 
o3d.visualization.draw_geometries()

#%% 5. RANSAC Planar Segmentation

pt_to_plane_dist = 

plane_model, inliers = 
[a, b, c, d] = 
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = 
outlier_cloud = 
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])

o3d.visualization.draw_geometries()

#%% 6. Multi-order RANSAC
max_plane_idx = 
pt_to_plane_dist = 

segment_models = {}
segments = {}
rest = pcd

for i in range(max_plane_idx):

    print("pass",i,"/",max_plane_idx,"done.")

o3d.visualization.draw_geometries()

#%% 7. Euclidean Considerations with DBSCAN
epsilon = 
min_cluster_points = 

labels = 
max_label = 
print(f"point cloud has {max_label + 1} clusters")

colors = 
pcd.colors = 

o3d.visualization.draw_geometries()

#%% 8. Using Euclidean Clustering within the RANSAC Definition
max_plane_idx = 
pt_to_plane_dist = 

segment_models = {}
segments = {}

rest = 

for i in range(max_plane_idx):

    print("pass",i,"/",max_plane_idx,"done.")

o3d.visualization.draw_geometries()

#%% Refined RANSAC with Euclidean clustering AND ORDERING

segment_models = 
segments = 
rest = 

for i in range(max_plane_idx):
    print("pass",i+1,"/",max_plane_idx,"done.")


labels = 
max_label = 
print(f"point cloud has {max_label + 1} clusters")

colors = 
rest.colors = 

o3d.visualization.draw_geometries()



#%%2 9. DBSCAN sur rest
o3d.visualization.draw_geometries()

epsilon = 
min_cluster_points = 

rest_db = 

labels = 
max_label = 
print(f"point cloud has {max_label + 1} clusters")

colors = 
rest_db.colors = 

o3d.visualization.draw_geometries()

o3d.visualization.draw_geometries()

#%% Voxelization with Open3D
voxel_size =

min_bound = pcd.get_min_bound()
max_bound = pcd.get_max_bound()

pcd_ransac = 
for i in segments:

pcd_ransac.paint_uniform_color()
rest.paint_uniform_color()

voxel_grid_clutter = 
voxel_grid_plane = 

o3d.visualization.draw_geometries()

idx_voxels = 

bounds_voxels = 

#%% Creating a voxel grid

def fit_voxel_grid(point_cloud, voxel_size, min_b=False, max_b=False):

    return voxel_grid, indices

voxel_size = 

ransac_voxels, idx_ransac = 

rest_voxels, idx_rest = 

filled_ransac = 
filled_rest = 

total = 
total_voxels, idx_total = 

empty_indices = 

#%% 10. Point Cloud Export
xyz_segments=[]
for idx in segments:
    

rest_w_segments = 


np.savetxt()

#%% Creating a 3D mesh voxel model

def cube(c,s,compteur=0):
    
    return cube

def generate_obj_file(filename, indices, voxel_size):

    return voxel_assembly