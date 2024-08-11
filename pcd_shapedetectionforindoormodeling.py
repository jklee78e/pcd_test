
#%% 1. Library setup

     # https://learngeodata.eu/3d-tutorials/
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

DATANAME = "ITC_groundfloor.ply"
pcd = o3d.io.read_point_cloud("./DATA/" + DATANAME)

# If you intend on visualizing the point cloud within open3dIt is good practice to shift your point cloud to bypass the large coordinates approximation, which creates shaky visualization effects. To apply such a shift to your pcd point cloud, first get the center of the point cloud, then translate it by subtracting it from the original variable:

pcd_center = pcd.get_center()
pcd.translate(-pcd_center)

# o3d.visualization.draw_geometries([pcd]) calls the draw_geometries() function from the visualization Module in Open3D. The function takes a list of geometries as an argument and displays them in a visualization window. In this case, the list contains a single geometry, which is the pcd variable representing the point cloud. The draw_geometries() function creates a 3D visualization window and renders the point cloud. You can interact with the visualization window to rotate, zoom, and explore the point cloud from different perspectives.

o3d.visualization.draw_geometries([pcd])

#%% 3.1. Point Cloud Random Sampling
#%% Let us consider random sampling methods that can effectively reduce point cloud size while preserving overall structural integrity and representativeness. If we define a point cloud as a matrix (m x n), then a decimated cloud is obtained by keeping one row out of n of this matrix :

retained_ratio = 0.2
sampled_pcd = pcd.random_down_sample(retained_ratio)

o3d.visualization.draw_geometries([sampled_pcd], window_name = "Random Sampling")

#  3D 포인트 클라우드를 연구할 때, 랜덤 샘플링은 중요한 정보가 누락되고 분석이 부정확해질 수 있는 한계가 있습니다. 이는 공간적 구성 요소나 포인트 간의 관계를 고려하지 않습니다. 따라서 보다 포괄적인 분석을 보장하기 위해 다른 방법을 사용하는 것이 필수적입니다.
#이 전략은 빠르지만, 무작위 샘플링은 "표준화" 사용 사례에 가장 적합하지 않을 수 있습니다. 다음 단계는 통계적 이상치 제거 기술을 통해 잠재적 이상치를 처리하여 후속 분석 및 처리를 위한 데이터 품질과 신뢰성을 보장하는 것입니다.

#%% 3.2. Statistical outlier removal
# Using an outlier filter on 3D point cloud data can help identify and remove any data points significantly different from the rest of the dataset. These outliers could result from measurement errors or other factors that can skew the analysis. By removing these outliers, we can get a more valid representation of the data and better adjust algorithms. However, we need to be careful not to delete valuable points.
# We will define a statistical_outlier_removal filter to remove points that are further away from their neighbors compared to the average for the point cloud. It takes two input parameters:
# nb_neighbors, which specifies how many neighbors are considered to calculate the average distance for a given point.
# std_ratio, which allows setting the threshold level based on the standard deviation of the average distances across the point cloud. The lower this number, the more aggressive the filter will be.
# This amount to the following:

nn = 16
std_multiplier = 10

filtered_pcd, filtered_idx = pcd.remove_statistical_outlier(nn, std_multiplier)

outliers = pcd.select_by_index(filtered_idx, invert=True)
outliers.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([filtered_pcd, outliers])

#%% 3.3. Point Cloud Voxel (Grid) Sampling
# The grid subsampling strategy is based on the division of the 3D space in regular cubic cells called voxels. For each cell of this grid, we only keep one representative point, and this point, the representative of the cell, can be chosen in different ways. When subsampling, we keep that cell's closest point to the barycenter.

voxel_size = 0.05
pcd_downsampled = filtered_pcd.voxel_down_sample(voxel_size = voxel_size)

o3d.visualization.draw_geometries([pcd_downsampled])


#%% 3.4. Point Cloud Normals Extraction
# A point cloud normal refers to the direction of a surface at a specific point in a 3D point cloud. It can be used for segmentation by dividing the point cloud into regions with similar normals, for example. In our case, normals will help identify objects and surfaces within the point cloud, making it easier to visualize. And it is an excellent opportunity to introduce a way to compute such normals semi-automatically. We first define the average distance between each point in the point cloud and its neighbors:

nn_distance = 0.05


radius_normals=nn_distance*4
pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normals, max_nn=16), fast_normal_computation=True)

pcd_downsampled.paint_uniform_color([0.6, 0.6, 0.6])
o3d.visualization.draw_geometries([pcd_downsampled,outliers])


#%% 4. Point Cloud Parameter setting
# In this tutorial, we have selected two of the most effective and reliable methods for 3D Shape detection and clustering for you to master: RANSAC and Euclidean Clustering using DBSCAN. However, before utilizing these approaches, hence understanding the parameters, it is crucial to comprehend the fundamental concepts in simple terms.
# The RANSAC algorithm, short for RANdom SAmple Consensus, is a powerful tool for handling data that contains outliers, which is often the case when working with real-world sensors. The algorithm works by grouping data points into two categories: inliers and outliers. By identifying and ignoring the outliers, you can focus on working with reliable inliers, making your analysis more effective.

nn_distance = np.mean(pcd.compute_nearest_neighbor_distance())


#%% 5. Point Cloud Segmentation with RANSAC

distance_threshold = 0.1
ransac_n = 3
num_iterations = 1000
plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,ransac_n=3,num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd.select_by_index(inliers) 
outlier_cloud = pcd.select_by_index(inliers, invert= True ) 

# 구름 그리기
inlier_cloud.paint_uniform_color([ 1.0 , 0 , 0 ]) 
outlier_cloud.paint_uniform_color([ 0.6 , 0.6 , 0.6 ]) 

# 인라이어와 아웃라이어 시각화
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

#%% 6. Scaling 3D Segmentation: Multi-Order RANSAC


segment_models={}
segments={}

max_plane_idx=10
rest=pcd
for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(distance_threshold=0.1,ransac_n=3,num_iterations=1000)
    segments[i]=rest.select_by_index(inliers)
    segments[i].paint_uniform_color(list(colors[:3]))
    rest = rest.select_by_index(inliers, invert=True)
    print("pass",i,"/",max_plane_idx,"done.")

o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest])
    
# DBSCAN for 3D Point Cloud Clustering
epsilon = 0.15
min_cluster_points = 5

for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(distance_threshold=0.1,ransac_n=3,num_iterations=1000)

    segments[i]=rest.select_by_index(inliers)
    labels = np.array(segments[i].cluster_dbscan(eps=epsilon, min_points=min_cluster_points))
    candidates=[len(np.where(labels==j)[0]) for j in np.unique(labels)]
    best_candidate=int(np.unique(labels)[np.where(candidates== np.max(candidates))[0]])

    rest = rest.select_by_index(inliers, invert=True) + segments[i].select_by_index(list(np.where(labels!=best_candidate)[0]))
    segments[i]=segments[i].select_by_index(list(np.where(labels== best_candidate)[0]))

    segments[i].paint_uniform_color(list(colors[:3]))
    rest = rest.select_by_index(inliers, invert=True)
    print("pass",i,"/",max_plane_idx,"done.")

o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest])

max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
rest.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([rest])

voxel_size=0.5
min_bound = pcd.get_min_bound()
max_bound = pcd.get_max_bound()
pcd_ransac=o3d.geometry.PointCloud()
for i in segments:
 pcd_ransac += segments[i]

pcd_ransac.paint_uniform_color([0.8, 0.1, 0.1])
rest.paint_uniform_color([0.1, 0.1, 0.8])

voxel_grid_clutter = o3d.geometry.VoxelGrid.create_from_point_cloud(rest, voxel_size=voxel_size)
voxel_grid_plane = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_ransac, voxel_size=voxel_size)

o3d.visualization.draw_geometries([voxel_grid_clutter,voxel_grid_plane])

idx_voxels=[v.grid_index for v in voxel_grid_clutter.get_voxels()]

# color_voxels=[v.color for v in voxel_grid.get_voxels()]
bounds_voxels=[np.min(idx_voxels, axis=0),np.max(idx_voxels, axis=0)]
print("debug")

# %%
