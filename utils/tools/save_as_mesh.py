import numpy as np
import pyvista as pv
import open3d as o3d
from PointCloudTransformer import PointCloudTransformer
import trimesh





pcd = o3d.io.read_point_cloud("samples/ply/sample_001.ply")
o3d.visualization.draw_geometries([pcd])
sample_pcd = pcd.farthest_point_down_sample(2048)
o3d.visualization.draw_geometries([sample_pcd])
print(len(pcd.points))
print(len(sample_pcd.points))
# pcd.estimate_normals()

# # to obtain a consistent normal orientation
# pcd.orient_normals_towards_camera_location(pcd.get_center())

# # or you might want to flip the normals to make them point outward, not mandatory
# pcd.normals = o3d.utility.Vector3dVector( - np.asarray(pcd.normals))

# surface reconstruction using Poisson reconstruction
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# # paint uniform color to better visualize, not mandatory
# mesh.paint_uniform_color(np.array([0.7, 0.7, 0.7]))

# sample_points = mesh.sample_points_uniformly(2048)
o3d.io.write_triangle_mesh("mesh_test.ply", mesh)
o3d.visualization.draw_geometries([mesh])
# o3d.visualization.draw_geometries([sample_points])



