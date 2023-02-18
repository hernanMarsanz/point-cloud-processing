import trimesh
import open3d as o3d
import copy
import numpy as np
import time
import keyboard
from pynput import keyboard

mesh = o3d.io.read_triangle_mesh("mesh_test.ply")
mesh2 = trimesh.load("mesh_test.ply")
points = trimesh.load(mesh2).sample(2048)
points2 = copy.deepcopy(points)
# print(type(points))
point_list = []
point_list.append(points)
point_list.append(points2)
array_list = np.array(point_list)
# print(f'Array list type is {type(array_list)}')
# print(f'Array element type is {type(array_list[0])}')
# print(array_list.shape)


vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(mesh)
vis.poll_events()
vis.update_renderer()
time.sleep(3)
vis.destroy_window()





# mesh2.show()
# o3d.visualization.draw_geometries([mesh])

# pcd = o3d.io.read_point_cloud('samples/other/munition_test.ply')
# radii = [0.005, 0.01, 0.02, 0.04]
# rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     pcd, o3d.utility.DoubleVector(radii))
# o3d.visualization.draw_geometries([rec_mesh])


# import threading

# timer = threading.Timer(60.0, callback)
# timer.start()  # after 60 seconds, 'callback' will be called

# ## (in the meanwhile you can do other stuff...)
