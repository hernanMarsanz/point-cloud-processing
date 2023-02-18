import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
import open3d as o3d
from PointCloudTransformer import PointCloudTransformer
from tkinter import filedialog as fd
import time
import glob
import copy


class PointCloudVisualizer:
    def __init__(self):
        ...


    @staticmethod
    def display_open3d_with_o3d(pcd):
        o3d.visualization.draw_geometries([pcd])


    @staticmethod
    def display_numpy_with_o3d(np_array):
        pcd = PointCloudTransformer.numpy_to_open3d(np_array)
        o3d.visualization.draw_geometries([pcd])


    @staticmethod
    def display_pandas_with_o3d(df):
        pcd = PointCloudTransformer.pandas_to_open3d(df)
        o3d.visualization.draw_geometries([pcd])


    @staticmethod
    def display_dask_with_o3d(df):
        pcd = PointCloudTransformer.dask_to_open3d(df)
        o3d.visualization.draw_geometries([pcd])


    def rotate_view(self, vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False


    def change_background_to_black(self, vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False


    @staticmethod
    def display_animation(pcd_list, pose_path=None, time_step=.1, save=False):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        geometry = pcd_list[0]
        vis.add_geometry(geometry)
        print(len(pcd_list))
        last_pcd = None
        for i in range(len(pcd_list)):
            geometry.points = pcd_list[i].points
            vis.update_geometry(geometry)
            vis.poll_events()
            vis.update_renderer()
            if save:
                filename = 'animations/3/frame_'+'0'*(4-len(str(i))) + str(i) + '.jpg'
                vis.capture_screen_image(filename, True)
            time.sleep(time_step)
        vis.run()


    @staticmethod
    def display_animation_on_pcd(base_pcd, pcd_list, pose_path=None, time_step=.1, save=False):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        geometry = pcd_list[0]
        vis.add_geometry(base_pcd)
        vis.add_geometry(geometry)
        ctr = vis.get_view_control()
        if pose_path:
            parameters = o3d.io.read_pinhole_camera_parameters(pose_path)
            ctr.convert_from_pinhole_camera_parameters(parameters)
        for i in range(len(pcd_list)):
            geometry.points = pcd_list[i].points
            vis.update_geometry(geometry)
            vis.update_geometry(base_pcd)
            vis.poll_events()
            vis.update_renderer()
            if save:
                filename = 'animations/frame_'+'0'*(4-len(str(i))) + str(i) + '.jpg'
                vis.capture_screen_image(filename, True)
            time.sleep(time_step)
        vis.run()
        vis.destroy_window()


    @staticmethod
    def display_box_animation_on_pcd(base_pcd, box_points_list, pose_path=None, time_step=.1):
        lines = [[0, 1], [0, 2], [0, 4], [1, 3],
                [1, 5], [2, 3], [2, 6], [3, 7],
                [4, 5], [4, 6], [5, 7], [6, 7]]
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(base_pcd)
        ctr = vis.get_view_control()
        if pose_path:
            print('Using pose path')
            parameters = o3d.io.read_pinhole_camera_parameters(pose_path)
            ctr.convert_from_pinhole_camera_parameters(parameters)
        last_box = None
        for i in range(len(box_points_list)):
            parameters = o3d.io.read_pinhole_camera_parameters(pose_path)
            ctr.convert_from_pinhole_camera_parameters(parameters)
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(box_points_list[i])
            line_set.lines = o3d.utility.Vector2iVector(lines)
            if i != 0:
                vis.remove_geometry(last_box)
            vis.add_geometry(line_set)
            vis.update_geometry(base_pcd)
            vis.poll_events()
            vis.update_renderer()
            last_box = line_set
            ctr = vis.get_view_control()
            time.sleep(time_step)
        vis.run()


    @staticmethod
    def convert_box_points_to_cloud_box(box_points, step=0.01):
        lines = [[0, 1], [0, 2], [0, 4], [1, 3],
                [1, 5], [2, 3], [2, 6], [3, 7],
                [4, 5], [4, 6], [5, 7], [6, 7]]
        np_points = np.asarray(box_points)
        np_pc_points = []
        for i,element in enumerate(lines):
            print('')
            print('-'*10)
            print(i)
            print(box_points[element[0]])
            print(box_points[element[1]])
            diff_array = box_points[element[0]]!=box_points[element[1]]
            index  = (np.where(diff_array == True))[0][0]
            print(diff_array)
            print(index)
            val1 = (box_points[element[0]])[index]
            val2 = (box_points[element[1]])[index]
            print(f'Value 1 is {val1} and value 2 is {val2}')
            val_list = np.arange(val1,val2,step).tolist()
            print(val_list[-10:-1])
            for i,value in enumerate(val_list):
                new_point = copy.deepcopy(box_points[element[0]])
                new_point[index] += value
                new_point = np.asarray(new_point)
                np_pc_points.append(new_point)
        for point in box_points:
            np_pc_points.append(point)
        print('-'*10)
        print('')
        np_pc_points = np.asarray(np_pc_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_pc_points)
        o3d.visualization.draw_geometries([pcd])


    @staticmethod
    def display_pcds_from_folder():
        folder_selected = fd.askdirectory()
        print(folder_selected)


    def get_object_extent(self, pcd):
        """Takes a point cloud and returns the extent of a box surrounding it."""
        pcd_copy = copy.deepcopy(pcd)
        box = pcd_copy.get_oriented_bounding_box()
        extent = box.extent
        return extent


    # Examples:


    def example_001_display_pcd(self):
        filename = fd.askopenfilename()
        pcd = o3d.io.read_point_cloud(filename)
        o3d.visualization.draw_geometries([pcd])
    

    def example_002_display_pcds_from_folder(self):
        folder_selected = fd.askdirectory()
        print(folder_selected)
        filename_list = []
        for filename in glob.glob(folder_selected+'/*ply'):
            filename_list.append(filename)
        filename_list = sorted(filename_list)
        for i,filename in enumerate(filename_list):
            print('\n',filename)
            print(f'\n{i}\n')
            pcd = o3d.io.read_point_cloud(filename)
            extent = self.get_object_extent(pcd)
            print(f'Number of points: {len(pcd.points)}')
            print(f'Estimated dimensions: {extent} meters.','\n')
            o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    visualizer = PointCloudVisualizer()
    # visualizer.example_001_display_pcd()
    visualizer.example_002_display_pcds_from_folder()

