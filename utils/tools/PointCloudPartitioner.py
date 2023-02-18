from PointCloudTransformer import PointCloudTransformer
from PointCloudWriter import PointCloudWriter
from PointCloudVisualizer import PointCloudVisualizer
from sklearn import preprocessing
import pandas as pd
import open3d as o3d
import numpy as np
from tkinter import filedialog as fd
import random
import itertools
import json
import time
import sys
import time
import copy
import os
import random
import glob
import math


class PointCloudPartitioner:
    def __init__(self):
        ...


    # General Point Cloud functions:


    def read_ply_as_o3d(self, filepath):
        """Takes a string filepath and returns a point cloud file."""
        pcd = o3d.io.read_point_cloud(filepath)
        return pcd
    
    
    def get_object_extent(self, pcd):
        """Takes a point cloud and returns the extent of a box surrounding it."""
        pcd_copy = copy.deepcopy(pcd)
        box = pcd_copy.get_oriented_bounding_box()
        extent = box.extent
        return extent
    

    def get_max_z(self, pcd):
        """Takes a point cloud and returns the max z value."""
        pcd_array = PointCloudTransformer.open3d_to_numpy(pcd)
        pcd_array_t = copy.deepcopy(pcd_array)
        pcd_array_t = np.transpose(pcd_array_t)
        max_z = np.max(pcd_array_t[2])
        return max_z


    def get_min_z(self, pcd):
        """Takes a point cloud and returns the min z value."""
        pcd_array = PointCloudTransformer.open3d_to_numpy(pcd)
        pcd_array_t = copy.deepcopy(pcd_array)
        pcd_array_t = np.transpose(pcd_array_t)
        min_z = np.min(pcd_array_t[2])
        return min_z


    def print_number_of_points(self, pcd):
        """Takes a point cloud, prints number of points and returns nothing."""
        num_points = len(pcd.points)
        print(f'\nNumber of points is {num_points}\n')


    def concatenate_point_cloud_list(self, pcd_list):
        init_pcd = pcd_list[0]
        cumul_array = np.asarray(init_pcd.points)
        for i,pcd in enumerate(pcd_list):
            if i == 0:
                pass
            else:
                temp_array = np.asarray(pcd.points)
                cumul_array = np.concatenate((temp_array,cumul_array),axis=0)
        out_pcd = o3d.geometry.PointCloud()
        out_pcd.points = o3d.utility.Vector3dVector(cumul_array)
        return out_pcd
    

    # Resample point cloud functions:


    def resample_to_2048_points(self, pcd):
        pcd_point_no = len(pcd.points)
        if pcd_point_no > 2048:
            resampled_pcd = pcd.farthest_point_down_sample(2048)
            if len(resampled_pcd.points) != 2048:
                resampled_pcd = self.random_downsample_point_cloud(pcd,2048)
        elif pcd_point_no < 2048:
            resampled_pcd = self.upsample_point_cloud_to_2048(pcd)
        else:
            resampled_pcd = pcd
        return resampled_pcd


    def downsample_point_cloud(self, pcd, number_of_points):
        sample_pcd = pcd.farthest_point_down_sample(number_of_points)
        return sample_pcd


    def random_downsample_point_cloud(self, pcd, number_of_points):
        copy_pcd = copy.deepcopy(pcd)
        array_pcd = np.asarray(copy_pcd.points)
        point_list = array_pcd.tolist()
        new_point_list = copy.deepcopy(point_list)
        new_pcd_point_no = len(new_point_list)
        point_diff = new_pcd_point_no - 2048
        for element in range(point_diff):
            rand_index = random.randint(0,len(point_list)-1)
            point_list.pop(rand_index)
        point_list = np.asarray(point_list)
        out_pcd = o3d.geometry.PointCloud()
        out_pcd.points = o3d.utility.Vector3dVector(point_list)
        return out_pcd


    def upsample_point_cloud_to_2048(self, pcd, step_range=1, step_divider=20):
        pcd_point_no = len(pcd.points)
        if 1024 <= pcd_point_no < 2028:
            copy_pcd = copy.deepcopy(pcd)
            array_pcd = np.asarray(copy_pcd.points)
            point_list = array_pcd.tolist()
            new_point_list = copy.deepcopy(point_list)
            for point in point_list:
                x_delta = (random.randint(-step_range,step_range))/step_divider
                y_delta = (random.randint(-step_range,step_range))/step_divider
                z_delta = (random.randint(-step_range,step_range))/step_divider
                new_point = np.asarray([point[0]+x_delta,point[1]+y_delta,point[2]]) # no change in z
                new_point_list.append(new_point)
            point_list.extend(new_point_list)
            new_pcd_point_no = len(point_list)
            if new_pcd_point_no > 2048:
                point_diff = new_pcd_point_no - 2048
                for element in range(point_diff):
                    rand_index = random.randint(0,len(point_list)-1)
                    point_list.pop(rand_index)
            point_list = np.asarray(point_list)
            out_pcd = o3d.geometry.PointCloud()
            out_pcd.points = o3d.utility.Vector3dVector(point_list)
            return out_pcd
        else:
            print(pcd_point_no)
            print('Number of points not allowed')
            return None


    def normalize_point_cloud(self, pcd):
        sample_data = np.asarray(pcd.points)

        sample_data_t = np.transpose(sample_data)
        array_len = sample_data_t.shape[1]
        np_linear_array = np.append(sample_data_t[0], sample_data_t[1])
        np_linear_array = np.append(np_linear_array, sample_data_t[2])

        df = pd.DataFrame(np_linear_array)
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        d = scaler.fit_transform(df)
        np_arr_1 = d[0:array_len]
        np_arr_2 = d[array_len:2*array_len]
        np_arr_3 = d[2*array_len:3*array_len]

        array_norm = np.concatenate((np_arr_1,np_arr_2,np_arr_3), axis=1)
        pcd_norm = o3d.geometry.PointCloud()
        pcd_norm.points = o3d.utility.Vector3dVector(array_norm)
        return pcd_norm

    
    def save_training_samples(self):
        selected_filepath = fd.askopenfilenames()
        original_pcd = self.read_ply_as_o3d(selected_filepath)
        pcd = copy.deepcopy(original_pcd)

        # Move to origin
        pcd, box, points, extent, translation_matrix = self.move_to_origin(pcd)



        # Normalize crop
        # self.normalize_point_cloud(sample_crop)



    # Translation functions:


    def align_with_pca(self, pcd):
        """Takes a point cloud object and returns a pca oriented box, the box extent, rot and center."""
        box, extent, r, center = self.get_oriented_box(pcd, (1,0,0))
        pcd = pcd.rotate(r, center)
        box = pcd.get_axis_aligned_bounding_box()
        box.color = (0,1,0)
        extent = box.get_extent
        rot = [[1,0,0], [0,1,0],[0,0,1]]
        center = box.get_center
        return pcd, box, extent, rot, center


    def move_to_origin(self, pcd, color=(1,0,0)):
        """
        Returns a point cloud aligned to origin on x, y and z,
        as well as a box that surrounds it, the box's extent,
        box points and translation matrix.

        Parameters:
                pcd (point cloud): Point cloud data
                color (tuple): Color to be used for the box surrounding the point cloud
        
        Returns:
                pcd (point cloud): PC aligned to origin (all point values are positive)
                box (o3d box): Box that surrounds PC based on PCA to minimize volume.
                box_points (o3d vector): 8 points(x,y,z vals for each) describing the box.
                box_extent (tuple): Extent of the box for x, y and z.
                translation_matrix (array): Transformation matrix of the movement experienced by the PC.
        """
        box = pcd.get_axis_aligned_bounding_box()
        box_points = box.get_box_points()
        box_points = np.asarray(box_points)
        reshaped_points = np.transpose(box_points)
        min_val = np.asarray([np.amin(reshaped_points[0]),
                              np.amin(reshaped_points[1]),
                              np.amin(reshaped_points[2])])
        neg_occ = (min_val!=0)*(-1)
        translation_matrix = min_val*neg_occ
        pcd = pcd.translate((translation_matrix[0],
                             translation_matrix[1],
                             translation_matrix[2]))
        box = pcd.get_axis_aligned_bounding_box()
        box.color = color
        box_points = box.get_box_points()
        box_points = np.asarray(box_points)
        box_extent = np.asarray(box.get_extent())
        return pcd, box, box_points, box_extent, translation_matrix


    # Box functions:


    def create_box_o3d(self, center, R, extent):
        """Takes a center point, R and extent and returns the generated box."""
        box = o3d.geometry.OrientedBoundingBox(center,R,extent)
        return box


    def get_oriented_box(self, pcd, color):
        """Takes a point cloud object and color and returns a pca oriented box, the box extent, r and center."""
        box = pcd.get_oriented_bounding_box()
        box.color = color
        extent = box.extent
        r = box.R
        center = box.center
        return box, extent, r, center


    def get_axis_aligned_box(self, pcd, color = (1,0,0)):
        """Takes a point cloud, returns an axis aligned box, the box's points and the extent."""
        box = pcd.get_axis_aligned_bounding_box()
        box.color = color
        box_points = box.get_box_points()
        box_points = np.asarray(box_points)
        box_extent = np.asarray(box.get_extent())
        num_points = len(pcd.points)
        return pcd, box, box_points, box_extent


    def generate_boxes_from_points(self, points_list, color=(0,1,0)):
        """Takes a list of list of points and color. Returns a list of boxes generated from points."""
        box_list = []
        for i,points in enumerate(points_list):
            box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(points)
            box.color = color
            box_list.append(box)
        return box_list


    def get_box_points_list_from_minmax_vals(self, xyz_minmax_vals_list):
        box_points_list = []
        for element in xyz_minmax_vals_list:
            box_points = self.create_box_points(element[0][0], element[0][1], element[1][0], element[1][1], element[2][0], element[2][1])
            box_points_list.append(box_points)
        return box_points_list
    

    def create_box_points(self, left_x, right_x, down_y, up_y, low_z, high_z):
        """Takes min and max values for x, y and z and returns box points."""
        points =    np.asarray([np.asarray([left_x,down_y,low_z]),
                                np.asarray([right_x,down_y,low_z]),
                                np.asarray([left_x,up_y,low_z]),
                                np.asarray([right_x,up_y,low_z]),
                                np.asarray([left_x,down_y,high_z]),
                                np.asarray([right_x,down_y,high_z]),
                                np.asarray([left_x,up_y,high_z]),
                                np.asarray([right_x,up_y,high_z])])
        box_points = o3d.utility.Vector3dVector(points)
        return box_points


    def get_box_list(self, crop_list, color=(1,0,0)):
        box_list = []
        for crop in crop_list:
            box = crop.get_axis_aligned_bounding_box()
            box.color = color
            box_list.append(box)
        return box_list


    def get_max_min_values_from_box(self, box):
        box_maxmin_list = []
        points = box.get_box_points()
        points = np.asarray(points)
        points_t = np.transpose(points)
        for i in range(3):
            val_max = np.max(points_t[i])
            val_min = np.min(points_t[i])
            box_maxmin_list.append([val_min,val_max])
        return box_maxmin_list


    def get_boxes_with_mean_z(self, point_list, mean_z_list):
        for i, points in enumerate(point_list):
            points[2][2] = mean_z_list[i]
            points[3][2] = mean_z_list[i]
            points[6][2] = mean_z_list[i]
            points[7][2] = mean_z_list[i]
        box_list = self.generate_boxes_from_points(point_list)
        return box_list


    def convert_box_points_to_cloud_box(self, box_points, step=0.01):
        lines = [[0, 1], [0, 2], [0, 4], [1, 3],
                [1, 5], [2, 3], [2, 6], [3, 7],
                [4, 5], [4, 6], [5, 7], [6, 7]]
        np_points = np.asarray(box_points)
        np_pc_points = []
        for i,element in enumerate(lines):
            diff_array = box_points[element[0]]!=box_points[element[1]]
            index  = (np.where(diff_array == True))[0][0]
            if (box_points[element[0]])[index] < (box_points[element[1]])[index]:
                first_index = 0
                second_index = 1
            else:
                fist_index = 1
                second_index = 0

            val1 = (box_points[element[first_index]])[index]
            val2 = (box_points[element[second_index]])[index]

            val_list = np.arange(val1,val2,step).tolist()
            for j,value in enumerate(val_list):
                new_point = copy.deepcopy(box_points[element[first_index]])
                new_point[index] = value
                new_point = np.asarray(new_point)
                np_pc_points.append(new_point)

        for point in box_points:
            np_pc_points.append(point)

        np_pc_points = np.asarray(np_pc_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_pc_points)
        return pcd


    def convert_box_points_list_to_cloud_box_list(self, box_points_list, step=0.01):
        cloud_box_list = []
        for box_points in box_points_list:
            cloud_box = self.convert_box_points_to_cloud_box(box_points,step)
            cloud_box_list.append(cloud_box)
        return cloud_box_list



    # Gridding & partitioning functions:


    def grid(self, pcd, extent, shape):
        """Takes a point cloud, the extent of the box around it and
        the shape of the desired gridding. Returns the point cloud,
        a dictionary with z values and additional info, the shape,
        the gridded model and the translation matrix."""
        # Move point cloud to origin
        pcd, box, points, extent, translation_matrix = self.move_to_origin(pcd)

        # Grid space using shape and extent, then get list of points.
        point_list = self.partition(extent, shape)

        # Create boxes from grid points
        box_list = self.generate_boxes_from_points(point_list)

        # Get crops from point cloud based on boxes
        crop_list = self.get_gridded_crop_list(pcd, box_list)
        # Calculate data for each crop and store it in data dictionary
        mean_z_list = []
        data_dict = {'z_value':[],
                     'coordinates':[],
                     'number_of_points':[],
                     'box_color':[],
                     'box_points':[],
                     'crop_points':[]}
        x = 0
        y = 0
        for i, crop in enumerate(crop_list):
            points_no = len(crop.points)
            box_color = (0,1,0)
            box_points = point_list[i]
            crop_points = crop_list[i]
            crop_array = PointCloudTransformer.open3d_to_numpy(crop)
            crop_array_t = copy.deepcopy(crop_array)
            crop_array_t = np.transpose(crop_array_t)
            if crop_array.any():
                # mean_z = np.mean(crop_array_t[2])
                mean_z = np.mean(crop_array_t[2])
            else:
                mean_z = .1
            crop_array_t[2] = mean_z
            crop_array_t = np.transpose(crop_array_t)
            mean_z = float(mean_z)
            if mean_z <= 0:
                mean_z = .1
            data_dict['z_value'].append(mean_z)
            data_dict['coordinates'].append([x,y])
            data_dict['number_of_points'].append(points_no)
            data_dict['box_color'].append(box_color)
            data_dict['box_points'].append(box_points)
            data_dict['crop_points'].append(crop_points)
            mean_z_list.append(mean_z)
            x += 1
            if x==shape[0]:
                x = 0
                y += 1
                if y==shape[1]:
                    y=0
        mean_z_box_list = self.get_boxes_with_mean_z(point_list, mean_z_list)
        gridded_model = []
        new_box_list = copy.deepcopy(mean_z_box_list)
        for box in new_box_list:
            translated_box = box.translate((-translation_matrix[0],
                                            -translation_matrix[1],
                                            -translation_matrix[2]))
            gridded_model.append(translated_box)
        return pcd, data_dict, shape, gridded_model, translation_matrix


    def partition_gridded_model(self, model_shape, partition_shape):
        """Takes shape of a point cloud and shape of a partition. Returns
        a list of partitions(boxes)."""
        x_len = round(model_shape[0]/partition_shape[0])
        y_len = round(model_shape[1]/partition_shape[1])
        x = 0
        y = 0
        partition_list = []
        for i in range(partition_shape[0]*partition_shape[1]):
            start_x = x*x_len
            end_x =  start_x + x_len - 1
            if x == (partition_shape[0]-1):
                if end_x!=(model_shape[0]-1):
                    end_x = (model_shape[0]-1)

            start_y = y*y_len
            end_y = start_y + y_len - 1
            if y == (partition_shape[1]-1):
                if end_y!=(model_shape[1]-1):
                    end_y = (model_shape[1]-1)
            a = [list(range(start_x,end_x+1)),list(range(start_y,end_y+1))]
            box_coor_list = list(itertools.product(*a))
            partition_list.append(box_coor_list)
            x += 1
            if end_x==(model_shape[0]-1):
                x = 0
                y += 1
                if y==(model_shape[1]-1):
                    y=0
        return partition_list


    def partition(self, extent, shape):
        extent = np.asarray(extent)
        x_partitions = shape[0]
        y_partitions = shape[1]
        x_partition_extent = extent[0]/x_partitions
        y_partition_extent = extent[1]/y_partitions
        min_z = 0
        max_z = extent[2]
        if max_z == 0:
            max_z = .1

        x_measures = []
        for element in range(x_partitions):
            val_0 = element*x_partition_extent
            val_1 = val_0 + x_partition_extent
            measures = [val_0, val_1]
            x_measures.append(measures)

        y_measures = []
        for element in range(y_partitions):
            val_0 = element*y_partition_extent
            val_1 = val_0 + y_partition_extent
            measures = [val_0, val_1]
            y_measures.append(measures)

        partition_point_list = []
        for y in range(y_partitions):
            for x in range(x_partitions):
                points = np.asarray([np.asarray([x_measures[x][0], y_measures[y][0], min_z]),
                                     np.asarray([x_measures[x][1], y_measures[y][0], min_z]),
                                     np.asarray([x_measures[x][0], y_measures[y][0], max_z]),
                                     np.asarray([x_measures[x][1], y_measures[y][0], max_z]),
                                     np.asarray([x_measures[x][0], y_measures[y][1], min_z]),
                                     np.asarray([x_measures[x][1], y_measures[y][1], min_z]),
                                     np.asarray([x_measures[x][0], y_measures[y][1], max_z]),
                                     np.asarray([x_measures[x][1], y_measures[y][1], max_z])])
                points = o3d.utility.Vector3dVector(points)
                partition_point_list.append(points)
        return partition_point_list


    def get_window_length(self, l, w, padding):
        # Get max value
        val_list = [float(l),float(w)]
        max_side = max(val_list)
        # print(f'Max value is {max_side}')

        # Calculate value of interior square side (i_len)
        i_len = max_side + 2*padding

        # Calculate value of exterior square side (e_len)
        e_len = (2*(math.sqrt(((i_len)**2))/2))
        # print(f"\nSearching for objects with length of {l} meters and width of {w} meters.\n")
        print(f"\nUsing a {e_len} x {e_len} meters window.\n")
        return e_len


    def sliding_window(self, window_len, pcd_shape, step):
        pcd_x = pcd_shape[0]
        pcd_y = pcd_shape[1]
        pcd_z = pcd_shape[2]
        x_counter = 0
        y_counter = 0
        window_list = []
        stop_flag = False

        while(True):
            w_y_min = step*y_counter
            w_y_max = w_y_min + window_len
            if w_y_max == pcd_y:
                stop_flag = True
                y_counter = 0
            elif w_y_max > pcd_y:
                stop_flag = True
                w_y_max = pcd_y
                w_y_min = pcd_y - window_len
                y_counter = 0
            else: 
                y_counter += 1
            while(True):
                w_x_min = step*x_counter
                w_x_max = w_x_min + window_len
                if w_x_max == pcd_x:
                    window_data = [[w_x_min, w_x_max],[w_y_min, w_y_max],[0, pcd_z]]
                    window_list.append(window_data)
                elif w_x_max > pcd_x:
                    w_x_max = pcd_x
                    w_x_min = pcd_x - window_len
                    window_data = [[w_x_min, w_x_max],[w_y_min, w_y_max],[0, pcd_z]]
                    window_list.append(window_data)
                    x_counter = 0
                    break
                else:
                    x_counter += 1
                    window_data = [[w_x_min, w_x_max],[w_y_min, w_y_max],[0, pcd_z]]
                    window_list.append(window_data)
            if stop_flag:
                break
        return window_list


    # Data dictionary functions:


    def get_index_from_coordinates(self, data_dict, coordinates):
        """Takes dictionary with varied info (z_value, coordinates
        number of points, box color, box points, crop points) as well as 
        coordinates of cell from a gridded model. Returns linear index
        of these coordinates."""
        index = data_dict['coordinates'].index(coordinates)
        return index


    def get_index_from_z_value(self, data_dict, z_value):
        """Takes dictionary with varied info and a z value. Returns
        a linear index corresponding to this z val."""
        index = data_dict['z_value'].index(z_value)
        return index


    def get_max_points_to_evaluate(self, data_dict, partition_list):
        """Takes a data dictionary with varied values and a list of boxes. Returns
        a list of coordinates of the max z value of each partition."""
        max_coordinates_list = []
        for i,partition in enumerate(partition_list):
            partition_z_list = []
            index_list = []
            for element in partition:
                coor = list(element)
                index = self.get_index_from_coordinates(data_dict, coor)
                z_val = data_dict['z_value'][index]
                partition_z_list.append(z_val)
                index_list.append(index)
            max_z = max(partition_z_list)
            max_index = partition_z_list.index(max_z)
            index = index_list[max_index]
            max_coor = data_dict['coordinates'][index]
            max_coordinates_list.append(max_coor)
        return max_coordinates_list


    # Detection functions:


    def search_for_objects(self, pcd, grid_shape, search_areas_shape,
                           extent, steepness_threshold):
        """Takes a point cloud, desired grid shape, shape of the areas to shape, extent of pcds box
        and a steepness threshold. Returns a list of detection boxes."""
        pcd, data_dict, shape, gridded_model, translation_matrix = partitioner.grid(pcd, extent, (10,20))
        max_z = max(data_dict['z_value'])
        start_index = data_dict['z_value'].index(max_z)
        clean_grid_model = copy.deepcopy(gridded_model)
        partition_list = self.partition_gridded_model(shape,
                                                             search_areas_shape)
        evaluation_coordinates = self.get_max_points_to_evaluate(data_dict,
                                                                    partition_list)
        detection_box_list = []
        for coordinates in evaluation_coordinates:
            temp_grid_model = copy.deepcopy(clean_grid_model)
            start_index = data_dict['coordinates'].index(coordinates)
            box_points, detection_box, result, temp_grid_model = self.detect_object(pcd,
                                                                                    data_dict,
                                                                                    shape,
                                                                                    temp_grid_model,
                                                                                    start_index,
                                                                                    steepness_threshold)
            if detection_box:
                detection_box = detection_box.translate((-translation_matrix[0],
                                                         -translation_matrix[1],
                                                         -translation_matrix[2]))
                detection_box_list.append(detection_box)
        if detection_box_list:
            detection_box_list = self.purge_duplicated_detections(detection_box_list,
                                                                  iou_threshold=.5)
        return detection_box_list


    def naive_detector(self, dif_values, max_z):
        w_tens = [.2,.2,.2,.2,.2,.2,.2,.2]
        w_tens = np.asarray(w_tens)
        dif_values = np.asarray(dif_values)
        np.append(dif_values, max_z)
        dif_values = dif_values*w_tens
        sum_val = np.sum(dif_values)
        sig_val = self.sigmoid(sum_val)
        return sig_val


    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))


    def purge_duplicated_detections(self, detection_box_list,
                                    iou_threshold = .5, result_list=[]):
        start_box = detection_box_list.pop(0)
        new_list = []
        for box in detection_box_list:
            overlap_flag, overlap_box, iou = self.detect_overlap(start_box, box)
            if overlap_flag == False:
                new_list.append(box)
        result_list.append(start_box)
        if new_list:
            result = self.purge_duplicated_detections(new_list,
                                                      iou_threshold,
                                                      result_list)
            return result
        else:
            return result_list


    def detect_object(self, pcd, data_dict, shape,
                      gridded_model, start_index, steepness):
        max_z = max(data_dict['z_value'])
        index = start_index
        gridded_model[index].color = (1,0,0)
        start_z_val = data_dict['z_value'][index]
        coordinates = data_dict['coordinates'][index]
        y_movements_up = shape[1] - coordinates[1] - 1
        y_movements_down = coordinates[1]
        x_movements_right = shape[0] - coordinates[0] - 1
        x_movements_left = coordinates[0]
        dif_values = []
        highest_dif_values = []

        # Draw Vertical lines
        
        # Up
        max_dif = 0
        prev_val = start_z_val
        dif_coordinates = [0,0]
        up_y = 0
        uphill_flag = False
        for element in range(y_movements_up):
            m = element + 1
            new_coordinates = copy.deepcopy(coordinates)
            new_coordinates[1] = new_coordinates[1] + m
            index = self.get_index_from_coordinates(data_dict, new_coordinates)
            box_points = data_dict['box_points'][index]
            curr_z_val = data_dict['z_value'][index]
            dif = prev_val-curr_z_val
            gridded_model[index].color = (1,0,0)
            if dif<0:
                if uphill_flag == True:
                    break
                uphill_flag = True
            else:
                uphill_flag = False
            if dif>max_dif:
                max_dif = dif
                dif_coordinates = new_coordinates
                up_y = box_points[4][1]
            prev_val = curr_z_val
        proportional_dif = max_dif/max_z
        dif_values.append(proportional_dif)
        highest_dif_values.append(dif_coordinates)
        if up_y == 0:
            new_coordinates = [shape[0]-1,shape[1]-1]
            index = self.get_index_from_coordinates(data_dict, new_coordinates)
            box_points = data_dict['box_points'][index]
            up_y = box_points[4][1]
        
        # Down
        max_dif = 0
        prev_val = start_z_val
        dif_coordinates = [0,0]
        down_y = 0
        uphill_flag = False
        for element in range(y_movements_down):
            m = element + 1
            new_coordinates = copy.deepcopy(coordinates)
            new_coordinates[1] = new_coordinates[1] - m
            index = self.get_index_from_coordinates(data_dict, new_coordinates)
            box_points = data_dict['box_points'][index]
            curr_z_val = data_dict['z_value'][index]
            dif = prev_val-curr_z_val
            gridded_model[index].color = (1,0,0)
            if dif<0:
                if uphill_flag == True:
                    break
                uphill_flag = True
            else:
                uphill_flag = False
            if dif>max_dif:
                max_dif = dif
                dif_coordinates = new_coordinates
                down_y = box_points[3][1]
            prev_val = curr_z_val
        proportional_dif = max_dif/max_z
        dif_values.append(proportional_dif)
        highest_dif_values.append(dif_coordinates)

        # Draw Horizontal Lines

        # Right
        max_dif = 0
        prev_val = start_z_val
        dif_coordinates = [0,0]
        right_x = 0
        uphill_flag = False
        for element in range(x_movements_right):
            m = element + 1
            new_coordinates = copy.deepcopy(coordinates)
            new_coordinates[0] = new_coordinates[0] + m
            index = self.get_index_from_coordinates(data_dict, new_coordinates)
            box_points = data_dict['box_points'][index]
            curr_z_val = data_dict['z_value'][index]
            dif = prev_val-curr_z_val
            gridded_model[index].color = (1,0,0)
            if dif<0:
                if uphill_flag == True:
                    break
                uphill_flag = True
            else:
                uphill_flag = False
            if dif>max_dif:
                max_dif = dif
                dif_coordinates = new_coordinates
                right_x = box_points[1][0]
            prev_val = curr_z_val
        proportional_dif = max_dif/max_z
        dif_values.append(proportional_dif)
        highest_dif_values.append(dif_coordinates)
        if right_x == 0:
            new_coordinates = [shape[0]-1,shape[1]-1]
            index = self.get_index_from_coordinates(data_dict, new_coordinates)
            box_points = data_dict['box_points'][index]
            right_x = box_points[1][0]

        # Left
        max_dif = 0
        prev_val = start_z_val
        dif_coordinates = [0,0]
        left_x = 0
        uphill_flag = False
        for element in range(x_movements_left):
            m = element + 1
            new_coordinates = copy.deepcopy(coordinates)
            new_coordinates[0] = new_coordinates[0] - m
            index = self.get_index_from_coordinates(data_dict, new_coordinates)
            box_points = data_dict['box_points'][index]
            curr_z_val = data_dict['z_value'][index]
            dif = prev_val-curr_z_val
            gridded_model[index].color = (1,0,0)
            if dif<0:
                if uphill_flag == True:
                    break
                uphill_flag = True
            else:
                uphill_flag = False
            if dif>max_dif:
                max_dif = dif
                dif_coordinates = new_coordinates
                left_x = box_points[0][0]
            prev_val = curr_z_val
        proportional_dif = max_dif/max_z
        dif_values.append(proportional_dif)
        highest_dif_values.append(dif_coordinates)
        
        # Diagonals

        # # diagonal_left_down_up
        max_dif = 0
        prev_val = start_z_val
        new_coordinates = copy.deepcopy(coordinates)
        uphill_flag = False
        while(True):
            new_coordinates[0] = new_coordinates[0] - 1
            new_coordinates[1] = new_coordinates[1] - 1
            if new_coordinates[0]<0 or new_coordinates[1]<0:
                break
            index = self.get_index_from_coordinates(data_dict, new_coordinates)
            curr_z_val = data_dict['z_value'][index]
            dif = prev_val-curr_z_val
            gridded_model[index].color = (1,0,0)
            if dif<0:
                if uphill_flag == True:
                    break
                uphill_flag = True
            else:
                uphill_flag = False
            if dif>max_dif:
                max_dif = dif
            prev_val = curr_z_val
            if new_coordinates[0]<=0 or new_coordinates[1]<=0:
                break
        proportional_dif = max_dif/max_z
        dif_values.append(proportional_dif)
               
        # diagonal_right_up_down
        max_dif = 0
        prev_val = start_z_val
        new_coordinates = copy.deepcopy(coordinates)
        uphill_flag = False
        while(True):
            new_coordinates[0] = new_coordinates[0] + 1
            new_coordinates[1] = new_coordinates[1] - 1
            if new_coordinates[0]>(shape[0]-1) or new_coordinates[1]<=0:
                break
            index = self.get_index_from_coordinates(data_dict, new_coordinates)
            curr_z_val = data_dict['z_value'][index]
            dif = prev_val-curr_z_val
            gridded_model[index].color = (1,0,0)
            if dif<0:
                if uphill_flag == True:
                    break
                uphill_flag = True
            else:
                uphill_flag = False
            if dif>max_dif:
                max_dif = dif
            prev_val = curr_z_val
            if new_coordinates[0]==(shape[0]-1) or new_coordinates[1]==0:
                break
        proportional_dif = max_dif/max_z
        dif_values.append(proportional_dif)

        # diagonal_left_up_down
        max_dif = 0
        prev_val = start_z_val
        new_coordinates = copy.deepcopy(coordinates)
        uphill_flag = False
        while(True):
            new_coordinates[0] = new_coordinates[0] - 1
            new_coordinates[1] = new_coordinates[1] + 1
            if new_coordinates[0]<0 or new_coordinates[1]>(shape[1]-1):
                break
            index = self.get_index_from_coordinates(data_dict, new_coordinates)
            curr_z_val = data_dict['z_value'][index]
            dif = prev_val-curr_z_val
            gridded_model[index].color = (1,0,0)
            if dif<0:
                if uphill_flag == True:
                    break
                uphill_flag = True
            else:
                uphill_flag = False
            if dif>max_dif:
                max_dif = dif
            prev_val = curr_z_val
            if new_coordinates[0]==0 or new_coordinates[1]==(shape[1]-1):
                break
        proportional_dif = max_dif/max_z
        dif_values.append(proportional_dif)

        # diagonal_right_down_up
        max_dif = 0
        prev_val = start_z_val
        new_coordinates = copy.deepcopy(coordinates)
        uphill_flag = False
        while(True):
            new_coordinates[0] = new_coordinates[0] + 1
            new_coordinates[1] = new_coordinates[1] + 1
            if new_coordinates[0]>(shape[0]-1) or new_coordinates[1]>(shape[1]-1):
                break
            index = self.get_index_from_coordinates(data_dict, new_coordinates)
            curr_z_val = data_dict['z_value'][index]
            dif = prev_val-curr_z_val
            gridded_model[index].color = (1,0,0)
            if dif<0:
                if uphill_flag == True:
                    break
                uphill_flag = True
            else:
                uphill_flag = False
            if dif>max_dif:
                max_dif = dif
            prev_val = curr_z_val
            if new_coordinates[0]==(shape[0]-1) or new_coordinates[1]==(shape[1]-1):
                break
        proportional_dif = max_dif/max_z
        dif_values.append(proportional_dif)
        low_z = self.get_min_z(pcd)
        high_z = self.get_max_z(pcd)
        box_points = self.create_box_points(left_x,right_x,down_y,up_y,low_z,high_z)
        detection_box = o3d.geometry.OrientedBoundingBox.create_from_points(box_points)
        detection_box.color = (1,0,0)
        det_val = self.naive_detector(dif_values, max_z)
        if det_val >= steepness:
            result = True
        else:
            result = False
            detection_box = None
        return box_points, detection_box, result, gridded_model


    def detect_overlap(self, box_a, box_b):
        """Takes two boxes and returns overlap boolean, the
        overlapping box and the IoU (Intersection of Union)."""
        x_overlap_flag = False
        y_overlap_flag = False
        z_overlap_flag = False

        x_len = 0
        y_len = 0
        z_len = 0

        a_vals = self.get_max_min_values_from_box(box_a)
        b_vals = self.get_max_min_values_from_box(box_b)

        min_a_x = a_vals[0][0]
        max_a_x = a_vals[0][1]
        min_a_y = a_vals[1][0]
        max_a_y = a_vals[1][1]
        min_a_z = a_vals[2][0]
        max_a_z = a_vals[2][1]
        
        min_b_x = b_vals[0][0]
        max_b_x = b_vals[0][1]
        min_b_y = b_vals[1][0]
        max_b_y = b_vals[1][1]
        min_b_z = b_vals[2][0]
        max_b_z = b_vals[2][1]

        # x overlap
        if max_a_x > max_b_x:
            if max_b_x > min_a_x:
                x_overlap_flag = True
                max_overlap_x = max_b_x
                if min_b_x > min_a_x:
                    x_len = max_b_x - min_b_x
                    min_overlap_x = min_b_x
                elif min_a_x > min_b_x:
                    x_len = max_b_x - min_a_x
                    min_overlap_x = min_a_x
                else:
                    x_len = max_b_x - min_b_x
                    min_overlap_x = min_a_x

        if max_b_x > max_a_x:
            if max_a_x > min_b_x:
                x_overlap_flag = True
                max_overlap_x = max_a_x
                if min_a_x > min_b_x:
                    x_len = max_a_x - min_a_x
                    min_overlap_x = min_a_x
                elif min_b_x > min_a_x:
                    x_len = max_a_x - min_b_x
                    min_overlap_x = min_b_x
                else:
                    x_len = max_a_x - min_a_x
                    min_overlap_x = min_b_x

        if max_a_x == max_b_x:
            x_overlap_flag = True
            max_overlap_x = max_a_x
            if min_a_x > min_b_x:
                x_len = max_a_x - min_a_x
                min_overlap_x = min_a_x
            elif min_b_x > min_a_x:
                x_len = max_b_x - min_b_x
                min_overlap_x = min_b_x
            else:
                x_len = max_a_x - min_a_x
                min_overlap_x = min_b_x

        # y overlap
        if max_a_y > max_b_y:
            if max_b_y > min_a_y:
                y_overlap_flag = True
                max_overlap_y = max_b_y
                if min_b_y > min_a_y:
                    y_len = max_b_y - min_b_y
                    min_overlap_y = min_b_y
                elif min_a_y > min_b_y:
                    y_len = max_b_y - min_a_y
                    min_overlap_y = min_a_y
                else:
                    y_len = max_b_y - min_b_y
                    min_overlap_y = min_a_y

        if max_b_y > max_a_y:
            if max_a_y > min_b_y:
                y_overlap_flag = True
                max_overlap_y = max_a_y
                if min_a_y > min_b_y:
                    y_len = max_a_y - min_a_y
                    min_overlap_y = min_a_y
                elif min_b_y > min_a_y:
                    y_len = max_a_y - min_b_y
                    min_overlap_y = min_b_y
                else:
                    y_len = max_a_y - min_a_y
                    min_overlap_y = min_b_y

        if max_a_y == max_b_y:
            y_overlap_flag = True
            max_overlap_y = max_a_y
            if min_a_y > min_b_y:
                y_len = max_a_y - min_a_y
                min_overlap_y = min_a_y
            elif min_b_y > min_a_y:
                y_len = max_b_y - min_b_y
                min_overlap_y = min_b_y
            else:
                y_len = max_a_y - min_a_y
                min_overlap_y = min_b_y
        
        # z overlap
        if max_a_z > max_b_z:
            if max_b_z > min_a_z:
                z_overlap_flag = True
                max_overlap_z = max_b_z
                if min_b_z > min_a_z:
                    z_len = max_b_z - min_b_z
                    min_overlap_z = min_b_z
                elif min_a_z > min_b_z:
                    z_len = max_b_z - min_a_z
                    min_overlap_z = min_a_z
                else:
                    z_len = max_b_z - min_b_z
                    min_overlap_z = min_a_z

        if max_b_z > max_a_z:
            if max_a_z > min_b_z:
                z_overlap_flag = True
                max_overlap_z = max_a_z
                if min_a_z > min_b_z:
                    z_len = max_a_z - min_a_z
                    min_overlap_z = min_a_z
                elif min_b_z > min_a_z:
                    z_len = max_a_z - min_b_z
                    min_overlap_z = min_b_z
                else:
                    z_len = max_a_z - min_a_z
                    min_overlap_z = min_b_z

        if max_a_z == max_b_z:
            z_overlap_flag = True
            max_overlap_z = max_a_z
            if min_a_z > min_b_z:
                z_len = max_a_z - min_a_z
                min_overlap_z = min_a_z
            elif min_b_y > min_a_y:
                z_len = max_b_z - min_b_z
                min_overlap_z = min_b_z
            else:
                z_len = max_a_z - min_a_z
                min_overlap_z = min_b_z

        if x_overlap_flag and y_overlap_flag and z_overlap_flag:
            result = True
        else:
            result = False

        box_a_vol = abs((a_vals[0][1]-a_vals[0][0])*(a_vals[1][1]-a_vals[1][0])*(a_vals[2][1]-a_vals[2][0]))
        box_b_vol = abs((b_vals[0][1]-b_vals[0][0])*(b_vals[1][1]-b_vals[1][0])*(b_vals[2][1]-b_vals[2][0]))
        if result:
            overlap_vol = abs(x_len*y_len*z_len)
            union_vol = box_a_vol + box_b_vol - overlap_vol
            iou = overlap_vol/union_vol
            overlap_points =    np.asarray([np.asarray([min_overlap_x, min_overlap_y, min_overlap_z]),
                                            np.asarray([max_overlap_x, min_overlap_y, min_overlap_z]),
                                            np.asarray([min_overlap_x, min_overlap_y, max_overlap_z]),
                                            np.asarray([max_overlap_x, min_overlap_y, max_overlap_z]),
                                            np.asarray([min_overlap_x, max_overlap_y, min_overlap_z]),
                                            np.asarray([max_overlap_x, max_overlap_y, min_overlap_z]),
                                            np.asarray([min_overlap_x, max_overlap_y, max_overlap_z]),
                                            np.asarray([max_overlap_x, max_overlap_y, max_overlap_z])])
            overlap_points = o3d.utility.Vector3dVector(overlap_points) 
            overlap_box = o3d.geometry.OrientedBoundingBox.create_from_points(overlap_points)
            overlap_box.color = (0,1,0)
        else:
            overlap_vol = 0
            union_vol = 0
            iou = 0
            overlap_box = None
        return result, overlap_box, iou


    def get_indexes(self, nested_dict, value, prepath=()):
        for k, v in nested_dict.items():
            path = prepath + (k,)
            if v == value: 
                return path
            elif hasattr(v, 'items'):
                p = self.get_indexes(v, value, path)
                if p is not None:
                    return p


    # Crop functions


    def crop_from_o3d(self, pcd, box):
        """Takes a point cloud and a box, and returns a cropped point cloud
        based on the box dimensions and position."""
        crop = pcd.crop(box)
        return crop
    

    def get_gridded_crop_list(self, pcd, box_list):
        crop_list = []
        for box in box_list:
            crop = self.crop_from_o3d(pcd, box)
            crop_list.append(crop)
        return crop_list
    

    def get_translated_crop_list(self, crop_list, extent, shape, separation_percentage):
        separation_x = extent[0]*(separation_percentage/100)
        separation_y = extent[1]*(separation_percentage/100)
        translated_crop_list = copy.deepcopy(crop_list)
        object_counter = 0
        for y in range(shape[1]):
            for x in range(shape[0]):
                shift_x = separation_x*x 
                shift_y = separation_y*y
                translated_crop_list[object_counter] = translated_crop_list[object_counter].translate((shift_x, shift_y, 0))
                object_counter += 1
        return translated_crop_list    


    def save_o3d_list_as_ply(self, output_path, object_list, name_prefix='sample',
                             translation_mat=[0,0,0]):
        for i, object in enumerate(object_list):
            file_name = ''.join([output_path, name_prefix,"_",str(i),'.ply'])
            PointCloudWriter.o3d_as_ply(object,file_name)


    def create_folder(self, directory, parent_dir, mode):
        mode = 0o666
        path = os.path.join(parent_dir, directory)
        os.mkdir(path, mode)


    # Draw functions:


    def draw_object_lists(self, *args):
        object_list = [j for i in args for j in i]
        o3d.visualization.draw_geometries(object_list)



    def draw_detection_on_partition(self, partition, partition_box, detection_points):
        detection_box = o3d.geometry.OrientedBoundingBox.create_from_points(detection_points)
        o3d.visualization.draw_geometries([partition, detection_box])

    
    def draw_detection_on_original_model(self, pcd, pcd_box, detection_points, tran_matrix):
        detection_box = o3d.geometry.OrientedBoundingBox.create_from_points(detection_points)
        o3d.visualization.draw_geometries([pcd, detection_box])
        o3d.visualization.close()


    def draw_o3d(self, *args):
        """Takes a variable number of lists with point clouds. Returns nothing."""
        o3d.visualization.draw_geometries(args)


    def draw_partition_crop_list(self, crop_list, extent, shape, separation_percentage):
        padding_x = extent[0]*(separation_percentage/100)
        padding_y = extent[1]*(separation_percentage/100)
        translated_crop_list = crop_list.copy()
        object_counter = 0
        for y in range(shape[1]):
            for x in range(shape[0]):
                shift_x = padding_x*x 
                shift_y = padding_y*y
                translated_crop_list[object_counter] = translated_crop_list[object_counter].translate((shift_x, shift_y, 0))
                object_counter += 1
        o3d.visualization.draw_geometries(translated_crop_list)


    def display_box_points_as_lines_on_pcd(self, pcd, box_point_list):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for box_points in box_point_list:
            lines = [[0, 1], [0, 2], [0, 4], [1, 3],
                    [1, 5], [2, 3], [2, 6], [3, 7],
                    [4, 5], [4, 6], [5, 7], [6, 7]]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(box_points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            # Display the bounding boxes:
            vis.add_geometry(line_set)
        vis.add_geometry(pcd)
        vis.run()

    
    def display_pcd_list(self, pcd_list, quantity=None, info=False):
        if len(pcd_list) == 0:
            print('[INFO] Empty PCD list.')
            return None
        if quantity:
            if quantity>(len(pcd_list)):
                print('[INFO] Number of elements to print is greater than lists length.')
                return None
            for i in range(quantity):
                extent = self.get_object_extent(pcd_list[i])
                if info:
                    print(f'\n Sample {i+1}.')
                    print(f'Number of points: {len(pcd_list[i].points)}')
                    print(f'Estimated dimensions: {extent} meters.')
                o3d.visualization.draw_geometries([pcd_list[i]])
        else:
            for i,element in enumerate(pcd_list):
                 o3d.visualization.draw_geometries([element])


    # Dataset functions:


    def crop_dataset_from_pcd(self, pcd, object_length, object_width, window_padding=.2, window_step=1.2):
        # Move to origin
        pcd, box, points, extent, translation_matrix = self.move_to_origin(pcd)

        # Get crop box points using sliding window
        win_len = self.get_window_length(object_length, object_width, window_padding)
        step = 1.2
        crop_box_minmax_vals = self.sliding_window(win_len, extent, step)
        box_points_list = self.get_box_points_list_from_minmax_vals(crop_box_minmax_vals)

        # Get boxes from points
        box_list = self.generate_boxes_from_points(box_points_list, color=(0,1,0))

        # Get cropped pcds 
        crop_list = self.get_gridded_crop_list(pcd, box_list)
        return crop_list, box_list, box_points_list


    def resample_dataset(self, pcd_list, num_points=2048):
        resampled_list = []
        for pcd in pcd_list:
            res_pcd = self.resample_to_2048_points(pcd)
            resampled_list.append(res_pcd)
        return resampled_list


    def normalize_dataset(self, pcd_list):
        normalized_list = []
        for pcd in pcd_list:
            norm_pcd = self.normalize_point_cloud(pcd)
            normalized_list.append(norm_pcd)
        return normalized_list


    def save_dataset(self, pcd_list, dataset_name, output_path, connector='_'):
        path = os.path.join(output_path, dataset_name)
        os.mkdir(path)
        for i,pcd in enumerate(pcd_list):
            file_number = '0'*(4-len(str(i))) + str(i)
            file_name = ''.join([path,'/', dataset_name, connector, file_number, '.ply'])
            print(file_name)
            o3d.io.write_point_cloud(file_name, pcd, write_ascii=True)


    # Examples:


    def example_001_move_to_origin(self):
        # Open point cloud file and make a copy
        filepath = 'samples/other/munition_test.ply'
        original_pcd = self.read_ply_as_o3d(filepath)
        pcd = copy.deepcopy(original_pcd)

        # Move to origin
        pcd, box, points, extent, translation_matrix = self.move_to_origin(pcd)

        # Show original object and translated object
        self.draw_object_lists([original_pcd], [pcd])


    def example_002_partition(self):
        # Open point cloud file and make a copy
        filepath = 'samples/other/munition_test.ply'
        original_pcd = self.read_ply_as_o3d(filepath)
        pcd = copy.deepcopy(original_pcd)

        # Move to origin
        pcd, box, points, extent, translation_matrix = self.move_to_origin(pcd)

        # Partition point cloud
        partition_shape = (5,3)
        partition_point_list = partitioner.partition(extent, partition_shape)

        # Generate boxes from points
        box_list = partitioner.generate_boxes_from_points(partition_point_list)

        # Get crop list
        crop_list = partitioner.get_gridded_crop_list(pcd, box_list)

        # Draw partition crop list
        separation_percentage = 10
        translated_crop_list = partitioner.get_translated_crop_list(crop_list, extent, partition_shape, separation_percentage)

        # Show translated crop list
        partitioner.draw_object_lists(translated_crop_list)


    def example_003_partition_with_boxes(self):
        # Open point cloud file and make a copy
        filepath = 'samples/other/munition_test.ply'
        original_pcd = self.read_ply_as_o3d(filepath)
        pcd = copy.deepcopy(original_pcd)

        # Move to origin
        pcd, box, points, extent, translation_matrix = self.move_to_origin(pcd)

        # Partition point cloud
        partition_shape = (5,3)
        partition_point_list = partitioner.partition(extent, partition_shape)

        # Generate boxes from points
        box_list = partitioner.generate_boxes_from_points(partition_point_list)

        # Get crop list
        crop_list = partitioner.get_gridded_crop_list(pcd, box_list)

        # Get crop box list
        crop_box_list = partitioner.get_box_list(crop_list)

        # Show  crop and box lists
        partitioner.draw_object_lists(crop_list, crop_box_list)


    def example_004_gridding(self):
        # Open point cloud file and make a copy
        filepath = 'samples/other/munition_test.ply'
        original_pcd = self.read_ply_as_o3d(filepath)
        pcd = copy.deepcopy(original_pcd)

        # Move to origin
        pcd, box, points, extent, translation_matrix = self.move_to_origin(pcd)

        # Partition point cloud
        partition_shape = (5,3)
        partition_point_list = self.partition(extent, partition_shape)

        # Generate boxes from points
        box_list = self.generate_boxes_from_points(partition_point_list)

        # Get crop list
        crop_list = self.get_gridded_crop_list(pcd, box_list)

        # Select single crop
        crop = crop_list[2]

        # Grid crop
        grid_shape = [10,15]
        crop, data_dict, shape, gridded_crop, translation_matrix = self.grid(crop, extent, grid_shape)

        # Show crop
        self.draw_object_lists([crop], gridded_crop)


    def example_005_sliding_window(self):
        # Open point cloud file and make a copy
        filepath = 'samples/other/munition_test.ply'
        original_pcd = self.read_ply_as_o3d(filepath)
        pcd = copy.deepcopy(original_pcd)

        # Move to origin
        pcd, box, points, extent, translation_matrix = self.move_to_origin(pcd)

        # Get crop boxes using sliding window
        window_len = 2
        step = 1.2
        crop_box_minmax_vals = self.sliding_window(window_len, extent, step)
        box_points_list = self.get_box_points_list_from_minmax_vals(crop_box_minmax_vals)

        # Convert boxes to pointclouds for visualization
        cloud_box_list = self.convert_box_points_list_to_cloud_box_list(box_points_list, step=0.01)

        # Visualize slidding window
        visualizer = PointCloudVisualizer()
        visualizer.display_animation_on_pcd(pcd, cloud_box_list, pose_path='settings/pose.json', time_step=0.1, save=True)


    def example_006_crop_with_sliding_window(self):
        # Open point cloud file and make a copy
        filepath = 'samples/other/munition_test.ply'
        original_pcd = self.read_ply_as_o3d(filepath)
        pcd = copy.deepcopy(original_pcd)

        # Move to origin
        pcd, box, points, extent, translation_matrix = self.move_to_origin(pcd)

        # Get crop box points using sliding window
        window_len = 2
        step = 1.2
        crop_box_minmax_vals = self.sliding_window(window_len, extent, step)
        box_points_list = self.get_box_points_list_from_minmax_vals(crop_box_minmax_vals)

        # Get boxes from points
        box_list = self.generate_boxes_from_points(box_points_list, color=(0,1,0))

        # Get cropped pcds 
        crop_list = self.get_gridded_crop_list(pcd, box_list)

        # Convert boxes to pointclouds for visualization
        cloud_box_list = self.convert_box_points_list_to_cloud_box_list(box_points_list, step=0.01)

        # Visualize slidding window
        visualizer = PointCloudVisualizer()
        visualizer.display_animation_on_pcd(pcd, cloud_box_list, pose_path='settings/pose.json', time_step=0.1, save=True)

        # Show some crops
        for i in range(10):
            print(len(crop_list[i].points))
            self.draw_object_lists([crop_list[i]])


    def example_007_downsample_point_cloud(self):
        # Open point cloud file and make a copy
        filepath = 'samples/other/munition_test.ply'
        original_pcd = self.read_ply_as_o3d(filepath)
        pcd = copy.deepcopy(original_pcd)

        # Move to origin
        pcd, box, points, extent, translation_matrix = self.move_to_origin(pcd) 

        # Print original number of points
        orig_point_no = len(pcd.points)
        print(f"Number of points in original point cloud: {orig_point_no}")

        # Downsample to 2048 points
        pcd_down = self.downsample_point_cloud(pcd, 2048)
        print(f'Number of points after downsampling: {len(pcd_down.points)}')
        down_point_no = len(pcd_down.points)

        # Draw original and downsampled models
        self.draw_object_lists([pcd])
        self.draw_object_lists([pcd_down])


    def example_008_upsample_point_cloud(self):
        # Open point cloud file and make a copy
        filepath = 'samples/other/munition_test.ply'
        original_pcd = self.read_ply_as_o3d(filepath)
        pcd = copy.deepcopy(original_pcd)

        # Move to origin
        pcd, box, points, extent, translation_matrix = self.move_to_origin(pcd)

        # Downsample to 1024 points
        pcd_down = self.downsample_point_cloud(pcd, 5550)

        # Upsample to 2048
        self.upsample_point_cloud(pcd_down, 2048)


    def example_009_normalize_point_cloud(self):
        # Open point cloud file and make a copy
        filepath = 'samples/other/munition_test.ply'
        original_pcd = self.read_ply_as_o3d(filepath)
        pcd = copy.deepcopy(original_pcd)

        # Move to origin
        pcd, box, points, extent, translation_matrix = self.move_to_origin(pcd)

        # Partition point cloud
        partition_shape = (5,3)
        partition_point_list = partitioner.partition(extent, partition_shape)

        # Generate boxes from points
        box_list = partitioner.generate_boxes_from_points(partition_point_list)

        # Get crop list
        crop_list = partitioner.get_gridded_crop_list(pcd, box_list)
        sample_crop = crop_list[2]

        # Normalize crop
        norm_crop = self.normalize_point_cloud(sample_crop)
        self.draw_object_lists([norm_crop],[sample_crop])


    def example_010_create_and_display_dataset(self):
        # Open point cloud file and make a copy
        filepath = fd.askopenfilename()
        original_pcd = self.read_ply_as_o3d(filepath)
        pcd = copy.deepcopy(original_pcd)

        # Create dataset
        object_length = 2
        object_width = 5
        base_pcd_dim = self.get_object_extent(pcd)
        print(f"\nSearch area has dimensions: {base_pcd_dim} meters.\n")
        print(f"\nSearching for objects with length of {object_length} meters and width of {object_width} meters.\n")
        crop_list, box_list, box_points_list = self.crop_dataset_from_pcd(pcd, object_length, object_width)

        # Convert boxes to pointclouds for visualization
        cloud_box_list = self.convert_box_points_list_to_cloud_box_list(box_points_list, step=0.01)

        # Visualize slidding window
        visualizer = PointCloudVisualizer()
        visualizer.display_animation_on_pcd(pcd, cloud_box_list, pose_path='settings/pose.json', time_step=0.1, save=False)

        # Resample dataset to 2048 points
        res_pcd_list = self.resample_dataset(crop_list)

        # Normalize dataset
        norm_pcd_list = self.normalize_dataset(res_pcd_list)

        # Display some of the preprocessed point clouds
        self.display_pcd_list(norm_pcd_list, 10, info=True)


    def example_011_create_and_save_dataset(self):
        # Open point cloud file and make a copy
        filepath = fd.askopenfilename()
        output_folder = fd.askdirectory()
        filename = filepath.split('/')[-1]
        dataset_name = filename.split('.')[-2]
        original_pcd = self.read_ply_as_o3d(filepath)
        pcd = copy.deepcopy(original_pcd)

        
        # Display source PCD
        self.draw_object_lists([pcd])
        
        # Create dataset
        crop_list, box_list, box_points_list = self.crop_dataset_from_pcd(pcd,3,4)

        # Convert boxes to pointclouds for visualization
        cloud_box_list = self.convert_box_points_list_to_cloud_box_list(box_points_list, step=0.01)


        # Visualize slidding window
        visualizer = PointCloudVisualizer()
        visualizer.display_animation_on_pcd(pcd, cloud_box_list, pose_path='settings/pose.json', time_step=0.05, save=False)

        # Resample dataset to 2048 points
        res_pcd_list = self.resample_dataset(crop_list)

        # Normalize dataset
        norm_pcd_list = self.normalize_dataset(res_pcd_list)

        # Save dataset
        self.save_dataset(norm_pcd_list, dataset_name, output_folder)

        # Display some of the preprocessed point clouds
        self.display_pcd_list(norm_pcd_list, 10, info=True)


if __name__ == '__main__':
    partitioner = PointCloudPartitioner()
    # partitioner.example_001_move_to_origin()
    # partitioner.example_002_partition()
    # partitioner.example_003_partition_with_boxes()
    # partitioner.example_004_gridding()
    # partitioner.example_005_sliding_window()
    # partitioner.example_006_crop_with_sliding_window()
    # partitioner.example_007_downsample_point_cloud()
    # partitioner.example_008_upsample_point_cloud()
    # partitioner.example_009_normalize_point_cloud()
    # partitioner.example_010_create_and_display_dataset()
    partitioner.example_011_create_and_save_dataset()
