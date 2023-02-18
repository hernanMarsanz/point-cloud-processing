from PointCloudTransformer import PointCloudTransformer
from PointCloudWriter import PointCloudWriter
from PointCloudPartitioner import PointCloudPartitioner
import open3d as o3d
import numpy as np
import pandas as pd
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


class PointCloudClassifier:
    def __init__(self):
        ...


    def search_for_objects(self, pcd, partition_shape, grid_shape, search_areas_shape, steepness_threshold):
        """Takes a point cloud, desired grid shape, shape of the areas to shape, extent of pcds box
        and a steepness threshold. Returns a list of detection boxes."""

        pcd, box, points, extent, orig_translation_matrix = partitioner.move_to_origin(pcd_1)
        # print(f'shape is {extent[0]} meters long, {extent[1]} meters wide and {extent[2]} meters high.')

        partition_point_list = partitioner.partition(extent, partition_shape)

        box_list = partitioner.generate_boxes_from_points(partition_point_list)
        
        crop_list = partitioner.get_gridded_crop_list(pcd, box_list)


        partitioner.draw_object_lists([pcd])

        result_list = []
        for crop in crop_list:
            crop, box, points, extent, translation_matrix = partitioner.move_to_origin(crop)
            crop, data_dict, shape, gridded_model, grid_matrix = partitioner.grid(crop, extent, grid_shape)
            partitioner.draw_object_lists(gridded_model, [crop])
            max_z = max(data_dict['z_value'])
            start_index = data_dict['z_value'].index(max_z)
            clean_grid_model = copy.deepcopy(gridded_model)
            partition_list = partitioner.partition_gridded_model(shape,
                                                                search_areas_shape)
            area_coordinates = self.get_areas_to_evaluate(data_dict, partition_list)
            detection_box_list = []
            for coordinates in area_coordinates:
                temp_grid_model = copy.deepcopy(clean_grid_model)
                start_index = data_dict['coordinates'].index(coordinates)
                box_points, detection_box, result, temp_grid_model = self.detect_object(crop,
                                                                                        data_dict,
                                                                                        shape,
                                                                                        temp_grid_model,
                                                                                        start_index,
                                                                                        steepness_threshold)
                if detection_box:
                    detection_box = detection_box.translate((-translation_matrix[0],
                                                            -translation_matrix[1],
                                                            -translation_matrix[2]))
                    detection_box = detection_box.translate((-orig_translation_matrix[0],
                                                             -orig_translation_matrix[1],
                                                             -orig_translation_matrix[2]))
                    detection_box_list.append(detection_box)
            if detection_box_list:
                # detection_box_list = self.purge_duplicated_detections(detection_box_list,
                #                                                     iou_threshold=.5)
                result_list.extend(detection_box_list)
        pcd = pcd.translate((-orig_translation_matrix[0], -orig_translation_matrix[1], -orig_translation_matrix[2]))
        # result_list = 0
        return result_list



    def get_areas_to_evaluate(self, data_dict, partition_list):
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
        det_val = self.numerical_evaluator(dif_values, max_z)
        if det_val >= steepness:
            result = True
        else:
            result = False
            detection_box = None
        return box_points, detection_box, result, gridded_model


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


    def read_ply_as_o3d(self, filepath):
        """Takes a string filepath and returns a point cloud file."""
        pcd = o3d.io.read_point_cloud(filepath)
        return pcd


    def numerical_evaluator(self, dif_values, max_z):
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
    

    def read_csv_labels(self, filepath):
        df = pd.read_csv(filepath)
        label_array = df.to_numpy()
        print(label_array)
        return label_array


    # Examples:

    def example_001_read_labels(self):
        self.read_csv_labels('datasets/dataset_01/train/train_labels.csv')


if __name__ == '__main__':
    classifier = PointCloudClassifier()
    classifier.example_001_read_labels()


    # samples_list = []

    # for i, name in enumerate(glob.glob('samples/ply/*.ply')):
    #     samples_list.append(name)

    # classifier = PointCloudClassifier()
    # partitioner = PointCloudPartitioner()

    # # filepath_1 = 'samples/munition_test.ply'
    # filepath_1 = 'samples/ply/sample_003.ply'
    # # filepath_2 = 'samples/sample_3.ply'


    # pcd_1 = classifier.read_ply_as_o3d(filepath_1)


    # partition_shape = (5,3)
    # sub_partition_shape = (10,20)
    # search_areas_shape = (2,4)

    # # partitioner.draw_ob   ject_lists([pcd_1])

    # detection_box_list = classifier.search_for_objects(pcd_1, partition_shape, sub_partition_shape, search_areas_shape, steepness_threshold=.57)
    # if detection_box_list:
    #     partitioner.draw_object_lists(detection_box_list, [pcd_1])
    # else:
    #     partitioner.draw_object_lists([pcd_1])



