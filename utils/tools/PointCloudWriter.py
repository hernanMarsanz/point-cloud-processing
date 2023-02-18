import os
import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
import datetime
import math
import open3d as o3d
from PointCloudTransformer import PointCloudTransformer
from PointCloudReader import PointCloudReader
import glob
from tkinter import filedialog as fd


class PointCloudWriter:
    def __init__(self):
        pass


    @staticmethod
    def save_parquet_as_ply(input_path, output_path): # needs to read the parquet filepath
        df = PointCloudReader.parquet_as_dask
        pcd = PointCloudTransformer.dask_to_open3d
        o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)


    @staticmethod
    def save_numpy_as_ply(np_array, output_path):
        pcd = PointCloudTransformer.numpy_to_open3d
        o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)


    @staticmethod
    def save_pandas_as_ply(df, output_path):
        pcd = PointCloudTransformer.pandas_to_open3d
        o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)


    @staticmethod
    def save_dask_as_ply(df, output_path):
        pcd = PointCloudTransformer.dask_to_open3d
        o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)


    @staticmethod
    def save_o3d_as_ply(pcd, output_path):
        o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)


    @staticmethod
    def save_csv_as_ply(filename, destination):
        ...


    @staticmethod
    def save_csv_as_parquet(filename, destination):
        ...


    @staticmethod
    def save_o3d_list_as_ply(object_list, output_path,  connector = '_', name_prefix='sample',
                             translation_mat=[0,0,0]):
        for i, object in enumerate(object_list):
            file_number = '0'*(4-len(str(i))) + str(i)
            file_name = ''.join([output_path, name_prefix, connector, file_number, '.ply'])
            o3d.io.write_point_cloud(file_name, object, write_ascii=True)


    def change_name_from_pcd_files(self, input_path, output_path, folder_name = 'dataset', name_prefix='sample', connector='_', file_extension='.ply'):
        open_path = ''.join([input_path,'/','*',file_extension])
        filename_list = [filename for filename in glob.glob(open_path)]
        output_folder_path = ''.join([output_path,'/',folder_name])
        os.mkdir(output_folder_path)
        for i,filename in enumerate(filename_list):
            file_number = '0'*(6-len(str(i))) + str(i)
            pcd = o3d.io.read_point_cloud(filename)
            new_filename = ''.join([output_folder_path,'/',folder_name, connector,name_prefix,connector,file_number,file_extension])
            o3d.io.write_point_cloud(new_filename, pcd, write_ascii=True)


    # Examples:


    def example_001_change_pcds_names(self):
            input_folder = fd.askdirectory()
            output_folder = fd.askdirectory()
            self.change_name_from_pcd_files(input_folder, output_folder, folder_name='dataset_01', name_prefix='sample')


if __name__ == '__main__':
    writer = PointCloudWriter()
    writer.example_001_change_pcds_names()
