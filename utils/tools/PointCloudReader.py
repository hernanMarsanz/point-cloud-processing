import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
import open3d as o3d
import datetime
import math
from PointCloudTransformer import PointCloudTransformer


class PointCloudReader:
    def __init__(self):
        pass
    

    @staticmethod
    def read_parquet_as_dask(filepath):
        df = dd.read_parquet(filepath)
        return df
    

    @staticmethod
    def read_parquet_as_pandas(filepath):
        df = dd.read_parquet(filepath)
        df = PointCloudTransformer.dask_to_pandas()
        return df
    

    @staticmethod
    def read_ply_as_pandas(filepath):
        pcd = o3d.io.read_point_cloud(filepath)
        df = PointCloudTransformer.open3d_to_pandas(pcd)
        return df


    @staticmethod
    def read_ply_as_numpy(filepath):
        pcd = o3d.io.read_point_cloud(filepath)
        np_array = PointCloudTransformer.open3d_to_numpy(pcd)
        return np_array


    @staticmethod
    def read_ply_as_o3d(filepath):
        pcd = o3d.io.read_point_cloud(filepath)
        return pcd

    
if __name__ == '__main__':
    transformer = PointCloudReader()