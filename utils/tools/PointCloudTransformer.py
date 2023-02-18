import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
import open3d as o3d
import datetime
import math


class PointCloudTransformer:
    def __init__(self):
        pass


    @staticmethod
    def pandas_to_numpy(df):
        np_array = df.to_numpy()
        return np_array


    @staticmethod
    def numpy_to_pandas(np_array):
        df = pd.DataFrame(np_array)
        return df


    @staticmethod
    def dask_to_pandas(df):
        df = df.compute()
        return df


    @staticmethod
    def numpy_to_open3d(np_array):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_array)
        return pcd


    @staticmethod
    def open3d_to_numpy(pcd):
        np_array = np.asarray(pcd.points)
        return np_array


    @staticmethod
    def open3d_to_pandas(pcd):
        np_array = PointCloudTransformer.open3d_to_numpy(pcd)
        df = PointCloudTransformer.numpy_to_pandas(np_array)
        return df


    @staticmethod
    def pandas_to_open3d(df):
        np_array = PointCloudTransformer.pandas_to_numpy(df)
        pcd = PointCloudTransformer.numpy_to_open3d(np_array)
        return pcd


    @staticmethod
    def dask_to_open3d(df):
        df = PointCloudTransformer.dask_to_pandas(df)
        pcd = PointCloudTransformer.pandas_to_open3d(df)
        return pcd


if __name__ == '__main__':
    transformer = PointCloudTransformer()

