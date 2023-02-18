import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
import datetime
import math


class PointCloudProcessor:
    def __init__(self):
        self.convert = self.Converter()

    class Converter:
        def parquet_to_csv(input, output):
            print(f'Converting parquet from input path {input} to csv file. Saving in destination path {output}.')


if __name__ == '__main__':
    processor = PointCloudProcessor()
    processor.convert.parquet_to_csv("inputFiles/", "outputFiles/")
