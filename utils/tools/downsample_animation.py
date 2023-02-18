
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

        # Downsample to 5000 points
        pcd_5000 = self.downsample_point_cloud(pcd, 5000)
        down_point_no = len(pcd_5000.points)


        # Create list of intermediate point quantities
        val_list = np.arange(100,5000,100).tolist()
        val_list.reverse()

        print(val_list)

        # Get downsampled pcds
        down_pcd_list = []
        for value in val_list:
            temp_pcd = copy.deepcopy(pcd_5000)
            temp_pcd = self.downsample_point_cloud(temp_pcd,value)
            down_pcd_list.append(temp_pcd)

        # Save downsampled pcds
        PointCloudWriter.save_o3d_list_as_ply(down_pcd_list,'samples/other/downsampling/','_','pcd')

        # Load downsampled pcds
        pcd_filename_list = [filename for filename in glob.glob('samples/other/downsampling/*.ply')]
        pcd_list = []
        for filename in pcd_filename_list:
            temp_pcd = o3d.io.read_point_cloud(filename)
            pcd_list.append(temp_pcd)


        print(len(pcd_list))
        
        # Show animation
        visualizer = PointCloudVisualizer()
        visualizer.display_animation(down_pcd_list,pose_path='pose.json',time_step=.1,save=True)

        # Draw original and downsampled models
        # self.draw_object_lists([pcd])
        # self.draw_object_lists([pcd_down])