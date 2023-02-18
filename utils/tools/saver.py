import open3d as o3d

pcd_full = o3d.io.read_point_cloud("samples/sample_000.ply")
pcd_full.paint_uniform_color([1, 1, 1])

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd_full)
vis.update_geometry(pcd_full)
vis.poll_events()
vis.update_renderer()
vis.capture_screen_image("2.jpg", True)
vis.destroy_window()