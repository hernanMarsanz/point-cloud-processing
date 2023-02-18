import trimesh
import open3d as o3d
import copy
import numpy as np
import time
import keyboard
from pynput import keyboard

mesh = o3d.io.read_triangle_mesh("mesh_test.ply")
mesh2 = trimesh.load("mesh_test.ply")
def on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    print('{0} released'.format(
        key))
    if key == keyboard.Key.esc:
        # Stop listener
        return False

# Collect events until released
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()

# ...or, in a non-blocking fashion:
listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(mesh)
vis.poll_events()
vis.update_renderer()
time.sleep(3)
vis.destroy_window()