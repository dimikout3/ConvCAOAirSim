import sys
import os
import numpy as np
import pickle
import open3d as o3d
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

"""Parses all the position and reconstructs 3D model of the full(!) depth map"""


def capture_depth(vis):
    depth = vis.capture_depth_float_buffer()
    plt.imshow(np.asarray(depth))
    plt.show()
    return False

def capture_image(vis):
    image = vis.capture_screen_float_buffer()
    plt.imshow(np.asarray(image))
    # plt.show()
    plt.savefig("test.png")
    return False

def custom_draw_geometry(pcd, position_dir):
    # The following code achieves the same effect as:
    # o3d.visualization.draw_geometries([pcd])
    # http://www.open3d.org/docs/release/tutorial/Advanced/customized_visualization.html#customized-visualization

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name = 'open3d',width = 1000, height = 1000)
    vis.add_geometry(pcd)

    # Rendering
    # http://www.open3d.org/docs/release/python_api/open3d.visualization.RenderOption.html
    # vis.get_render_option().load_from_json("RenderOption_2020-02-26-10-53-30.json")
    renderer = vis.get_render_option()
    renderer.point_size = 1
    # renderer.background_color = np.asarray([0, 0, 0])

    # View Control
    # http://www.open3d.org/docs/release/python_api/open3d.visualization.ViewControl.html
    ctr = vis.get_view_control()
    # View Control from PinHole Camera
    # http://ftp.cs.toronto.edu/pub/psala/VM/camera-parameters.pdf
    param = ctr.convert_to_pinhole_camera_parameters()
    # print(f"param={param.extrinsic}")
    param.extrinsic = np.array([[ -0.63849157945929502, -0.76679317172116246, -0.06600556613934061, 50.221199759940745],
                                [ 0.43757735236884049, -0.43223326149844726, 0.78847984650737302, 32.636292824133406],
                                [-0.63313076347106256, 0.47455520169545601, 0.61150862372523795, 128.65327749719455],
                                [ 0., 0. ,0., 1.]])
    ctr.convert_from_pinhole_camera_parameters(param)
    # Read camera params
    # https://github.com/intel-isl/Open3D/issues/1110
    # param = o3d.io.read_pinhole_camera_parameters('cameraparams.json')
    # ctr = vis.get_view_control()
    # ctr.convert_from_pinhole_camera_parameters(param)

    # vis.update_geometry()
    # vis.poll_events()
    # vis.update_renderer()
    vis.run()
    # capture_image(vis)
    pcdPNG_dir = os.path.join(position_dir, "pointCloud.png")
    vis.capture_screen_image(pcdPNG_dir)
    vis.destroy_window()


def plot3D(position_dir):
    pcd_dir = os.path.join(position_dir, "pointCloud.asc")
    pcd = o3d.io.read_point_cloud(pcd_dir, format='xyzrgb')
    # o3d.visualization.draw_geometries([pcd]) # Visualize the point cloud
    custom_draw_geometry(pcd, position_dir)


if __name__ == "__main__":

    simulation_dir = os.path.join(os.getcwd() ,"results_1_legacy")

    parent_dir = os.path.join(simulation_dir, "swarm_raw_output")
    detected_dir = os.path.join(simulation_dir, "swarm_detected")

    dronesID = os.listdir(parent_dir)
    dronesID = [drone for drone in dronesID if drone!="GlobalHawk"]
    wayPointsID = os.listdir(os.path.join(detected_dir, dronesID[0]))

    for drone in dronesID:

        print(f"=== Woriking on {drone} ===")

        camera_dir = os.path.join(simulation_dir, "swarm_raw_output",f"{drone}",f"state_{drone}.pickle")
        state = pickle.load(open(camera_dir,"rb"))

        for posIndex, position in enumerate(tqdm(wayPointsID)):

            position_dir = os.path.join(simulation_dir, "swarm_raw_output",f"{drone}", f"{position}")

            plot3D(position_dir)
