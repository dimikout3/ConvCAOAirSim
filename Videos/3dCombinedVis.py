import sys
import os
import numpy as np
import pickle
import open3d as o3d
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import airsim

from scipy.spatial.transform import Rotation as R

import optparse

"""Parses all the pointCloud.asc and generates a png image"""


def get_options():

    optParser = optparse.OptionParser()
    optParser.add_option("--plot",action="store_true", default=False, dest="plot", help="Ploting every single point cloud")
    optParser.add_option("--NoSave",action="store_true", default=False, dest="NoSave", help="Do not save the output *.png")
    optParser.add_option("--quit", default=500, dest="quit", type="int", help="Quit after specified number")
    options, args = optParser.parse_args()

    return options


def getParamExtrinsicConst():

    # http://ksimek.github.io/2012/08/22/extrinsic/

    constX = -300
    constY = -25
    constZ = -300

    # rotation = R.from_euler('xyz', [90, 0, 180], degrees=True)
    rotationX = 90 - 40
    rotationY = 0
    rotationZ = 90

    rotation = R.from_euler('xyz', [rotationX, rotationY, rotationZ], degrees=True)
    Rc = rotation.as_dcm()
    C = np.array([constX, constY, constZ])

    t = np.dot(-Rc.T,C)

    # Note that Rc here should be Transpose, but I can not figure out how to do it
    # in numpy ... (fail me ...)
    params = np.vstack((Rc,[t])).T

    # to have a square matrix
    params = np.vstack((params,[[0,0,0,1]]))

    return params


def custom_draw_geometry(pcd_list):

    global options, combinedPCD_folder, posIndex

    # The following code achieves the same effect as:
    # o3d.visualization.draw_geometries([pcd])
    # http://www.open3d.org/docs/release/tutorial/Advanced/customized_visualization.html#customized-visualization

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name = 'open3d',width = 1000, height = 1000)

    for pcd in pcd_list:
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
    # param.extrinsic = np.array([[ -0.63849157945929502, -0.76679317172116246, -0.06600556613934061, 50.221199759940745],
    #                             [ 0.43757735236884049, -0.43223326149844726, 0.78847984650737302, 32.636292824133406],
    #                             [-0.63313076347106256, 0.47455520169545601, 0.61150862372523795, 128.65327749719455],
    #                             [ 0., 0. ,0., 1.]])
    # param.extrinsic = getParamExtrinsic()
    param.extrinsic = getParamExtrinsicConst()
    ctr.convert_from_pinhole_camera_parameters(param)
    # Read camera params
    # https://github.com/intel-isl/Open3D/issues/1110
    # param = o3d.io.read_pinhole_camera_parameters('cameraparams.json')
    # ctr = vis.get_view_control()
    # ctr.convert_from_pinhole_camera_parameters(param)

    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()

    if options.plot:
        vis.run()
    # capture_image(vis)
    if not options.NoSave:
        pcdPNG_dir = os.path.join(combinedPCD_folder, f"CombinedPCD_{posIndex}.png")
        vis.capture_screen_image(pcdPNG_dir)

    vis.destroy_window()


def plot3D(pcd_list):

    custom_draw_geometry(pcd_list)


if __name__ == "__main__":

    global options, state, posIndex, combinedPCD_folder
    options = get_options()

    simulation_dir = os.path.join(os.getcwd(), "..","results_pointCloud")

    combinedPCD_folder = os.path.join(simulation_dir, f"combinedPCD")
    try:
        os.makedirs(combinedPCD_folder)
    except OSError:
        if not os.path.isdir(combinedPCD_folder):
            raise

    parent_dir = os.path.join(simulation_dir, "swarm_raw_output")
    detected_dir = os.path.join(simulation_dir, "swarm_detected")

    dronesID = os.listdir(parent_dir)
    dronesID = [drone for drone in dronesID if drone!="GlobalHawk"]
    wayPointsID = os.listdir(os.path.join(detected_dir, dronesID[0]))

    for posIndex in tqdm(range(len(wayPointsID))):

        pcd_list = []

        for drone in dronesID:

            position_dir = os.path.join(simulation_dir, "swarm_raw_output",f"{drone}", f"position_{posIndex}")

            pcd_dir = os.path.join(position_dir, "pointCloud.asc")
            pcd = o3d.io.read_point_cloud(pcd_dir, format='xyzrgb')

            pcd_list.append(pcd)

        plot3D(pcd_list)

        if posIndex >= options.quit:
            quit()
