# Autonomous and cooperative design of the monitor positions for a team of UAVs to maximize the quantity and quality of detected objects #

This project deals with the problem of positioning a swarm of UAVs inside a completely unknown terrain, having as objective to maximize the overall situational awareness.

Example:
![RA-L_mainFigure](http://kapoutsis.info/wp-content/uploads/2020/02/RA-L_mainFigure.png)

[![Video demonstration](http://kapoutsis.info/wp-content/uploads/2020/02/video_thumbnail.png)](https://www.youtube.com/watch?v=L8ycmS20rZs)

[AirSim platform](https://github.com/microsoft/AirSim) was utilized to evaluate the perfmance of the swarm. 

The implemented algorithm is not specifically tailored to the dynamics of either UAVs or the environment, instead, it learns, from the real-time images, exactly the most effective formations of the swarm for the underlying monitoring task. Moreover, and to be able to evaluate at each iteration the swarm formation, images from the UAVs are fed to a novel computation scheme that assigns a single scalar score, taking into consideration the number and quality of all unique objects of interest.

# Installation #

The ConvCAO_AirSim repository contains the following applications:
- MultiAgent: Positioning a swarm of UAVs inside a completely unknown terrain, having as objective to maximize the overall situational awareness.
- appExhaustiveSearch: Centralized, semi-exhaustive methodology.
- appHoldLine: A rather simple problem (toy-problem), where the robots should be deployed in a specific formation (line).
- appLinux: Implementation for working on Linux OS.
- appNavigate: Navigating a UAV swarm on a predetermined path.

This section provides a step-by-step guide for installing the ConvCAO_AirSim framework.

### Dependencies

First, install the required system packages 
(NOTE: the majority of the experiments we have concluded are done in a conda enviroment, therefore we stongly advise you to download and install a conda virtual enviroment):
```
$ pip install airsim Shapely descartes opencv-contrib-python=4.1.26
```

### Detector
In our experiments we are using a YOLOv3 detector, trained on the [COCO dataset](http://cocodataset.org/#home). However, you can utilize a different detector (tailored to the application needs) and our methodology will still be capable of delivering an optimized set of UAVs’ monitor positions, adapting to the detector’s specific characteristics. [Download](https://convcao.hopto.org/index.php/s/mh8WIDpprE70SO3) the pretrained detector we are using and copy the yolo-coco folder inside your ConvCAO_AirSim path.

### Enviroments
Download any of the available [AirSim Enviroments](https://github.com/microsoft/AirSim/releases)

### Run Example
Lastly, you can have an illustrative example by running the "MultiAgent.py" script in the ConvCAO_AirSim folder, simply add the path "detector-path":"path-to-your-detector-folder" to the detector folder in the "appSettings.json". Detailed Instructions for running specific applications are inside every corresponding app folder
```
$ python MultiAgent.py
```


# 3D Reconstruction #
Combining the information extracted from the Depth Image and the focal length of the camera we can recreate the 3D percepective for each UAV
<p align="center">
  <img width="712" height="400" src="toGiF.gif">
</p>

# Combined 3D Reconstruction #
Combining the aforementioned 3D reconstruction of each UAV we can generate the a point cloud for the whole enviroment 
<p align="center">
  <img width="712" height="400" src="combined.gif">
</p>

