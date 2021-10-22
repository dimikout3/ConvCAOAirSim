# Autonomous and cooperative design of the monitor positions for a team of UAVs to maximize the quantity and quality of detected objects #

This project deals with the problem of positioning a swarm of UAVs inside a completely unknown terrain, having as objective to maximize the overall situational awareness.

For full description and performance analysis, please check out our companion paper [MarsExplorer: Exploration of Unknown Terrains via Deep Reinforcement Learning and Procedurally Generated Environments](https://arxiv.org/abs/2007.01247)

Example:
![RA-L_mainFigure](http://kapoutsis.info/wp-content/uploads/2020/02/RA-L_mainFigure.png)

[![Video demonstration](http://kapoutsis.info/wp-content/uploads/2020/02/video_thumbnail.png)](https://www.youtube.com/watch?v=L8ycmS20rZs)

[AirSim platform](https://github.com/microsoft/AirSim) was utilized to evaluate the perfmance of the swarm.

The implemented algorithm is not specifically tailored to the dynamics of either UAVs or the environment, instead, it learns, from the real-time images, exactly the most effective formations of the swarm for the underlying monitoring task. Moreover, and to be able to evaluate at each iteration the swarm formation, images from the UAVs are fed to a novel computation scheme that assigns a single scalar score, taking into consideration the number and quality of all unique objects of interest.

The following video is a presentation of our work

[![](http://img.youtube.com/vi/C53Gsex-op8/0.jpg)](http://www.youtube.com/watch?v=C53Gsex-op8 "")

# Installation #

The ConvCAO_AirSim repository contains the following applications:
- [appConvergence](https://github.com/dimikout3/ConvCAO_AirSim/tree/master/appConvergence): Positioning a swarm of UAVs inside a completely unknown terrain, having as objective to maximize the overall situational awareness.
- [appExhaustiveSearch](https://github.com/dimikout3/ConvCAO_AirSim/tree/master/appExhaustiveSearch): Centralized, semi-exhaustive methodology.
- [appLinux](https://github.com/dimikout3/ConvCAO_AirSim/tree/master/appLinux): Implementation for working on Linux OS.
- [appNavigate](https://github.com/dimikout3/ConvCAO_AirSim/tree/master/appNavigate): Navigating a UAV swarm on a predetermined path.

This section provides a step-by-step guide for installing the ConvCAO_AirSim framework.

### 1. Dependencies
First, install the required system packages
(NOTE: the majority of the experiments were conducted in a conda enviroment, therefore we stongly advise you to download and [install](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) a conda virtual enviroment):
```
$ pip install airsim Shapely descartes opencv-contrib-python=4.1.26
```

### 2. Detector
Second, you have to define a detector capable of producing bounding boxes of objects along with the corresponding confidences levels from RGB images.

For the needs of our application we utilized YOLOv3 detector, trained on the [COCO dataset](http://cocodataset.org/#home). You can download this detector from [here](https://convcao.hopto.org/index.php/s/mh8WIDpprE70SO3). After downloading the file, extract the yolo-coco folder inside your local ConvCao_AirSim folder.

It is worth highlighting that, you could use a deifferent detector (tailored to the application needs), as the proposed methodology is agnostic as far the detector's choise is concerned.

### 3. Enviroments
Download any of the available [AirSim Enviroments](https://github.com/microsoft/AirSim/releases)

### 4. Run Example
To run an example with the Convergence testbed you need to just replace the "detector-path" entry - inside this [file](https://github.com/dimikout3/ConvCAO_AirSim/blob/master/appConvergence/appSettings.json) - with your path to the previsously downloaded detector.

Finally run the "MultiAgent.py" script:
```
$ python MultiAgent.py
```
Detailed instructions for running specific applications are inside every corresponding app folder


# 3D Reconstruction #
Combining the information extracted from the Depth Image and the focal length of the camera we can recreate the 3D percepective for each UAV
<p align="center">
  <img width="712" height="400" src="Videos/toGiF.gif">
</p>

# Combined 3D Reconstruction #
Combining the aforementioned 3D reconstruction of each UAV we can generate the a point cloud for the whole enviroment
<p align="center">
  <img width="712" height="400" src="Videos/combined.gif">
</p>

# Cite as: #

```
@article{koutras2020autonomous,
  title={Autonomous and cooperative design of the monitor positions for a team of UAVs to maximize the quantity and quality of detected objects},
  author={Koutras, Dimitrios I and Kapoutsis, Athanasios Ch and Kosmatopoulos, Elias B},
  journal={IEEE Robotics and Automation Letters},
  volume={5},
  number={3},
  pages={4986--4993},
  year={2020},
  publisher={IEEE}
}
```
