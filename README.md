# Autonomous and cooperative design of the monitor positions for a team of UAVs to maximize the quantity and quality of detected objects #

This project deals with the problem of positioning a swarm of UAVs inside a completely unknown terrain, having as objective to maximize the overall situational awareness.

Example:
![RA-L_mainFigure](http://kapoutsis.info/wp-content/uploads/2020/02/RA-L_mainFigure.png)

[![Video demonstration](http://kapoutsis.info/wp-content/uploads/2020/02/video_thumbnail.png)](https://www.youtube.com/watch?v=L8ycmS20rZs)

[AirSim platform](https://github.com/microsoft/AirSim) was utilized to evaluate the perfmance of the swarm. 

The implemented algorithm is not specifically tailored to the dynamics of either UAVs or the environment, instead, it learns, from the real-time images, exactly the most effective formations of the swarm for the underlying monitoring task. Moreover, and to be able to evaluate at each iteration the swarm formation, images from the UAVs are fed to a novel computation scheme that assigns a single scalar score, taking into consideration the number and quality of all unique objects of interest.

# 3D Reconstruction #
Combining the information extracted from the Depth Image and the focal length of the camera we can recreate the 3D percepective for each UAV
<p align="center">
  <img width="460" height="300" src="toGiF.gif">
</p>

# Combined 3D Reconstruction #
Combining the aforementioned 3D reconstruction of each UAV we can generate the a point cloud for the whole enviroment 
![](combinedPCD.gif)

