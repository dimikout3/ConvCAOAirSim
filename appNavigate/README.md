# Navigate #

An app developed for providing a predetermined path to the UAVs. Used mainly on reproducing experiments for debuging purposes

# Specific App Instructions #

### 1. Dependencies
In order to run this application you will nee a number of pickle objects. Each pickle object corresponds to a UAV and it is a list which element is an airsim multirotor [state](https://microsoft.github.io/AirSim/apis/). The UAV will move to all the positions  described in the aforementioned multirotor state list sequentially.

### 2. Run
Run the script
```
$ python navigate.py
```
