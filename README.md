# Lidar_coloration
## Initial Setup:
This repo currently contains the starter files.

Clone repo and create a virtual environment
```
$ git clone https://github.com/yahyahajlaoui/Lidar_coloration
$ python3 -m venv venv
$ .\venv\Scripts\activate
```
Install requirements
```
$ (venv) pip install -r requirements.txt
```
run the script all ( this will take img, lidar and calibration files and save the colored pcd file in results/pcd )
!!! the image, lidar and calibration should be in the img lidar and calib folder
img format : .png
lidar format : .bin
calibration format : same as kitti dataset.

you can check the first example : '000000'

```
$ (venv) python coloration.py
```
