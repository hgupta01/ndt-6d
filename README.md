# ndt-6d
Implementation of NDT-6D method for registration of color point clouds

dependencies: 
    1. OpenCV >=3
    2. PCL
    3. Flann
    4. Ceres 
    5. Eigen 3.3.4 (compatible with Ceres old and new)
    6. Yaml-cpp

Steps to running the code:
    1. create build director inside ndt-6d folder (mkdir build && cd build)
    2. run 'cmake ..'
    3. run 'make' to create the executable
    4. set the path to images in config.yaml
    5. run './main' to visualize the registration