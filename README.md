# FrameSkipper

YOLOv7 based Vehicle Detection Accelerator

Overview

FrameSkipper is a cutting-edge project designed to accelerate video processing by implementing an advanced deep neural network model. Our solution focuses on improving video querying efficiency by intelligently skipping redundant frames in videos, which significantly reduces evaluation costs and inference time.


Problem Statement

In the realm of video processing, the challenge lies in managing extensive computational resources and time. Our project addresses this by optimizing the video analysis process, specifically targeting vehicle detection in video streams.


Project Description

FrameSkipper utilizes a custom dataset and YOLOv7 framework to detect vehicles in video frames. By classifying frames as critical or non-critical, our system retains only the essential frames, thus speeding up the video analysis process.

DEMO:

Original video:

![original_video](https://github.com/JackyTang0516/FrameSkipper/assets/111934442/9b001f40-3d82-4175-af40-cd895cc5e9e3)

Accelerated video:

![accelerated_video](https://github.com/JackyTang0516/FrameSkipper/assets/111934442/9d2ef4a8-2a47-4fdb-8f3b-fbe096bdbdab)

Trained Model: [https://drive.google.com/file/d/181tTcNFwHdHQXgzxvTrqBOGToPmf18Tn/view?usp=sharing
](https://drive.google.com/file/d/1MXDrJ5GPsyCM_mthDItrUNmaYjZH3DxE/view?usp=sharing)

Slides: [https://docs.google.com/presentation/d/19Lre-3UEnAXudEL7ISnRFfmfQxd2tobz7RiuIZ0zxWY/edit?usp=sharing](https://docs.google.com/presentation/d/19Lre-3UEnAXudEL7ISnRFfmfQxd2tobz7RiuIZ0zxWY/edit#slide=id.g21efff247e5_0_12)

References: 

Papers: https://ieeexplore.ieee.org/document/10129395

Codes: https://github.com/WongKinYiu/yolov7

Datasets: https://detrac-db.rit.albany.edu/

Core Components:

1. yolov7_train.py: This Python script is responsible for training the YOLOv7 model. It likely includes code to load the dataset, define the neural network architecture, specify the loss function and optimizer, and iterate over the dataset in epochs to train the model for object detection tasks.

2. frame_skipper.py: This script probably contains the algorithm for the FrameSkipper tool. It would analyze video frames and determine which ones are redundant and can be skipped without losing important information, thus speeding up the video processing. The script may employ methods to detect minimal changes between frames and apply a threshold for deciding when to skip a frame.

3. yolov7_detect.py: After the model is trained, this Python script is used for the detection phase. It would use the trained YOLOv7 model to process new images or video frames and identify and label the objects (likely vehicles) detected in each frame.

4. UA_DETRAC_onecls.yaml: This YAML file serves as a configuration file, which would include parameters for training the YOLOv7 model. This includes settings such as the path to the dataset, hyperparameters for the training, validate and test process, the number of classes, and other relevant configurations that are necessary to train the model on the UA-DETRAC dataset for one class, which in the context of this project, is probably vehicles.

