# demo

Quik Start
=

1.Check all the dependencies installed
-
pip install -r requirements.txt

2.Clone this repository
-
    cd detector/YOLOv3/weight/
    wget https://pjreddie.com/media/files/yolov3.weights
    wget https://pjreddie.com/media/files/yolov3-tiny.weights
    cd ../../../

3.Download YOLOv3 parameters
-
    cd deep_sort/deep/checkpoint
# download ckpt.t7 from
    https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6 to this folder
    cd ../../../

4.Download deepsort parameters ckpt.t7
-
    cd deep_sort/deep/checkpoint
    # download ckpt.t7 from
    https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6 to this folder
    cd ../../../

5.Compile nms module
-
    cd detector/YOLOv3/nms
    sh build.sh
    cd ../../..
    
6.Create Face recognition database(db.npz)
-
    pip install face_recognition
    usage: face_database.py(dont't forget to change face database path in the python file)
    
7.Run demo
-
    usage: python yolov3_deepsort.py RGB_VIDEO_PATH DEPTH_VIDEO_PATH
                                [--help]
                                [--frame_interval FRAME_INTERVAL]
                                [--config_detection CONFIG_DETECTION]
                                [--config_deepsort CONFIG_DEEPSORT]
                                [--display]
                                [--display_width DISPLAY_WIDTH]
                                [--display_height DISPLAY_HEIGHT]
                                [--save_path SAVE_PATH]          
                                [--cpu]      

