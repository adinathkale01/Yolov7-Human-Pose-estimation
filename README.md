# Yolov7-Human-Pose-estimation
In this project,Implemented Yolov7-pose for Human pose estimation


# setup to run code on windows 11
1.create virtual environment in anaconda prompt using following command 
  * conda create -n yolov7_custom python=3.9
2.To activate environment 
  * conda activate yolov7_custom 
3. clone the repository
  * 
4. Go to cloned folder
  * cd yolov7-pose-estimation
5. Install pakages using following command
  * pip install -r requirements.txt
6.Download Yolov7 pose estimation weights from official github and put it inside current working directory 
  * https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt
 7.For pose estimation on video/webcam use pose-estimate.py fileo and to execute this file use following command.
  #For CPU
  * python pose-estimate.py --source "your custom video.mp4" --device cpu
 8. For Pushup_counting use pushup_counter.py file and to execute this file use following command
  #For CPU
  * python pushup_counter.py --source "pushup.mp4" --device 0 --curltracker=True
# References 
* https://github.com/WongKinYiu/yolov7.git
* https://github.com/RizwanMunawar/yolov7-pose-estimation.git
* YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss: https://arxiv.org/abs/2204.06806
* https://learnopencv.com/tag/yolov7-pose/
 
