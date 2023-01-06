import cv2
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt,strip_optimizer,xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt


from trainer import findAngle
from PIL import ImageFont,ImageDraw,Image

@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt",source="fall1.mp4",device='cpu',curltracker=False,drawskeleton=False):
    path = source
    ext = path.split('/')[-1].split('.')[-1].strip().lower()
    if ext in ["mp4","webm","avi"] or ext not in ["mp4","webm","avi"] and ext.isnumeric():
        
        input_path = int(path) if path.isnumeric() else path
    
        device = select_device(opt.device) #select device
        half = device.type != 'cpu'

        model = attempt_load(poseweights, map_location=device)  #Load model
        _ = model.eval()
        #names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    
        
        cap = cv2.VideoCapture(input_path)   #input_path #pass video to videocapture object
        webcam = False
        
        if (cap.isOpened() == False):
            print('Error while trying to read video. Please check path again')
        
        fw,fh = int(cap.get(3)),int(cap.get(4))  #get video frame width
        if ext.isnumeric():
            webcam =True
            fw,fh = 1280,768
        
        vid_write_image = letterbox(cap.read()[1], (fw), stride=64, auto=True)[0] #init videowriter
        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = "output" if path.isnumeric() else f"{input_path.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f"{source}_keypoint.mp4",cv2.VideoWriter_fourcc(*'mp4v'), 30,(resize_width, resize_height))
        if webcam:
            out = cv2.VideoWriter(f"{out_video_name}_kpt.mp4",cv2.VideoWriter_fourcc(*'mp4v'),30,(fw,fh))

        frame_count, total_fps = 0,0
        #2.0 variables count of pushup
        push_ups = 0
        direction = 0
        bar = 0
     
 
        Percentage = 0 
        #2.2 load custom font
        fontpath = "futur.ttf"
        font = ImageFont.truetype(fontpath,32)
        
        font1 = ImageFont.truetype(fontpath,160)


        while(cap.isOpened):
            print("Frame {} Processing".format(frame_count+1))

            ret, frame = cap.read()  #get frame and success from video capture
            
            if ret: #if success is true, means frame exist
                orig_image = frame #store frame
                #preprocess
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB) #convert frame to RGB
                
                if webcam:
                    image = cv2.cvtColor(image,(fw,fh),interpolation = cv2.INTER_LINEAR)
                    
                
                image = letterbox(image, (fw), stride=64, auto=True)[0]
                image_ = image.copy()
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
            
                image = image.to(device)  #convert image data to device
                image = image.float() #convert image to float precision (cpu)
                start_time = time.time() #start time for fps calculation
            
                with torch.no_grad():
                    output_data, _ = model(image)

                output_data = non_max_suppression_kpt(output_data,0.5,0.65,nc=model.yaml['nc'],nkpt=model.yaml['nkpt'],kpt_label=True)    
                                                                     
            
                output = output_to_keypoint(output_data)

                img = image[0].permute(1, 2, 0) * 255 # Change format [b, c, h, w] to [h, w, c] for displaying the image.
                img = img.cpu().numpy().astype(np.uint8)
                
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #reshape image format to (BGR)
                #gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                
                #4.0 pushup tracking and counting
                if curltracker:
                    
                    color = (128,0,0)#color = (254,118,136)
                    for idx in range(output.shape[0]):
                        kpts = output[idx,7:].T
                        #right arm = (5,7,9) , left arm = (6,8,10)
                        
                        angleR = findAngle(img,kpts,5,7,9,draw = True)
                        angleL = findAngle(img,kpts,6,8,10,draw =True)
                     
                        Percentage = np.interp(angleR,(210,280),(0,100))
                        
                 
                        bar = np.interp(angleR,(220,280),(int(fh)-100,100))
                        
                       
                 
                  
                        #check for pushup press
                        if direction == 0:
                            if Percentage == 100:
                                
                                push_ups += 0.5
                                
                                direction = 1
                            
                                
                        if direction == 1:
                            if Percentage == 0:
                               
                                push_ups += 0.5
                               
                                direction = 0
                            
                        #bar
                        cv2.line(img,(100,100),(100,int(fh)-100),(128,128,128),30)
                        cv2.line(img,(100,int(bar)),(100, int(fh)-100),color,30)
                        
                        if (int(Percentage) < 10):
                            cv2.line(img,(155,int(bar)),(190,int(bar)),color,40)
                        elif ((int(Percentage) >= 10) and (int(Percentage) < 100)):
                            cv2.line(img,(155,int(bar)),(200,int(bar)),color,40)
                        else:
                            cv2.line(img,(155,int(bar)),(210,int(bar)),color,40)
                      
                    im = Image.fromarray(img)
                    draw = ImageDraw.Draw(im)
                    #draw.rounded_rectangle((fw-300,(fh//2)+100 , fw-50,(fh//2)+100),fill = color,radius = 40)
                    
                    draw.text((145,int(bar)-17),f"{int(Percentage)}%",font=font, fill= (255,255,255))
                        
                    draw.text((fw-200,(fh//2)-250),f"{int(push_ups)}",font=font1, fill= (128,0,0))
                    
                    
                        
                    img = np.array(im)
                    #cv2.imshow("Pushup Counter", image)
                    #cv2.waitKey(1)


                    
                        
                #draw skeleton       
                #if drawskeleton:
                    #for idx in range(output.shape[0]):
                        #plot_skeleton_kpts(img,output[idx,7:].T,3)
                  
                #display image
                if webcam:
                    cv2.imshow('detection',img)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                else:
                    img_ = img.copy()
                    img_= cv2.resize(img_,(960,540),interpolation = cv2.INTER_LINEAR)
                    cv2.imshow('detection',img_)
                    cv2.waitKey(1)
                  
                 
                end_time = time.time()
                fps = 1 / (end_time-start_time)
                total_fps += fps
                frame_count +=1
                out.write(img)
       
            else:
                break
                        
        cap.release()
        cv2.destroyAllWindows()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")
        
    #plot the comparision graph
    #plot_fps_time_comparision(time_list=time_list,fps_list=fps_list)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, help='video/0 for webcam') #video source
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')   #device arugments
    parser.add_argument('--curltracker', type=bool, default='True', help='set as true to check count bicep curls')   #curltracker 
    parser.add_argument('--drawskeleton', type=bool, default='False', help='set as True to draw skeleton')   #curltracker
    

    #parser.add_argument('--view-img', action='store_true', help='display results')  #display results
    #parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') #save confidence in txt writing
    #parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)') #box linethickness
    #parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels') #box hidelabel
    #parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences') #boxhideconf
    opt = parser.parse_args()
    return opt

#function for plot fps and time comparision graph
def plot_fps_time_comparision(time_list,fps_list):
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('FPS and Time Comparision Graph')
    plt.plot(time_list, fps_list,'b',label="FPS & Time")
    plt.savefig("FPS_and_Time_Comparision_pose_estimate.png")
    

#main function
def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device,opt.poseweights)
    main(opt)

                    
                        
                        
                        
                         
                             
                         
                        
                        
                        
                        
                                                     

                        
                                                     

