import math
import cv2

#3.0 take 3 points and return the angle between them
def findAngle(image,kpts,p1,p2,p3,draw=True):
    coord = []
    no_kpt = len(kpts)//3
    for i in range (no_kpt):
        cx,cy = kpts[3*i] , kpts[3*i +1]
        conf = kpts[3*i +2]
        coord.append([i,cx,cy,conf])
    points = (p1,p2,p3)
    
    #get landmarks
    x1,y1 = coord[p1][1:3]
    x2,y2 = coord[p2][1:3]
    x3,y3 = coord[p3][1:3]
    
    #claculate the angle
    angle = math.degrees(math.atan2(y3-y2,x3-x2) - math.atan2(y1-y2, x1-x2))
    if angle < 0:
        angle += 360
    # draw coordinates
    if draw:
        cv2.line(image,(int(x1),int(y1)),(int(x2),int(y2)),(50,240,250),3)
        cv2.line(image,(int(x3),int(y3)),(int(x2),int(y2)),(50,240,250),3)
        
        cv2.circle(image,(int(x1),int(y1)),10,(50,240,250),cv2.FILLED)
        cv2.circle(image,(int(x1),int(y1)),20,(50,240,250),4)
        cv2.circle(image,(int(x2),int(y2)),10,(50,240,250),cv2.FILLED)
        cv2.circle(image,(int(x2),int(y2)),20,(50,240,250),4)
        cv2.circle(image,(int(x3),int(y3)),10,(50,240,250),cv2.FILLED)
        cv2.circle(image,(int(x3),int(y3)),20,(50,240,250),4)
        
    return int(angle)
        
    
        
    
        
    
    