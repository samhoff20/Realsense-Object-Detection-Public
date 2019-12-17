
from edgetpu.detection.engine import DetectionEngine
from PIL import Image
import pyrealsense2 as rs
import numpy as np
import argparse
import imutils
import time 
import sys
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to Tensorflow Lite object detection model")
ap.add_argument("-l", "--labels", required=True, help="path to labels file")
ap.add_argument("-c", "--confidence", type=float, default=0.3, help = "Confidence threshold expressed as a decimal")
ap.add_argument("-w", "--width", type = int, default = 500, help = "Size of image as it's inputed into the model")
ap.add_argument("-i", "--info", type = int, default = 0, help = "Prints relevent runtime information in the console on startup")
args = vars(ap.parse_args())

labels={}

for row in open(args["labels"]):
    (classID, label) = row.strip().split(maxsplit=1)
    labels[int(classID)] = label.strip()

print("Tensorflow-Lite model loading...")
model = DetectionEngine(args["model"])

print("Initializing Realsense...")
pipeline = rs.pipeline()
config = rs.config()
depth_resolution_x =640
depth_resolution_y =480
color_resolution_x =640
color_resolution_y =480
depth_fps =30
color_fps =60
config.enable_stream(rs.stream.depth, depth_resolution_x, depth_resolution_y, rs.format.z16, depth_fps) 
config.enable_stream(rs.stream.color, color_resolution_x, color_resolution_y, rs.format.bgr8, color_fps)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

if args["info"] > 0:
    print("Depth Input: ",depth_resolution_x,"x",depth_resolution_y,"at",depth_fps,"fps")
    print("Color Input: ",color_resolution_x,"x",color_resolution_y,"at",color_fps,"fps")

#align_with = rs.stream.color
#align = rs.align(align_with)

start_time = time.time()
x = 1 # displays the frame rate every 1 second
counter = 0


while True:
    ##wait for a new frame and then get the depth and color frame 
    frames = pipeline.wait_for_frames() 
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue
    
    ##create numpy array of depth and color frames
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    ##resize image based upon argument and create a copy to annotate and display
    color_image = imutils.resize(color_image, width=args["width"])
    orig = color_image
    #color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    color_image = Image.fromarray(color_image)

    ##start a timer for inferencing time and feed the frame into the model  
    start_inference = time.time()
    results = model.DetectWithImage(color_image, threshold=args["confidence"],
        keep_aspect_ratio=True, relative_coord=False)
    end_inference = time.time()
    
    ##put a bounding box on the result in the copy image
    for r in results:
        bounding_box = r.bounding_box.flatten().astype("int") #try changing r
        (startX, startY, endX, endY) = bounding_box #try changing startx etc
        label = labels[r.label_id]
        cv2.rectangle(orig, (startX, startY), (endX, endY),
                (0, 255, 0), 2)
        
        ##add a dot to the center of the bounding box
        centerX = int((startX - ((startX - endX)*0.5)))
        centerY = int((startY - ((startY - endY)*0.5)))
        cv2.circle(orig, (centerX, centerY), 1, (0, 255, 0), 2 )

        #-------Single Point-----------
        ##calculates depth of the center point of the bounding box
        depth = depth_image[centerX, centerY].astype(float)
        depth = depth * depth_scale
        #------------------------------
        
        #-----------Average------------
        ##calculates depth of the center point of the bounding box
        #depth_center = depth_image[centerX, centerY].astype(float)
        #depth_collection = depth_center + depth_image[(centerX+1), (centerY+1)].astype(float) + depth_image[(centerX+1), (centerY-1)].astype(float) + depth_image[(centerX-1), (centerY-1)].astype(float) + depth_image[(centerX-1), (centerY+1)].astype(float)
        #depth_average = depth_collection/5
        #depth = depth_average * depth_scale
        #print(depth)
        #------------------------------

        y = startY - 15 if startY - 15 > 15 else startY + 15
        text = "{}: {:.2f}% {:.3f}".format(label, r.score*100, depth) 
        cv2.putText(orig, text, (startX,y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2) #try changing font


    ##create the window to display the result
    cv2.namedWindow('Real Sense Object Detection', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Real Sense Object Detection', orig)
    key = cv2.waitKey(1) & 0xFF

    if args["info"] > 0:
        ##create counter for measuring inference time and frames per second
        counter += 1
        if (time.time() - start_time) > x:
            print("FPS: ", counter / (time.time() - start_time))
            fps = []
            fps.append(counter / (time.time() - start_time))
            inference_time = []
            inference_time.append(end_inference - start_inference)
            counter = 0
            start_time = time.time()

    if key == ord("q") or key == 27:
        break

##calculate the average FPS and inference time    
if args["info"] > 0:
    fps_sum = 0
    for num in fps:
        fps_sum += num
        fps_average = fps_sum / len(fps)
    print("Average FPS:",fps_average)

    inference_sum = 0
    for num in inference_time:
        inference_sum += num
    inference_average = inference_sum / len(inference_time)
    print("Average Inference Time:",inference_average)   

          
pipeline.stop()
    
    

