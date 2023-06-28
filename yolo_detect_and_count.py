

import numpy as np
import random
import os
import cv2
import sort

import time

# Commented out IPython magic to ensure Python compatibility.
# %pip install ultralytics
import ultralytics
ultralytics.checks()
from ultralytics import YOLO

class YOLOv8_ObjectDetector:

    def __init__(self, model_file = 'yolov8n.pt', labels= None, conf = 0.25, iou = 0.45 ):

        self.conf = conf
        self.iou = iou

        self.model = YOLO(model_file)
        self.model_name = model_file.split('.')[0]
        self.results = None

        # if no labels are provided then use default COCO names 
        if labels == None:
            self.labels = self.model.names
        else:
            self.labels = labels

    def predict_img(self, img, verbose=True):

        # Run the model on the input image with the given parameters
        results = self.model(img, conf=self.conf, iou=self.iou, verbose=verbose)

        # Save the original image and the results for further analysis if needed
        self.orig_img = img
        self.results = results[0]

        # Return the detection results
        return results[0]



    def default_display(self, show_conf=True, line_width=None, font_size=None, 
                        font='Arial.ttf', pil=False, example='abc'):

        # Check if the `predict_img()` method has been called before displaying the detected objects
        if self.results is None:
            raise ValueError('No detected objects to display. Call predict_img() method first.')
        
        # Call the plot() method of the `self.results` object to display the detected objects on the original image
        display_img = self.results.plot(show_conf, line_width, font_size, font, pil, example)
        
        # Return the displayed image
        return display_img

        

    def custom_display(self, colors, show_cls = True, show_conf = True):
        img = self.orig_img
        # calculate the bounding box thickness based on the image width and height
        bbx_thickness = (img.shape[0] + img.shape[1]) // 450

        for box in self.results.boxes:
            textString = ""

            # Extract object class and confidence score
            score = box.conf.item() * 100
            class_id = int(box.cls.item())

            x1 , y1 , x2, y2 = np.array(box.xyxy.tolist()).astype(int)

            # Print detection info
            if show_cls:
                textString += f"{self.labels[class_id]}"

            if show_conf:
                textString += f" {score:,.2f}%"

            # Calculate font scale based on object size
            font = cv2.FONT_HERSHEY_COMPLEX
            fontScale = (((x2 - x1) / img.shape[0]) + ((y2 - y1) / img.shape[1])) / 2 * 2.5
            fontThickness = 1
            textSize, _ = cv2.getTextSize(textString, font, fontScale, fontThickness)

            # Draw bounding box, a centroid and label on the image
            img = cv2.rectangle(img, (x1,y1), (x2,y2), colors[class_id], bbx_thickness)
            center_coordinates = ((x1 + x2)//2, (y1 + y2) // 2)

            img =  cv2.circle(img, center_coordinates, 5 , (0,0,255), -1)
            
             # If there are no details to show on the image
            if textString != "":
                if (y1 < textSize[1]):
                    y1 = y1 + textSize[1]
                else:
                    y1 -= 2
                # show the details text in a filled rectangle
                img = cv2.rectangle(img, (x1, y1), (x1 + textSize[0] , y1 -  textSize[1]), colors[class_id], cv2.FILLED)
                img = cv2.putText(img, textString , 
                    (x1, y1), font, 
                    fontScale,  (0, 0, 0), fontThickness, cv2.LINE_AA)
        return img


    def predict_video(self, video_path, save_dir, save_format="avi", display='custom', verbose=True, **display_args):

        cap = cv2.VideoCapture(video_path)
        vid_name = os.path.basename(video_path)

        # Get the dimensions of each frame in the input video file
        width = int(cap.get(3))  # get `width`
        height = int(cap.get(4))  # get `height`

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # Set the name and path for the output video file
        save_name = self.model_name + '_' + vid_name.split('.')[0] + '.' + save_format
        save_file = os.path.join(save_dir, save_name)

        # Print information about the input and output video files if verbose is True
        if verbose:
            print("----------------------------")
            print(f"DETECTING OBJECTS IN : {vid_name} : ")
            print(f"RESOLUTION : {width}x{height}")
            print('SAVING TO :' + save_file)

        # Define an output VideoWriter object
        out = cv2.VideoWriter(save_file,
                              cv2.VideoWriter_fourcc(*"MJPG"),
                              30, (width, height))

        if not cap.isOpened():
            print("Error opening video stream or file")

        # Read each frame of the input video file
        while cap.isOpened():
            ret, frame = cap.read()

            # If the frame was not read successfully, break the loop
            if not ret:
                print("Error reading frame")
                break

            # Run object detection on the frame and calculate FPS
            beg = time.time()
            results = self.predict_img(frame, verbose=False)
            if results is None:
                print('***********************************************')
            fps = 1 / (time.time() - beg)

            # Display the detection results
            if display == 'default':
                frame = self.default_display(**display_args)
            elif display == 'custom':
                frame == self.custom_display(**display_args)

            # Display the FPS on the frame
            frame = cv2.putText(frame, f"FPS : {fps:,.2f}",
                                (5, 15), cv2.FONT_HERSHEY_COMPLEX,
                                0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Write the frame to the output video file
            out.write(frame)

            # Exit the loop if the 'q' button is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # After the loop release the cap and video writer
        cap.release()
        out.release()

class YOLOv8_ObjectCounter(YOLOv8_ObjectDetector):


    def __init__(self, model_file = 'yolov8n.pt', labels= None, conf = 0.25, iou = 0.45, 
                 track_max_age = 45, track_min_hits= 15, track_iou_threshold = 0.3 ):

        super().__init__(model_file , labels, conf, iou)

        self.track_max_age = track_max_age
        self.track_min_hits = track_min_hits
        self.track_iou_threshold = track_iou_threshold


    def predict_video(self, video_path, save_dir, save_format = "avi", 
                      display = 'custom', verbose = True, **display_args):
        
        cap = cv2.VideoCapture(video_path)
        # Get video name 
        vid_name = os.path.basename(video_path)


        # Get frame dimensions and print information about input video file
        width  = int(cap.get(3) )  # get `width` 
        height = int(cap.get(4) )  # get `height` 

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        save_name = self.model_name + '_' + vid_name.split('.')[0] + '.' + save_format
        save_file = os.path.join(save_dir, save_name)

        if verbose:
            print("----------------------------")
            print(f"DETECTING OBJECTS IN : {vid_name} : ")
            print(f"RESOLUTION : {width}x{height}")
            print('SAVING TO :' + save_file)

        out = cv2.VideoWriter(save_file,
                            cv2.VideoWriter_fourcc(*"MJPG"),
                            30,(width,height))

        # Check if the video is opened correctly
        if not cap.isOpened():
            print("Error opening video stream or file")

        # Initialize object tracker
        tracker = sort.Sort(max_age = self.track_max_age, min_hits= self.track_min_hits , 
                            iou_threshold = self.track_iou_threshold)
        
        totalCount = []
        # Read the video frames
        while cap.isOpened():

            detections = np.empty((0, 5),dtype = int)
            ret, frame = cap.read()

            # If the frame was not read successfully, break the loop
            if not ret:
                print("Error reading frame")
                break

            # Run object detection on the frame and calculate FPS
            beg = time.time()
            results = self.predict_img(frame, verbose = False)
            if results == None:
                print('***********************************************')
            fps = 1 / (time.time() - beg)
            for box in results.boxes:
                score = box.conf.item() * 100

                boxx = box.xyxy.tolist()

                detections = np.vstack((detections, np.append(boxx, score)))

            # Update object tracker 
            resultsTracker = tracker.update(detections)
            for result in resultsTracker:
                #print(type(result))

                # Get the tracker results
                x1, y1, x2, y2, id = result.astype(int)
                x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
                #print(result)

                # Display current objects IDs
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2
                id_txt = f"ID: {str(id)}"
                cv2.putText(frame, id_txt, (cx, cy), 4, 0.5, (0, 0, 255), 1)

                # if we haven't seen aprticular object ID before, register it in a list 
                if totalCount.count(id) == 0:
                    totalCount.append(id)

            # Display detection results
            if display == 'default':
                frame = self.default_display(**display_args)
            
            elif display == 'custom':
                frame == self.custom_display( **display_args)

            # Display FPS on frame
            frame = cv2.putText(frame,f"FPS : {fps:,.2f}" , 
                                (5,55), cv2.FONT_HERSHEY_COMPLEX, 
                            0.5,  (0,255,255), 1, cv2.LINE_AA)
            
            # Display Counting results
            count_txt = f"TOTAL COUNT : {len(totalCount)}"
            frame = cv2.putText(frame, count_txt, (5,45), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        

            # append frame to the video file
            out.write(frame)
            
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # After the loop release the cap 
        cap.release()
        out.release()
        print(len(totalCount))
        print(totalCount)