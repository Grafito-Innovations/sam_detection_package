import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import torch
import time
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from geometry_msgs.msg import Point

class StemDetector(Node):
    def __init__(self):
        super().__init__('stem_detector')
        self.cam1 = cv2.VideoCapture(2)
        self.cam2 = cv2.VideoCapture(4)

        # Set up camera parameters (update these with the correct values)
        self.focal_length = 100.0  # Focal length of the cameras (in pixels)
        self.baseline = 10.0  # Baseline between the two cameras (in arbitrary units)

        # Initialize SAM models and predictors
        sam_checkpoint = "/home/pranav/ros2_ws/src/sam_detection_package/sam_detection_package/mobile_sam.pt"
        #--------------------------------------CHANGE THIS PATH TO YOUR CHECKPOINT------------------------------------<<
        model_type = "vit_t"
        device = "cuda"
        self.sam1 = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam2 = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam1.to(device=device)
        self.sam2.to(device=device)
        self.sam1.eval()
        self.sam2.eval()
        self.predictor1 = SamPredictor(self.sam1)
        self.predictor2 = SamPredictor(self.sam2)

        # Initialize the coordinate variables
        self.first_coordinates = None
        self.second_coordinates = None
        self.difference_rounded = None

        # Create a publisher
        self.publisher = self.create_publisher(Point, 'stem_alignment_data', 10)


    def calculate_coordinates(self, ends):
        # For each pair of stem ends
        for i, (end1, end2) in enumerate(ends):
            if end1 is None or end2 is None:
                self.get_logger().warn('Endpoints not detected. Unable to calculate coordinates.')
                return

            # Compute disparity
            disparity = abs(end2[0] - end1[0])

            # Compute depth
            depth = round(self.baseline * self.focal_length / disparity, 2)

            # Compute x_cam and y_cam
            x_cam = (end1[0] + end2[0]) / 2
            y_cam = (end1[1] + end2[1]) / 2

            # Compute z_cam
            z_cam = round(self.focal_length * self.baseline / disparity, 2)

            # Compute x_world and y_world
            x_world = round(z_cam * (x_cam - self.cam1.get(cv2.CAP_PROP_FRAME_WIDTH) / 2) / self.focal_length, 2)
            y_world = round(z_cam * (y_cam - self.cam1.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2) / self.focal_length, 2)
            
            # Store the coordinates
            if i == 0:
                self.first_coordinates = (x_world, y_world, depth)
            else:
                self.second_coordinates = (x_world, y_world, depth)

    def stem_boundingbox(self, frame, draw_box=True):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([25, 50, 65])
        upper_green = np.array([55, 190, 200])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        adaptive_thresh = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 20)
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Define the minimum contour area
        min_contour_area = 500  # You can set the value as per your requirement

        # Filter out contours with area less than the minimum contour area
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        bounding_boxes = []

        if contours:
            largest_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:2]  # get largest and 2nd largest contours
            for contour in largest_contours:
                x, y, w, h = cv2.boundingRect(contour)
                padding = 20
                x -= padding
                y -= padding
                w += 2 * padding
                h += 2 * padding
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                bounding_boxes.append((x, y, w, h))

        return bounding_boxes
    
    def show_mask(self, mask, image):
        color = np.array([235, 220, 177])  
        mask = (mask * 255).astype(np.uint8)  
        image[mask == 255] = color                               
        return image

    def show_box(self, box, image):
        x0, y0, x1, y1 = box
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)  
        return image
    
    def show_dot(self, point, image):
        x, y = point  
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Draw the dot
        return image

    def sort_masks(self, masks):
        # Calculate centroids for each mask
        centroids = []
        for mask in masks:
            y, x = np.where(mask)
            if y.size != 0:
                centroid_y = np.mean(y)  # Calculate the y-coordinate of the centroid
                centroids.append(centroid_y)

        # Sort the masks based on the centroid's y-coordinate (ascending order)
        sorted_masks = [mask for _, mask in sorted(zip(centroids, masks))]

        return sorted_masks

    def sam_segmentation(self, frame1, frame2):
        self.predictor1.set_image(frame1)
        bounding_boxes1 = self.stem_boundingbox(frame1)

        self.predictor2.set_image(frame2)
        bounding_boxes2 = self.stem_boundingbox(frame2)

        results = []

        masks1 = []
        points1 = []
        flag = 0
        for bb1 in bounding_boxes1:
            x1, y1, w1, h1 = bb1
            input_box1 = np.array([x1, y1, x1 + w1, y1 + h1])
            mask1, _, _ = self.predictor1.predict(point_coords=None, point_labels=None, box=input_box1[None, :], multimask_output=False)
            masks1.append(mask1[0])

            y, x = np.where(mask1[0])  # Get all non-zero points
            if y.size != 0:
                if flag == 0:
                    y_max = np.max(y)  # Find maximum y-coordinate
                    x_max = x[np.argmax(y)]  # Corresponding x-coordinate
                    dot = (x_max, y_max)
                    points1.append(dot)
                    flag = 1
                else:
                    y_min = np.min(y)  # Find minimum y-coordinate
                    x_min = x[np.argmin(y)]  # Corresponding x-coordinate
                    dot = (x_min, y_min)
                    points1.append(dot)

        masks2 = []
        points2 = []
        flag = 0
        for bb2 in bounding_boxes2:
            x2, y2, w2, h2 = bb2
            input_box2 = np.array([x2, y2, x2 + w2, y2 + h2])
            mask2, _, _ = self.predictor2.predict(point_coords=None, point_labels=None, box=input_box2[None, :], multimask_output=False)
            masks2.append(mask2[0])

            y, x = np.where(mask2[0])  # Get all non-zero points
            if y.size != 0:
                if flag == 0:
                    y_max = np.max(y)  # Find maximum y-coordinate
                    x_max = x[np.argmax(y)]  # Corresponding x-coordinate
                    dot = (x_max, y_max)
                    points2.append(dot)
                    flag = 1
                else:
                    y_min = np.min(y)  # Find minimum y-coordinate
                    x_min = x[np.argmin(y)]  # Corresponding x-coordinate
                    dot = (x_min, y_min)
                    points2.append(dot)


        # Sort the masks based on the centroid's y-coordinate
        masks1 = self.sort_masks(masks1)
        masks2 = self.sort_masks(masks2)

        results.append((masks1, bounding_boxes1, points1))
        results.append((masks2, bounding_boxes2, points2))

        return results

    def process_frame(self):
        frame_count = 0
        while True:
            # Capture frames from both cameras
            ret1, frame1 = self.cam1.read()
            ret2, frame2 = self.cam2.read()
            frame_count += 1
            if not ret1 or not ret2:
                self.get_logger().error("Unable to capture images from one or both cameras")
                break
            # Process every 5th frame
            if frame_count % 1 == 0:
                results = self.sam_segmentation(frame1, frame2)

                points1 = None
                points2 = None

                for idx, (masks, input_box, points) in enumerate(results):

                    if masks:  # Check if masks list is not empty
                        if idx == 0 and len(masks) > 1:
                            frame1 = self.show_mask(masks[0], frame1)
                            frame1 = self.show_mask(masks[1], frame1)
                        elif idx == 1 and len(masks) > 1:
                            frame2 = self.show_mask(masks[0], frame2)
                            frame2 = self.show_mask(masks[1], frame2)
                        
                        # Show points
                        if points:
                            if idx == 0 and len(masks) > 1:
                                frame1 = self.show_dot(points[0], frame1)
                                frame1 = self.show_dot(points[1], frame1)
                            elif idx == 1 and len(masks) > 1:
                                frame2 = self.show_dot(points[0], frame2)
                                frame2 = self.show_dot(points[1], frame2)

                        # Store the points of the first item in "results" in "points1"
                        if idx == 0:
                            points1 = points
                        # Store the points of the second item in "results" in "points2"
                        elif idx == 1:
                            points2 = points

                # Calculate the 3D coordinates of the stem ends
                if points1 is not None and points2 is not None:
                    self.calculate_coordinates(zip(points1, points2))

                # Merge the frames
                both_frames = np.hstack((frame1, frame2))
                
                # Calculate and log the difference in coordinates
                if self.first_coordinates and self.second_coordinates:
                    difference = tuple(np.subtract(self.first_coordinates, self.second_coordinates))
                    self.difference_rounded = tuple(round(i, 2) for i in difference)  # Round to two decimal places

                    # Convert difference_rounded to a Point message
                    point_msg = Point()
                    point_msg.x = self.difference_rounded[0]
                    point_msg.y = self.difference_rounded[1]
                    point_msg.z = self.difference_rounded[2]

                    # Publish the point
                    self.publisher.publish(point_msg)

                    # Convert difference_rounded to string
                    difference_str = "Difference: " + str(self.difference_rounded)

                    # Put text on the frames
                    cv2.putText(both_frames, difference_str, (730,440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Show the frames
                cv2.imshow('Camera 1 & 2', both_frames)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cam1.release()
        self.cam2.release()
        cv2.destroyAllWindows()


# main function
def main(args=None):
    rclpy.init(args=args)
    stem_detector = StemDetector()
    stem_detector.process_frame()
    rclpy.shutdown()

if __name__ == '__main__':
    main()