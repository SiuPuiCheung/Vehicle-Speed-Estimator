import os
import csv
import cv2
import argparse
import supervision as sv
from ultralytics import YOLO
import torch
import numpy as np
from collections import defaultdict, deque
from datetime import datetime

class VehicleTracker:
    """
    A class to perform vehicle detection, tracking, and speed estimation using YOLO and Supervision.
    """

    def __init__(self, video_path, conf=0.4, iou=0.7, show=False):
        """
        Initializes the VehicleTracker with the specified video path, confidence threshold, and IOU threshold.
        """
        self.video_path = video_path
        self.conf = conf
        self.iou = iou
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.show = show

        # Load YOLO model
        self.model = YOLO("yolo11n.pt").to(self.device)
        self.class_names = self.model.model.names

        # Define vehicle classes (Car, Motorcycle, Bus, Truck)
        self.classes = [2, 3, 5, 7]

        # Supervision utilities
        self.bounding_box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=1)
        self.byte_track = sv.ByteTrack()

        # Define the estimated zone (Polygon region)
        self.estimated_zone = sv.PolygonZone(
            polygon=np.array([(0, 719), (1279, 719), (1279, 533), (551, 168), (192, 170)])
        )

        # Perspective transformation for real-world speed estimation
        self.view_transformer = ViewTransformer()

        # Output directory for saving videos and data
        self.output_folder = "output"
        os.makedirs(self.output_folder, exist_ok=True)

    def process_frame(self, frame):
        """
        Runs YOLO detection and ByteTrack tracking on a given frame.
        """
        result = self.model(frame, conf=self.conf, iou=self.iou, classes=self.classes, device=self.device)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[self.estimated_zone.trigger(detections)]
        detections = self.byte_track.update_with_detections(detections=detections)
        return detections
    
    def display_frame(self, window_name, frame):
        """
        Displays the video frame and checks for exit signal.
        Returns True if 'q' is pressed, otherwise False.
        """
        if self.show == True: 
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                print("Exiting program...")
                return True
        return False



    def timestamp(self):
        """Returns a timestamp string for unique file names."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")


    def object_detection(self):
        """
        Performs object detection, annotates detected vehicles, and saves the output video.
        """
        output_video = os.path.join(self.output_folder, f"detection_{self.timestamp()}.mp4")
        video_info = sv.VideoInfo.from_video_path(self.video_path)

        with sv.VideoSink(output_video, video_info) as sink:
            for frame in sv.get_video_frames_generator(self.video_path):
                detections = self.process_frame(frame)

                # Create labels for detected objects
                labels = [f"{self.class_names[cid]} ({conf:.2f})" for cid, conf in zip(detections.class_id, detections.confidence)]

                # Annotate the frame with bounding boxes and labels
                annotated_frame = self.label_annotator.annotate(
                    scene=self.bounding_box_annotator.annotate(scene=frame.copy(), detections=detections),
                    detections=detections,
                    labels=labels
                )

                # Save the annotated frame to the output video
                sink.write_frame(annotated_frame)

                if self.display_frame("Object Detection", annotated_frame):
                    break  # Exit loop if 'q' is pressed

                

        print(f"Object detection video saved: {output_video}")

    def object_tracking(self):
        """
        Performs object tracking, annotates tracked vehicles, and saves the output video.
        """
        output_video = os.path.join(self.output_folder, f"tracking_{self.timestamp()}.mp4")
        video_info = sv.VideoInfo.from_video_path(self.video_path)

        with sv.VideoSink(output_video, video_info) as sink:
            for frame in sv.get_video_frames_generator(self.video_path):
                detections = self.process_frame(frame)

                # Create labels with tracker IDs
                labels = [f"ID {tid}: {self.class_names[cid]} ({conf:.2f})" 
                          for tid, cid, conf in zip(detections.tracker_id, detections.class_id, detections.confidence)]

                # Annotate the frame with bounding boxes and labels
                annotated_frame = self.label_annotator.annotate(
                    scene=self.bounding_box_annotator.annotate(scene=frame.copy(), detections=detections),
                    detections=detections,
                    labels=labels
                )

                # Save the annotated frame to the output video
                sink.write_frame(annotated_frame)

                if self.display_frame("Object Tracking", annotated_frame):
                    break  # Exit loop if 'q' is pressed

        print(f"Object tracking video saved: {output_video}")

    def speed_estimation(self):
        """
        Estimates vehicle speed, saves results to a video and CSV file.
        """
        timestamp = self.timestamp()
        output_video = os.path.join(self.output_folder, f"speed_{timestamp}.mp4")
        output_csv = os.path.join(self.output_folder, f"speed_{timestamp}.csv")
        video_info = sv.VideoInfo.from_video_path(self.video_path)

        # Stores past coordinates for speed calculation
        coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
        speed_records = defaultdict(list)

        with sv.VideoSink(output_video, video_info) as sink:
            for frame in sv.get_video_frames_generator(self.video_path):
                detections = self.process_frame(frame)

                # Get bottom-center points of bounding boxes
                points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

                # Convert image-space points to real-world feet
                real_world_points = self.view_transformer.transform_points(points=points).astype(int)

                labels = []
                for tracker_id, (_, real_y) in zip(detections.tracker_id, real_world_points):
                    coordinates[tracker_id].append(real_y)

                # Calculate speed for each tracked vehicle
                for tracker_id in detections.tracker_id:
                    if len(coordinates[tracker_id]) < video_info.fps / 2:
                        labels.append(f"ID {tracker_id}")  
                    else:
                        distance_feet = abs(coordinates[tracker_id][-1] - coordinates[tracker_id][0])
                        time_seconds = len(coordinates[tracker_id]) / video_info.fps
                        speed_kmh = (distance_feet / time_seconds) * 1.09728
                        labels.append(f"ID {tracker_id} | {int(speed_kmh)} km/h")
                        speed_records[tracker_id].append(speed_kmh)

                # Annotate the frame with speed labels
                annotated_frame = self.label_annotator.annotate(
                    scene=self.bounding_box_annotator.annotate(scene=frame.copy(), detections=detections),
                    detections=detections,
                    labels=labels
                )

                sink.write_frame(annotated_frame)
                
                if self.display_frame("Speed Estimation", annotated_frame):
                    break  # Exit loop if 'q' is pressed

        print(f"Speed estimation video saved: {output_video}")

        # Save speed statistics to CSV
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Tracker ID", "Min Speed (km/h)", "Max Speed (km/h)", "Average Speed (km/h)"])
            for tracker_id, speeds in speed_records.items():
                min_speed = min(speeds) if speeds else 0
                max_speed = max(speeds) if speeds else 0
                avg_speed = sum(speeds) / len(speeds) if speeds else 0
                writer.writerow([tracker_id, round(min_speed, 2), round(max_speed, 2), round(avg_speed, 2)])

        print(f"Speed data saved to CSV: {output_csv}")



class ViewTransformer:
    """
    Handles perspective transformation to convert image-space coordinates to real-world measurements.
    """
    def __init__(self):
        source = np.array([[249, 638], [252, 546], [450, 229], [781, 286]])
        target = np.array([[0, 0], [0, 19], [33, 228], [77, 154]])
        self.m, _ = cv2.findHomography(source.astype(np.float32), target.astype(np.float32), method=cv2.RANSAC)

    def transform_points(self, points):
        """Transforms image-space points to real-world coordinates."""
        return cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float32), self.m).reshape(-1, 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Required video path argument
    parser.add_argument("--video_path", type=str, default="videos/video_01.mp4", help="Path to the video file (default: videos/video_01.mp4)")

    # Task selection: detection, tracking, or speed
    parser.add_argument("--task", type=str, default="speed", choices=["detection", "tracking", "speed"], help="Task to perform (default: speed)")

    # Optional arguments for confidence threshold and IOU threshold
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    parser.add_argument("--iou", type=float, default=0.7, help="IOU threshold (default: 0.7)")

    parser.add_argument("--show", type=bool, default=False, help="Display the video while processing (default: False)")
    args = parser.parse_args()

    # Pass the arguments to the VehicleTracker class
    analyzer = VehicleTracker(args.video_path, args.conf, args.iou, args.show)

    # Execute the selected task
    if args.task == "detection":
        analyzer.object_detection()
    elif args.task == "tracking":
        analyzer.object_tracking()
    elif args.task == "speed":
        analyzer.speed_estimation()
