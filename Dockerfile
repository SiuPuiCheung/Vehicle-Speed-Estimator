# Use Ultralytics prebuilt image
FROM ultralytics/ultralytics:latest

WORKDIR /app

# Install additional dependencies
RUN pip install --no-cache-dir supervision==0.25.1

# Copy your application files
COPY . .

# Run your application
ENTRYPOINT ["python", "VehicleTracker.py"]
CMD ["--video_path", "videos/video_01.mp4", "--task", "speed", "--conf", "0.4", "--iou", "0.7", "--show", "False"]
