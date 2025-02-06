# Use Ultralytics prebuilt image
FROM ultralytics/ultralytics:latest

WORKDIR /app

# Install additional dependencies
RUN pip install --no-cache-dir supervision==0.25.1

# Copy your application files
COPY . .

# Run your application
CMD ["python", "VehicleTracker.py"]
