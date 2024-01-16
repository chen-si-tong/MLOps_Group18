# Use the official lightweight Python image
# https://hub.docker.com/_/python
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirments_run.txt .
# COPY mlopslfm-3cc1ffa05d44.json /root/.config/gcloud/

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirments_run.txt

# Copy the application code into the container
COPY cloud_run.py .

# Set the PORT environment variable
ENV PORT 8080

# Command to run the application
CMD ["python", "cloud_run.py"]
