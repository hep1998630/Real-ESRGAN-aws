# Use PyTorch's official CUDA base image
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

ENTRYPOINT ["uvicorn", "main:app", "--port", "8000"]