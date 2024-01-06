## Pull from existing image
# FROM nvcr.io/nvidia/pytorch:21.05-py3
FROM python:3.10-slim

RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN apt-get -y install libglib2.0-dev
RUN apt install libgomp1 
##open3d

## Copy requirements
COPY ./requirements.txt .

## Install Python packages in Docker image
# RUN python -m pip install --user torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
RUN python -m pip install --user torch torchvision torchaudio

RUN pip3 install -r requirements.txt

## Copy all files
COPY ./ ./

## Execute the inference command 
CMD ["./main_test.py"]
ENTRYPOINT ["python"]