FROM tensorflow/tensorflow:latest-gpu-py3
RUN apt-get update ; apt-get install vim git wget -y
RUN pip install --upgrade scikit-image tqdm keras

