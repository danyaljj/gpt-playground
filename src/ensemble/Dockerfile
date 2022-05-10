FROM python:3.8

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

RUN pip3 install wheel
RUN python3 -m pip install -U --force-reinstall pip
RUN pip3 install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN pip3 install --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY classification.py .
COPY multiple_choice_ensemble.py .
