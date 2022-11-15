#FROM docker.io/pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
#FROM docker.io/pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
FROM docker.io/pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel


RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt update
RUN apt-get install -y git
RUN apt-get install -y wget
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install debugpy

#install azcopy
RUN wget https://azcopyvnext.azureedge.net/release20220511/azcopy_linux_amd64_10.15.0.tar.gz 
RUN tar -xvzf azcopy_linux_amd64_10.15.0.tar.gz
RUN cp ./azcopy_linux_amd64_*/azcopy /usr/bin/ 
RUN chmod 775 /usr/bin/azcopy
RUN rm azcopy_linux_amd64_10.15.0.tar.gz
RUN rm -rf azcopy_linux_amd64_*

#download models, features, dataset and checkpoints for ALADIN
RUN mkdir models
#RUN azcopy copy 'https://biglmdiag.blob.core.windows.net/vinvl/image_features/coco_X152C4_frcnnbig2_exp168model_0060000model.roi_heads.nm_filter_2_model.roi_heads.score_thresh_0.2/model_0060000/' ./features --recursive
#RUN azcopy copy 'https://biglmdiag.blob.core.windows.net/vinvl/model_ckpts/coco_ir/base/checkpoint-0132780/' ./checkpoints --recursive
#RUN azcopy copy 'https://biglmdiag.blob.core.windows.net/vinvl/datasets/coco_ir' ./data --recursive
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1PQvzOyuzDPgs9NO8rWkzzKFVE8ENnH93' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1PQvzOyuzDPgs9NO8rWkzzKFVE8ENnH93" -O ./models/best_model_align_and_distill.pth.tar && rm -rf /tmp/cookies.txt



# #install apex
RUN git clone https://github.com/NVIDIA/apex.git
WORKDIR /workspace/apex
RUN git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0
RUN python setup.py install --cuda_ext --cpp_ext
WORKDIR /workspace

#install OSCAR
RUN git clone --recursive https://github.com/mesnico/OSCAR-TERAN-distillation
WORKDIR /workspace/OSCAR-TERAN-distillation/coco_caption
RUN ls
RUN ./get_stanford_models.sh
WORKDIR /workspace/OSCAR-TERAN-distillation/
RUN python setup.py build develop
#installing the requirements of ALADIN
RUN pip install -r requirements.txt
RUN pip install tensorboard
WORKDIR /workspace

#installing detectron2
#RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --use-feature=2020-resolver
RUN git clone https://github.com/lorebianchi98/detectron2.git
WORKDIR /workspace/detectron2
RUN pip install -e .
WORKDIR /workspace

#installing Detic
RUN git clone https://github.com/lorebianchi98/Detic.git --recurse-submodules
WORKDIR /workspace/Detic
RUN pip install -r requirements.txt
WORKDIR /workspace/