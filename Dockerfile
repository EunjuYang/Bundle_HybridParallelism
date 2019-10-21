FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime
MAINTAINER Eun-Ju Yang <yejyang@kaist.ac.kr>

RUN apt-get update  && apt-get install -y git
RUN pip install termcolor
RUN git clone https://github.com/EunjuYang/Bundle_HybridParallelism.git
WORKDIR /workspace/Bundle_HybridParallelism
ENV PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
ENTRYPOINT ["python", "main.py"]
