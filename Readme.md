# Bundle-based Hybrid Parallelism Example

This repository contains example code for Bundle-based Hybrid Parallelism.
How to split the model should be defined under `model.py` with its model name.

----
### Bundle-based Hybrid Parallelism


![Example of Hybrid Parallelism](Figure/Bundle_HP_Figure.png)

By default, bundle works with two sub-models: front and rear model. 
Each model runs on one GPU and they communicate each other via CPU.
It is possible to run several bundles on one server (unless enough GPUs are installed in the server).
Then, a bundle with rank 0 (i.e., local MASTER) collects all gradients in every synchronization step.
All local MASTER use all-reduce to synchronize all gradients among them.
For more detail, please refer to paper.

----

### How to execute

This code supports ...  
 - Bundle-based hybrid parallelism 
 - Data parallelism
 - Model parallelism
 
By default, the code supports to execute split model with hybrid parallelism.
Following its definition, model parallelism with two workers is easily implemented using the code.
 

#### Hybrid Parallelism with Bundle

The code provides 

```bash
 python main.py \
        --batch-size = {batch size for one bundle} 
        --rank = {rank of this node} \
        --itr = {number of iteration you run} \
        --bundle-shape = {shape of bundle, e.g., 1,1 or 4,2 etc.}\
        --num-bundle = {number of bundles runnning in this cluster}\
        --IP = {IP address of master node}\
        --portNum = {portnumber of master node}\
        --model = {model name - partitioned model name}\
        --world-size = {the number of total nodes running in this cluster}\
        {PATH for ImageNet Data DIR}\
```

Here, 
`--num-bundle` option gets the number of bundles running on the designated node.
`--world-size` denotes the number of servers which are interconnected each other.
`--bundle-shape` denotes the shape of bundle (We only support partitioned in two parts, i.e. front & rear)


#### Pure data Parallelism

The code provides `--DP-ONLY` option. 
If you type the flag, then it means you will run the code only using data parallelism.
Then, `--num-hp` flag will only get the value of the number of workers used for data parallelism.

```bash
python main.py \
        --batch-size = 32 \
        --rank = {rank of this node} \
        --num-hp = {number of workers for data parallelism} \
        --IP = {IP_address_for_MASTER} \
        --modela = resnet101 \
        --world-size = {number of inter-DP servers} \
        --DP-ONLY \
        ~/Downloads/ImageNet/
```

----

#### Using Docker

##### Pre-Requisites

- Docker
- Nvidia-Docker or [Accelerator-Docker](https://github.com/KAIST-NCL/Accelerator-Docker.git) to support running containers with GPUs

##### Compile container image
```bash
docker build -t yejyang/budlehp:v01 ./
```
##### Run the bundle container with the script file

Passing parameters as arguments
```bash
./run-bundle-hp.sh [batch_size] [rank] [shape] [num_bundle] [IP] [portNum] [world_size] [model] [data_path] [itr]
``` 

Or just running shell script to get some helps.
```bash
./run-bundle-hp.sh 
``` 
