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
If you give `--num-hp` as 1, then you can easily run simple model parallelism in a single node.
For the data parallelism, we provides flag `--DP-ONLY`, which execute data parallelism only.
For more detail, please refer to following explanation.
 

#### Hybrid Parallelism with Bundle

The code provides 

```bash
python main.py \ 
        --batch-size = 32 \
        --rank = {rank of this node} \
        --num-hp = {number of bundles running in this node} \
        --IP = {IP_address_for_MASTER} \
        --world-size = {number of inter-connected servers} \
        {ImageNet Data Path}
```

Here, `--num-hp` option gets the number of bundles running on the designated node.
`--world-size` denotes the number of servers which are interconnected each other.


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



