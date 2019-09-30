# Bundle-based Hybrid Parallelism Example

This repository contains example code for Bundle-based Hybrid Parallelism.
How to split the model should be defined under `model.py` with its model name.


#### How to execute

```bash
python main.py \ 
        --batch-size = 32 \
        --rank = 0 \
        --num-hp = {number of bundles running in this node} \
        --IP = {IP_address_for_MASTER} \
        --world-size = {number of inter-connected servers} \
        {ImageNet Data Path}
```

Here, `--num-hp` option gets the number of bundles running on the designated node.
`--world-size` denotes the number of servers which are interconnected each other.

#### Bundle-based Hybrid Parallelism

By default, bundle works with two sub-models: front and rear model. 
Each model runs on one GPU and they communicate each other via CPU.
It is possible to run several bundles on one server (unless enough GPUs are installed in the server).
Then, a bundle with rank 0 (i.e., local MASTER) collects all gradients in every synchronization step.
All local MASTER use all-reduce to synchronize all gradients among them.
For more detail, please refer to paper.