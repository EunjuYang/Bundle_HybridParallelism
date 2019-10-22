#!/bin/bash

IMG_NAME="yejyang/bundlehp:v01"
OVERLAY_NAME="hp_bundle"

printf "\n====== RUN 'Bundle-based Hybrid-Parallelism' ======\n\n"
if [ "$#" -lt 1 ]; then
    echo "You can pass arguments with : $0 [batch_size] [rank] [shape] [num_bundle] [world_size] [model] [data_path] [itr]"
    printf "\t global batch size: "
    read batch_size
    printf "\t global rank of this node: "
    read rank
    printf "\t shape of bundle (front,rear): "
    read shape
    printf "\t number of hp bundles: "
    read num_hp
    printf "\t The number of total nodes in cluster used for training: "
    read world_size
    printf "\t Model name to run: "
    read model
    printf "\t Path of ImageNet data: "
    read data
    printf "\t Number of iterations to run: "
    read itr
else
    args=("$@")
    #[batch_size] [rank] [shape] [num_bundle] [world_size] [model] [data_path] [itr]"
    batch_size=${args[0]}
    rank=${args[1]}
    shape=${args[2]}
    num_hp=${args[3]}
    world_size=${args[4]}
    model=${args[5]}
    data=${args[6]}
    itr=${args[7]}
fi

if [ "$rank" == "0" ];then
  echo docker network rm $OVERLAY_NAME
  docker network rm $OVERLAY_NAME
  echo network create --driver=overlay
  docker network create --driver=overlay  --gateway 10.0.1.1 --subnet 10.0.1.0/24 --attachable $OVERLAY_NAME
fi

ip_=$(($rank + 10))
CONTAINER_IP="10.0.1.${ip_}"
MASTER_IP="10.0.1.10"
port_num=9998


run_cmd="--batch-size $batch_size
        --rank $rank
        --bundle-shape $shape
        --num-bundle $num_hp
        --itr $itr
        --IP $MASTER_IP
        --portNum $port_num
        --model $model
        --world-size $world_size
        $data"
echo Running Bundle+based+Hybrid+Parallelism with following options
echo $run_cmd
docker run --runtime=nvidia --network $OVERLAY_NAME --rm -v $data:$data --ipc=host --ip $CONTAINER_IP $IMG_NAME $run_cmd
