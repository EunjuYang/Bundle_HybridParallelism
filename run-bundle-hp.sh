#!/bin/bash

IMG_NAME="yejyang/bundlehp:v01"

printf "\n====== RUN 'Bundle-based Hybrid-Parallelism' ======\n\n"
if [ "$#" -lt 1 ]; then
    echo "You can pass arguments with : $0 [batch_size] [rank] [shape] [num_bundle] [IP] [portNum] [world_size] [model] [data_path] [itr]"
    printf "\t global batch size: "
    read batch_size
    printf "\t global rank of this node: "
    read rank
    printf "\t shape of bundle (front,rear): "
    read shape
    printf "\t number of hp bundles: "
    read num_hp
    printf "\t IP address of master node: "
    read IP
    printf "\t Port number of master node: "
    read port_num
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
    #[batch_size] [rank] [shape] [num_bundle] [IP] [portNum] [world_size] [model] [data_path] [itr]"
    batch_size=${args[0]}
    rank=${args[1]}
    shape=${args[2]}
    num_hp=${args[3]}
    IP=${args[4]}
    port_num=${args[5]}
    world_size=${args[6]}
    model=${args[7]}
    data=${args[8]}
    itr=${args[9]}
fi

run_cmd="--batch-size $batch_size
        --rank $rank
        --bundle-shape $shape
        --num-bundle $num_hp
        --itr $itr
        --IP $IP
        --portNum $port_num
        --model $model
        --world-size $world_size
        $data"
echo Running Bundle+based+Hybrid+Parallelism with following options
echo $run_cmd
docker run --runtime=nvidia -p $port_num:$port_num -v $data:$data --ipc=host $IMG_NAME $run_cmd
