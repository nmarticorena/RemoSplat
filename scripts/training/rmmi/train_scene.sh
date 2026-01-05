#!/bin/bash

set -e

#######################################
# Default parameters
#######################################
voxel_size=0.01
skip_frames=1
down_sample_frames=30
extra=0.0
max_depth=10.0

########################################
# Paths
########################################
scene_dir="data/scenes/ReMoSplat-synthetic"
result_dir="data/gsplat_2D"

#######################################
# Scene Selection
#######################################
if [ "$1" == "bookshelf" ]; then
    folder_path="$scene_dir/bookshelf_cage/bookshelf_cage_"
    result_path="$result_dir/bookshelf_cage/bookshelf_cage_"

elif [ "$1" == "table" ]; then
    folder_path="$scene_dir/table_new/table_new_"
    result_path="$result_dir/table_new/table_new_"

else
    echo "Usage:"
    echo "   ./train_scene.sh bookshelf [N]"
    echo "   ./train_scene.sh table [N]"
    echo ""
    echo "If N is omitted, the script assume 500 scenes"
    exit 1
fi

#######################################
# Determine number of datasets
#######################################
if [ -n "$2" ]; then
    n_scenes=$2
    echo "Using user-defined number of scenes: $n_scenes"
else
    n_scenes=500
    echo "Using default number of scenes: 500"
fi

#######################################
# Main loop
#######################################
total_ti=$(date +%s)

for ((i=0; i<n_scenes; i++))
do
    dataset_path="$folder_path$(printf '%04d' $i)"
    splat_result_path="$result_path$(printf '%04d' $i)"

    echo ""
    echo "Processing dataset: $dataset_path"
    echo "Saving splats to:   $splat_result_path"
    echo ""
    ti=$(date +%s)
    compute-aabb  --dataset.dataset-path $dataset_path --save \
        --pcd.down-sample-voxel-size $voxel_size \
        --pcd.skip-frames $skip_frames \
        --pcd.down-sample-frames $down_sample_frames \
        --extra $extra \
        --pcd.max-depth $max_depth \
        --pcd.max_points 60000  \
        --no-gui

    CUDA_VISIBLE_DEVICES=0 python scripts/training/simple_trainer_2dgs.py default \
      --data_dir $dataset_path \
      --data_factor 1 \
      --result_dir $splat_result_path \
      --no-normalize-world-space \
      --depth-loss \
      --init_type sfm \
      --normal-loss \
      --no-random-bkgd \
      --max-steps 1000 \
      --save-steps 1 1000  \
      --test-every 40 \
      --disable-viewer
    iteration_end_time=$(date +%s)

    # Calculate iteration time
    iteration_time=$((iteration_end_time - ti))
    echo "Iteration $i completed in $((iteration_time / 60)) minutes and $((iteration_time % 60)) seconds"
done

total_end_time=$(date +%s)
total_time=$((total_end_time - total_ti))
echo "Total time: $((total_time / 3600)) hours $((total_time / 60)) minutes and $((total_time % 60)) seconds"
