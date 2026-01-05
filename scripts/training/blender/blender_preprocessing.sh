# Training all the datasets available as blender scenes
#
#
# We first need to compute the aabb of all the scenes, this also
# would transform them to the colmap frame
#
# Example of a manual call:
#  compute_aabb --dataset.dataset-path /media/nmarticorena/DATA/datasets/nerf_standard/bookshelf --save --pcd.down-sample-voxel-size 0.001 --pcd.skip-frames 1 --pcd.down-sample-frames 10 --extra 0.0 --pcd.max-depth 4

DATASET_FOLDER="/media/nmarticorena/DATA/datasets/nerf_standard/"

SCENES=("bookshelf" "counter" "industrial_a" "tunnel" "tunnel_sdf")

for s in "${SCENES[@]}"; do
  compute-aabb \
    --dataset.dataset-path ${DATASET_FOLDER}${s} \
    --save \
    --pcd.down-sample-voxel-size 0.01 \
    --pcd.skip-frames 1 \
    --pcd.down-sample-frames 10 \
    --extra 0.0 \
    --pcd.max-depth 4 \
    --gui
done
