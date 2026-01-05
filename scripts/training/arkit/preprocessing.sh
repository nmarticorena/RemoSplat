# Training all the datasets available as blender scenes
#
#
# We first need to compute the aabb of all the scenes, this also
# would transform them to the colmap frame
#
# Example of a manual call:
#  python ~/Documents/phd_tools/scripts/compute_aabb.py --dataset.dataset-path /media/nmarticorena/DATA/datasets/nerf_standard/bookshelf --save --pcd.down-sample-voxel-size 0.001 --pcd.skip-frames 1 --pcd.down-sample-frames 10 --extra 0.0 --pcd.max-depth 4

DATASET_FOLDER="$NERF_CAPTURE"

SCENES=("printer_2", "cupboard")

for s in "${SCENES[@]}"; do
  compute-aabb \
    --dataset.dataset-path ${DATASET_FOLDER}/${s} \
    --save \
    --pcd.down-sample-voxel-size 0.01 \
    --pcd.max-points 25_000
    --pcd.skip-frames 1 \
    --pcd.down-sample-frames 10 \
    --extra 0.0 \
    --pcd.max-depth 4 \
    --no-gui
done
