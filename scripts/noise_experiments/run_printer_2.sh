# Get the initialization

python deps/nerf_tools/scripts/compute_aabb.py --save \
    --pcd.down-sample-voxel-size 0.01 \
    --pcd.max_points 50000 \
    --pcd.skip-frames 1 \
    --pcd.down-sample-frames 5 \
    --extra 0.0 \
    --pcd.max-depth 4 \
    --add-points 1000 \
    --no-gui \
    --dataset.dataset-path $NERF_CAPTURE/printer_2_noise


# Train
CUDA_VISIBLE_DEVICES=0 python scripts/training/cleaner_trainer_2dgs.py default \
    --data_dir $NERF_CAPTURE/printer_2_noise --data_factor 1 \
    --result_dir ~/Documents/papers/remo_splat/data/gsplat_2D/printer_2_noise \
    --no-normalize-world-space --depth-loss --max-steps 2000 --save-steps 1 1000 2000  --batch-size 1 --init_type sfm --scale_reg 0 \
    --disable-viewer

python3 scripts/99_PaperFigures/03_depth_vs_euclidean_noise.py --scene printer_2_noise --exp-name printer_2_noise


python3 scripts/09_RealWorld/test_full_run.py --robot-name curobo --sensor euclidean_less --controller.collision-cost "w_avg" --controller.collision-gain 1 --gsplat.scene printer_2_noise
python3 scripts/09_RealWorld/test_full_run.py --robot-name curobo --sensor depth --controller.collision-cost "w_avg" --controller.collision-gain 1 --gsplat.scene printer_2_noise
