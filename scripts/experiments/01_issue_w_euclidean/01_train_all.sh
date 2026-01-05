folder_path=/media/nmarticorena/DATA/gsplat_synthetic_dataset/bookshelf_cage/bookshelf_cage_
i=5

CUDA_VISIBLE_DEVICES=0 python scripts/training/simple_trainer_2dgs.py \
      --data_dir $folder_path$(printf "%04d" $i)\
      --data_factor 1 \
      --result_dir ~/Documents/papers/remo_splat/data/gsplat_2D/01-euclidean_test/random_05 \
      --no-normalize-world-space \
      --depth-loss \
      --depth-lambda 0.1 \
      --normal-loss \
      --init-opa 0.01 \
      --random-bkgd \
      --max-steps 10000 \
      --save-steps 10000 \
      --init-type random \
      --init_extent 1.0 \
      --disable-viewer


CUDA_VISIBLE_DEVICES=0 python scripts/training/simple_trainer_2dgs.py \
      --data_dir $folder_path$(printf "%04d" $i)\
      --data_factor 1 \
      --result_dir ~/Documents/papers/remo_splat/data/gsplat_2D/01-euclidean_test/sfm_05 \
      --no-normalize-world-space \
      --depth-loss \
      --depth-lambda 0.1 \
      --normal-loss \
      --init-opa 0.01 \
      --random-bkgd \
      --max-steps 10000 \
      --save-steps 10000 \
      --init-type 'sfm' \
      --init_extent 1.0  \
      --disable-viewer
