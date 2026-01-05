# Training script for Blender scenes,
# It requires to run the blender_preprocessing.sh first
#

SCRIPT_PATH="scripts/training/simple_trainer.py"

DATASET_FOLDER="$NERF_CAPTURE"
RESULT_FOLDER="$HOME/Documents/papers/remo_splat/data/gsplat"

SCENES=("printer_2")

for s in "${SCENES[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH mcmc \
    --data_dir ${DATASET_FOLDER}/${s}  --data_factor 1 \
    --result_dir $RESULT_FOLDER/${s} \
    --no-normalize-world-space \
    --depth-loss \
    --depth-lambda 0.01 \
    --opacity-reg 0.01 \
    --scale-reg 0.01 \
    --disable-viewer \
    --port 8081 \
    --strategy.cap-max 100000 
done




