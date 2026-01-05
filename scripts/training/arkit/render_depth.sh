SCRIPT_PATH="scripts/rendering/gsplat_render.py"
DATASET_FOLDER="$NERF_CAPTURE"

SCENES=("printer_2", "cuboard")

for s in "${SCENES[@]}"; do
  python $SCRIPT_PATH \
    --no-hist\
    --no-ply\
    --dataset_folder $DATASET_FOLDER \
    --scene $s \
    --exp_name $s \
    --high \
    --is_2D \
    --last
done

