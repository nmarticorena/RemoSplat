SCRIPT_PATH="scripts/rendering/gsplat_render.py"
DATASET_FOLDER="$NERF_STANDARD"

SCENES=("bookshelf" "counter" "industrial_a" "tunnel" "tunnel_sdf")

for s in "${SCENES[@]}"; do
  python $SCRIPT_PATH \
    --no-hist\
    --no-ply\
    --dataset_folder $DATASET_FOLDER \
    --scene $s \
    --exp_name $s \
    --no-high \
    --is_3D
done

