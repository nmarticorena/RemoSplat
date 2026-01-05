file="scripts/03_Benchmark/rmmi_one.py"

dimension=(true false) # 3D and 2D
active=(false) # Active and non active

for d in "${dimension[@]}"
do
  for a in "${active[@]}"
  do
    if [ "$d" == true ]; then
      DIMENSION_FLAG="--dimension"
    else
      DIMENSION_FLAG="--no-dimension"
    fi
    
    if [ "$a" == true ]; then
      ACTIVE_FLAG="--active"
    else
      ACTIVE_FLAG="--no-active"
    fi


    echo "Calling $file $DIMENSION_FLAG $ACTIVE_FLAG"
    python3 $file $DIMENSION_FLAG $ACTIVE_FLAG &
  done
done

wait
echo "Done"


