# Real world scans

The idea here is first to load a real world scene with some target poses

## Step 1:
Compute the TSDF of the scene, so I can Load them into blender

```bash
python ./deps/nerf_tools/nerf_tools/scripts/tsdf_intergration.py --dataset.dataset-path $NERF_CAPTURE/printer_2/ --name printer_2 --depth-trunc 5
```

This command save the generated mesh on results/meshes/printer_2.ply

## Step 2: Load in blender

We create a .blend file based on a previous scene

Then we move the mesh in order to align with the real world, the script needs to take this pose and we stored under the name of: T_WG -> From world to gaussian

Experiment file:


### Blender script

```py
import bpy
import json
D = bpy.data
import numpy
scene = bpy.context.scene


def get_poses():
    poses = []

    poses_ids = [i.name for i in D.collections["TargetPoses"].objects]
    poses_ids = [i for i in poses_ids if "target" in i]
    poses = [D.objects[i].matrix_world for i in poses_ids]
    poses = [numpy.array(p).tolist() for p in poses]
    return poses

dataset = {}

dataset["base"] = numpy.array(D.objects["Robot"].matrix_world).tolist()
dataset["poses"] = get_poses()
dataset["scene"] = {"T_WG": numpy.array(D.objects["printer_2"].matrix_world).tolist()}

print(dataset)

with open(f"/home/nmarticorena/Documents/papers/remo_splat/printer_2.json", "w") as f:
    json.dump(dataset, f, indent = 4)

scene.frame_current = 0
print(dataset)
```

##

