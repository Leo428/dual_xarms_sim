#!/bin/bash

SRC_DIR="/home/huzheyuan/dual_xarms/dual_xarms_sim/data"
DEST_HOST="babel-vscode"
DEST_DIR="/data/group_data/rl/dexterous_robot_data"

# Find directories matching *sim*hdf5* under SRC_DIR
for dir in "$SRC_DIR"/*sim*hdf5*; do
    if [ -d "$dir" ]; then
        echo "Transferring $dir to $DEST_HOST:$DEST_DIR..."
        rsync -avz --info=progress2 "$dir" "$DEST_HOST:$DEST_DIR" || {
            echo "Error transferring $dir. Skipping..."
            continue
        }
    fi
done

echo "All matching folders uploaded."
