#!/bin/sh
# Need sudo access to use docker
echo "sudo access required"
sudo echo ""

# Moving to the directory of the script
cd $(pwd)/$(dirname $0)

xhost local:root

# -v mount the folder notebook folder in /home/dev
# --rm will remove the container as soon as it ends
sudo docker run --rm \
    -i -t \
    -v $(pwd)/$(dirname $0)/notebook:/home/dev \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -e DISPLAY=unix$DISPLAY \
    timcera-spyder-pyspark spyder3 -w /home/dev