#! /bin/bash
# Need sudo access to use docker
echo "sudo access required"
sudo echo ""

# Moving to the directory of the script
cd $(pwd)/$(dirname $0)

xhost local:root

sudo docker run --rm \
    -i -t \
    -u root \
    -v $(pwd)/$(dirname $0)/notebook:/home/dev \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -e DISPLAY=unix$DISPLAY \
    spyder-jupyter-pyspark /bin/bash #spyder3 -w /home/dev
