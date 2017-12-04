#ª /bin/bash
# jupyter/all-spark-notebook
echo "sudo access required"
sudo echo ""
echo "Copy the following url in a browser. To finish the docker process, use ^C^C";
sudo docker run -it --rm -p 8888:8888 \
    -v $(pwd)/$(dirname $0)/notebook:/home/jovyan \
    jupyter/pyspark-notebook start-notebook.sh | grep http://localhost:8888/?token=
