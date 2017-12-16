Based on the docker repo https://hub.docker.com/r/jupyter/pyspark-notebook/

## Requirements

  - docker
  - 6GB of space in the hard drive

## Usage

### Spyder3
  - Move to the folder where the repo is placed `i.e. cd SpamDetector_LogisticReg`
  - Build the custom docker file `sudo docker build -t spyder-jupyter-pyspark .`
  - Exec the script file `./spyder_spark.sh`

### Jupyther
  - The process starts with the script `pyspark-notebook.sh`
  - Then copy the url to your browser and start working
  - The files are saved in the `notebook` folder
