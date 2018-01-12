Based on the docker repo https://hub.docker.com/r/jupyter/pyspark-notebook/

## Requirements

  - docker
  - 6GB of space in the hard drive

## Usage

Move to the folder where the repo is placed `i.e. cd SpamDetector_LogisticReg`

### Spyder3

  - Build the custom docker file `sudo docker build -t timcera-spyder-pyspark .`
  - Exec the script file `./spyder_spark.sh`
  - The files are saved in the `notebook` folder (arg `-v` of docker)

### Jupyther
  - The process starts with the script `pyspark-notebook.sh`
  - Then copy the url to your browser and start working
  - The files are saved in the `notebook` folder (arg `-v` of docker)
  - To end the program use `Ctr+C` and then `Ctrl+C` 
