# Spyder3(python3) and Pyspark
#
# VERSION               0.3

FROM timcera/spyder-desktop-ubuntu:latest

# Installing oracle java automatically (needed from pyspark)
ENV    TERM xterm
RUN    apt-get update
RUN    apt-get install -y apt-utils dialog
RUN    apt-get -y install software-properties-common
RUN    add-apt-repository -y ppa:webupd8team/java
RUN    echo debconf shared/accepted-oracle-license-v1-1 select true | debconf-set-selections
RUN    echo debconf shared/accepted-oracle-license-v1-1 seen true | debconf-set-selections
RUN    apt-get update
RUN    apt-get -y install oracle-java8-installer

# Upgrading pip to last version and installing pyspark, spyder3 and its dependencies
RUN    pip3 install --upgrade pip
RUN    pip3 install pyspark
RUN    pip3 install -U pyspark

# Creating workspace
RUN    mkdir /home/dev
RUN    cd /home/dev

# Create env variable for using python3 with spark
ENV    PYSPARK_PYTHON=/usr/bin/python3
ENV    PYSPARK_DRIVER_PYTHON=ipython3

ENV    USER root


CMD /bin/bash