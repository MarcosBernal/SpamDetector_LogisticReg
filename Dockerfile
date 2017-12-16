# Spyder3, python3 and Pyspark
#
# VERSION               0.1

FROM ubuntu:xenial


# Installing python3 and other important packages
RUN    apt-get -y update
RUN    apt-get -y install python3 python3-dev python3-numpy python3-scipy python3-matplotlib ipython3 ipython3-notebook python3-pandas python3-sympy python3-nose
RUN    apt-get -y install libjs-jquery libjs-mathjax python3-pyqt4 tortoisehg gitk ipython3-qtconsole python3-pep8 pyflakes pylint python3-jedi python3-psutil python3-sphinx
RUN    apt-get -y install python3-pip

# Installing oracle java automatically (needed from pyspark)
RUN    apt-get -y install software-properties-common
RUN    add-apt-repository -y ppa:webupd8team/java
RUN    echo debconf shared/accepted-oracle-license-v1-1 select true | debconf-set-selections
RUN    echo debconf shared/accepted-oracle-license-v1-1 seen true | debconf-set-selections
RUN    apt-get update
RUN    apt-get -y install oracle-java8-installer

# Installing spyder3
RUN    apt-get -y install spyder3

# Cleaning apt-get and freeing up space
RUN    apt-get clean && apt-get purge

# Upgrading pip to last version and installing pyspark, spyder3 and its dependencies
RUN    pip3 install --upgrade pip
RUN    pip3 install pyspark
RUN    pip3 install -U pyspark
RUN    pip3 install -U spyder

# Creating workspace
RUN    mkdir /home/dev
RUN    cd /home/dev
ADD    start-spyder.sh /start-spyder.sh
RUN    chmod +x /start-spyder.sh

CMD /bin/bash
