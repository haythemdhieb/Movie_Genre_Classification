FROM ubuntu:18.04 AS develop

# RUN apk add --update python make g++ python-dev py-pip && rm -rf /var/cache/apk/*
RUN apt-get update
RUN apt-get install -y libmysqlclient-dev wget make g++ curl libspatialindex-dev  python3-pip  python3.7 && rm -rf /var/lib/apt/lists/*
RUN python3.7 -m pip install --upgrade pip setuptools
# Set working directory
WORKDIR /Movie_Genre_Classification
COPY . .
# install requirements
RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip install -r requirements.txt

# export pythonpath
ENV PYTHONPATH "/Movie_Genre_Classification"
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 40
