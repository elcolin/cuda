FROM nvidia/cuda:12.2.0-devel-ubuntu20.04 as base

RUN apt-mark showhold
RUN apt update -y && apt upgrade -y && apt install -y wget dpkg

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
RUN mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.2-535.104.05-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.2-535.104.05-1_amd64.deb
RUN cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update
RUN apt-get -y install cuda
COPY config.sh /app

FROM base as build

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

CMD ["./config.sh"]