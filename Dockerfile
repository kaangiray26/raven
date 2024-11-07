# A simple Dockerfile to install raven on a ubuntu 24.04 image
FROM nvidia/cuda:12.6.2-devel-ubuntu24.04
WORKDIR /

# Copy files
COPY lib /lib
COPY app.py /
COPY requirements.txt /

# Install dependencies
RUN apt-get -y update && \
    apt-get -y install --no-install-recommends python3-venv && \
    python3 -m venv env && \
    env/bin/pip3 install --upgrade pip && \
    env/bin/pip3 install --no-cache-dir -r requirements.txt && \
    env/bin/python3 -m spacy download en_core_web_sm

# Expose port
EXPOSE 8000
ENTRYPOINT ["env/bin/python3", "-u", "app.py"]