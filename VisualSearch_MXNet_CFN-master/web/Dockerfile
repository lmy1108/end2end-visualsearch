FROM ubuntu:latest
LABEL maintainer="gaonkarr@amazon.com"

RUN apt-get update -y && apt-get install -y apt-transport-https curl ca-certificates wget unzip python-pip python-dev build-essential && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*

COPY . /web
WORKDIR /web

ENTRYPOINT ["python","-m", "SimpleHTTPServer"]

EXPOSE 8000



