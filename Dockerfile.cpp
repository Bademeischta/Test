FROM ubuntu:22.04
RUN apt-get update && apt-get install -y build-essential cmake git && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY superengine /src/superengine
WORKDIR /src/superengine
RUN cmake -B build -S . && cmake --build build -j
CMD ["./build/test_movegen"]
