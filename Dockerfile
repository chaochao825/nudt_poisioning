FROM python:3.10-slim

WORKDIR /workspace
COPY . /workspace

RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1

CMD ["bash"]

