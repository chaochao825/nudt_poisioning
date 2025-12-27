#!/bin/bash

# Base directory
BASE_DIR=$(dirname $(dirname $(readlink -f $0)))
cd $BASE_DIR

INPUT_DIR="./input"
DATA_DIR="$INPUT_DIR/data"
IMAGES_DIR="$INPUT_DIR/images"
mkdir -p "$DATA_DIR"
mkdir -p "$IMAGES_DIR"

CIFAR10_URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
TAR_FILE="$DATA_DIR/cifar-10-python.tar.gz"

if [ ! -d "$DATA_DIR/cifar-10-batches-py" ]; then
    echo "Downloading CIFAR-10..."
    if command -v curl >/dev/null 2>&1; then
        curl -L $CIFAR10_URL -o "$TAR_FILE"
    elif command -v wget >/dev/null 2>&1; then
        wget $CIFAR10_URL -O "$TAR_FILE"
    else
        echo "Error: curl or wget not found. Please install one of them."
        exit 1
    fi

    echo "Extracting CIFAR-10..."
    tar -xzf "$TAR_FILE" -C "$DATA_DIR"
    rm "$TAR_FILE"
fi

echo "Converting CIFAR-10 to image files..."
python3 <<EOF
import os
import pickle
import numpy as np
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data_dir = '$DATA_DIR/cifar-10-batches-py'
images_dir = '$IMAGES_DIR'
os.makedirs(images_dir, exist_ok=True)

meta = unpickle(os.path.join(data_dir, 'batches.meta'))
label_names = [t.decode('utf8') for t in meta[b'label_names']]

for i in range(1, 6):
    batch = unpickle(os.path.join(data_dir, f'data_batch_{i}'))
    for j in range(len(batch[b'data'])):
        img_data = batch[b'data'][j]
        img_label = batch[b'labels'][j]
        img_name = batch[b'filenames'][j].decode('utf8')
        
        img = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img)
        
        label_name = label_names[img_label]
        target_dir = os.path.join(images_dir, 'train', label_name)
        os.makedirs(target_dir, exist_ok=True)
        img.save(os.path.join(target_dir, img_name))

test_batch = unpickle(os.path.join(data_dir, 'test_batch'))
for j in range(len(test_batch[b'data'])):
    img_data = test_batch[b'data'][j]
    img_label = test_batch[b'labels'][j]
    img_name = test_batch[b'filenames'][j].decode('utf8')
    
    img = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
    img = Image.fromarray(img)
    
    label_name = label_names[img_label]
    target_dir = os.path.join(images_dir, 'test', label_name)
    os.makedirs(target_dir, exist_ok=True)
    img.save(os.path.join(target_dir, img_name))

print(f"CIFAR-10 images prepared at {images_dir}")
EOF
