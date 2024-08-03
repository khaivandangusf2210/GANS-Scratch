#!/bin/bash

FILE=$1
URL=https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/$FILE.tar.gz
TAR_FILE=./$FILE.tar.gz
TARGET_DIR=./$FILE/

# Check if the dataset directory already exists
if [ -d "$TARGET_DIR" ]; then
    echo "Dataset $FILE already exists. Skipping download."
else
    # Download the dataset
    wget -N $URL -O $TAR_FILE
    if [ $? -ne 0 ]; then
        echo "Failed to download $URL"
        exit 1
    fi

    # Create target directory
    mkdir -p $TARGET_DIR

    # Extract the dataset
    tar -zxvf $TAR_FILE -C ./
    if [ $? -ne 0 ]; then
        echo "Failed to extract $TAR_FILE"
        rm $TAR_FILE
        exit 1
    fi

    # Remove the tar file after successful extraction
    rm $TAR_FILE
fi

echo "Dataset $FILE is ready."
