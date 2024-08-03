#!/bin/bash

FILE=$1

# List of available datasets
AVAILABLE_DATASETS=("ae_photos" "apple2orange" "summer2winter_yosemite" "horse2zebra" "monet2photo" "cezanne2photo" "ukiyoe2photo" "vangogh2photo" "maps" "cityscapes" "facades" "iphone2dslr_flower")

# Check if the provided dataset name is in the list of available datasets
if [[ ! " ${AVAILABLE_DATASETS[@]} " =~ " ${FILE} " ]]; then
    echo "Available datasets are: ${AVAILABLE_DATASETS[@]}"
    exit 1
fi

URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip
ZIP_FILE=./$FILE.zip
TARGET_DIR=./$FILE

# Check if the dataset directory already exists
if [ -d "$TARGET_DIR" ]; then
    echo "Dataset $FILE already exists. Skipping download."
else
    wget -N $URL -O $ZIP_FILE
    if [ $? -ne 0 ]; then
        echo "Failed to download $URL"
        exit 1
    fi

    unzip $ZIP_FILE -d .
    if [ $? -ne 0 ]; then
        echo "Failed to unzip $ZIP_FILE"
        rm $ZIP_FILE
        exit 1
    fi
    rm $ZIP_FILE
fi

# Adapt to project expected directory hierarchy
mkdir -p "$TARGET_DIR/train" "$TARGET_DIR/test"
mv "$TARGET_DIR/trainA" "$TARGET_DIR/train/A"
mv "$TARGET_DIR/trainB" "$TARGET_DIR/train/B"
mv "$TARGET_DIR/testA" "$TARGET_DIR/test/A"
mv "$TARGET_DIR/testB" "$TARGET_DIR/test/B"

# Check if the moves were successful
if [ $? -ne 0 ]; then
    echo "Failed to organize dataset structure"
    exit 1
fi

echo "Dataset $FILE is ready."
