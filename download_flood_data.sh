#!/bin/bash
# Script to download Sen1Floods11 dataset

echo "=== Sen1Floods11 Data Download Script ==="
echo ""

# Check if gsutil is installed
if ! command -v gsutil &> /dev/null
then
    echo "gsutil not found. Installing Google Cloud SDK..."
    echo ""
    echo "Please run the following command to install:"
    echo "  pip install gsutil"
    echo ""
    echo "Or install full Google Cloud SDK:"
    echo "  https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Create data directory
mkdir -p datasets/floods/sen1floods11

echo "Downloading Sen1Floods11 dataset (~14 GB)..."
echo "This will take 30-60 minutes depending on your connection"
echo ""

# Download the full dataset
gsutil -m rsync -r gs://sen1floods11 datasets/floods/sen1floods11

echo ""
echo "Download complete! Dataset saved to: datasets/floods/sen1floods11"
echo ""
echo "Dataset structure:"
ls -lh datasets/floods/sen1floods11
