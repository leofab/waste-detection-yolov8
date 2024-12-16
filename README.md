---

# Waste Detection with YOLO

This project uses the (***YOLOv8***)[https://github.com/ultralytics/ultralytics] architecture to detect objects in images related to waste detection from (***Arkadiy Serezhkin dataset***)[https://www.kaggle.com/datasets/arkadiyhacks/drinking-waste-classification]. Below is a detailed description of the steps involved in dataset preparation, model training, and results evaluation.

## Installation

First, install the `ultralytics` package to use YOLO tools:

```sh
!pip install ultralytics
```

## Dataset Preparation

The following script divides image data into training, testing, and validation sets. It expects all images and annotation `.txt` files to be in a specific input directory.

```sh
#!/bin/bash

# Input directory containing all .jpg and .txt files
input_dir="/path/to/image_folder"

# Output directories
train_images_dir="/path/to/waste-detection/train/images"
train_labels_dir="/path/to/waste-detection/train/labels"
test_images_dir="/path/to/waste-detection/test/images"
test_labels_dir="/path/to/waste-detection/test/labels"
val_images_dir="/path/to/waste-detection/val/images"
val_labels_dir="/path/to/waste-detection/val/labels"

# Create output directories if they do not exist
mkdir -p "$train_images_dir" "$train_labels_dir"
mkdir -p "$test_images_dir" "$test_labels_dir"
mkdir -p "$val_images_dir" "$val_labels_dir"

# Get all .jpg files from the input directory
jpg_files=("$input_dir"/*.jpg)

# Total number of items
total_items=${#jpg_files[@]}

# Calculate the number of items for each split
train_count=$((total_items * 70 / 100))  # 70% for training
test_count=$((total_items * 15 / 100))   # 15% for testing
val_count=$((total_items - train_count - test_count)) # Remaining 15% for validation

# Shuffle the list of .jpg files
shuffled_files=($(shuf -e "${jpg_files[@]}"))

# Helper function to copy files
copy_files() {
    local start_index=$1
    local end_index=$2
    local images_dir=$3
    local labels_dir=$4

    for ((i = start_index; i < end_index; i++)); do
        jpg_file="${shuffled_files[$i]}"
        txt_file="${jpg_file%.jpg}.txt"  # Replace .jpg with .txt

        # Copy the .jpg file to the images directory
        cp "$jpg_file" "$images_dir/"

        # Copy the .txt file to the labels directory if it exists
        if [[ -f "$txt_file" ]]; then
            cp "$txt_file" "$labels_dir/"
        fi
    done
}

# Copy files for the training set
copy_files 0 "$train_count" "$train_images_dir" "$train_labels_dir"

# Copy files for the testing set
copy_files "$train_count" $((train_count + test_count)) "$test_images_dir" "$test_labels_dir"

# Copy files for the validation set
copy_files $((train_count + test_count)) "$total_items" "$val_images_dir" "$val_labels_dir"

# Notify the user
echo "Dataset splitting and file copying completed successfully!"
echo "Training set: $train_count images"
echo "Testing set: $test_count images"
echo "Validation set: $val_count images"
```

## Model Training

Train the YOLOv8 model with the training data:

```sh
!yolo task=detect mode=train model=yolov8m.pt data=/content/waste-detection/data.yaml epochs=20 imgsz=640
```

## Results Evaluation

After training, visualize the results:

```python
from IPython.display import Image, display

# Display the confusion matrix
Image(filename=f"/content/runs/detect/train/confusion_matrix.png", width=1800)

# Display training results
Image(filename=f"/content/runs/detect/train/results.png", width=1800)
```

## Model Validation

Validate the trained model:

```sh
!yolo task=detect mode=val model=/content/runs/detect/train/weights/best.pt data=/content/waste-detection/data.yaml
```

## Predictions with the Trained Model

Run predictions using the trained model on new test images:

```sh
!yolo task=detect mode=predict model=/content/runs/detect/train/weights/best.pt conf=0.5 source=/content/waste-detection/test/images
```

## Displaying Prediction Images

To display 20 random images from the prediction results:

```python
import glob
import random
from IPython.display import Image, display

# Get all paths of prediction images
image_paths = glob.glob('/content/runs/detect/predict/*.jpg')

# Select 20 random images
random_image_paths = random.sample(image_paths, 20)

# Display the 20 random images
for image_path in random_image_paths:
    display(Image(filename=image_path, height=600))
    print('\n')
```

---