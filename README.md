# Wildlife Species Classifier (YOLOv5)

This repository contains the data processing and training pipeline for a two-stage wildlife camera trap classifier. It leverages MegaDetector for bounding box extraction and a custom-trained YOLOv5 classification model to identify the species within those crops.

## Project Setup
To get this pipeline running locally (e.g., in VS Code), you must clone the YOLOv5 repository and install its dependencies, as well as the data augmentation libraries used to prevent overfitting.

1. Clone the YOLOv5 repository into the root of this project:
   `git clone https://github.com/ultralytics/yolov5.git`
2. Install the core requirements:
   `pip install -r yolov5/requirements.txt`
3. Install additional libraries for dataset processing and augmentation:
   `pip install requests Pillow albumentations`

## Required Dataset Files
Before generating the dataset, you need to download two specific metadata and detection files from the LILA BC (Labeled Information Library of Alexandria: Biology and Conservation) repository:

1. **CCT Metadata JSON**: Download `caltech_camera_traps.json` directly from the LILA BC dataset page.
2. **MegaDetector Results JSON**: Download the `(MDv5a)` results file for the Caltech Camera Traps dataset from the LILA BC MegaDetector results directory.

Place both of these `.json` files in the root directory of this project.

## Caltech Camera Traps (CCT)
The Caltech Camera Traps (CCT) dataset is a heavily utilized benchmark in conservation AI, focusing on urban and suburban wildlife in Southern California. The dataset is inherently imbalanced (featuring massive amounts of opossums and cats, but very few coyotes or bobcats) and suffers from repeated background frames. A location-based train/val/test split is strictly enforced in this pipeline to ensure the model learns biological features rather than memorizing specific camera station backgrounds.

## Pipeline Scripts

### `dataset_creation.py`
**How to run:** `python dataset_creation.py`
**What it does:** Reads the CCT metadata and MegaDetector results, maps species categories, and applies a location-based split (70% train, 15% val, 15% test). It streams the raw images from LILA's Azure blob storage into memory, crops the animal using MD's bounding box coordinates, pads the image with black pixels to create a perfect square, and saves it into YOLOv5's required directory structure (`/train`, `/val`, `/test`).

### `count_species.py`
**How to run:** `python count_species.py`
**What it does:** Traverses the generated `wildlife_yolo_dataset` directory and calculates the total image count for every species across all splits. It outputs a formatted table sorted by frequency, allowing you to easily identify empty or extreme minority classes.

### `clean_dataset.py`
**How to run:** `python clean_dataset.py`
**What it does:** Addresses the extreme class imbalance in the CCT dataset. It calculates a global count for each species and permanently deletes the folders of any species that fall below the minimum threshold (e.g., 50 images). Most importantly, it syncs the folder structure across `train`, `val`, and `test` to ensure PyTorch does not crash during data loading.

### `fix_empty_folders.py`
**How to run:** `python fix_empty_folders.py`
**What it does:** Due to the location-based split, certain rare species might not appear in the validation or test stations, leaving their respective folders empty. PyTorch's `ImageFolder` strictly requires at least one image per class folder. This script finds empty folders and copies a single "donor" image from the train split into them to satisfy the data loader.

### `train_classifier.py`
**How to run:** `python train_classifier.py`
**What it does:** A Python wrapper for YOLOv5's classification training loop (`yolov5/classify/train.py`). It uses the `subprocess` module to execute the training command with predefined hyperparameters, ensuring reproducible experiments.
**Code Breakdown:**
* **Weights:** Uses `yolov5s-cls.pt` (Small model) for a balance of speed and baseline accuracy.
* **Dimensions & Epochs:** Trains for 50 epochs on 224x224 input tensors.
* **Dropout:** Explicitly sets `--dropout 0.3` (30%) to randomly zero out elements in the classification head, fighting background memorization.
* **Albumentations:** The training loop automatically detects the `albumentations` library if installed and applies aggressive real-time image augmentation (blur, contrast shifts, flipping) to close the domain gap.

## How to Evaluate the Training
Once the 50 epochs finish, you can evaluate the model's metrics:
1. **Validation Metrics:** Run `python yolov5/classify/val.py --weights wildlife_species_classifier/augmented_run_02/weights/best.pt --data wildlife_yolo_dataset` to evaluate the hold-out test set.
2. **Real-World Inference:** To prove the model isn't overfitting to the CCT backgrounds, test it on a brand-new image using: `python yolov5/classify/predict.py --weights <path_to_best.pt> --source test_image.jpg`.

## Issues with Training and Troubleshooting
**Problem: 100% Accuracy / Overfitting**
If your `top1_acc` reaches 1.0 (100%) early in training, the model is likely "cheating" by memorizing the background foliage of the camera traps instead of learning the animal features.
**Solution:**
When testing a 100% accuracy model on an out-of-distribution image (like a daytime photo of a coyote), the model's confidence will plummet. To fix this domain gap, you must enforce heavy data augmentation and regularization. Ensure `albumentations` is installed via pip so YOLOv5 applies it automatically, and pass the `--dropout` flag to your training runner to prevent the neural network from relying on specific background pixels.
