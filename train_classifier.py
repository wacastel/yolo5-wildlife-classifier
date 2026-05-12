import subprocess
import sys
import os

def run_yolo_training():
    """
    Executes the YOLOv5 classification training loop with specified hyperparameters.
    Assumes the 'yolov5' repository is cloned in the same root directory.
    """
    
    # Define paths
    yolov5_train_script = os.path.join("yolov5", "classify", "train.py")
    dataset_path = os.path.abspath("wildlife_yolo_dataset")
    
    # Check if YOLOv5 exists
    if not os.path.exists(yolov5_train_script):
        print(f"Error: Could not find {yolov5_train_script}. Make sure the yolov5 repo is cloned here.")
        sys.exit(1)

    # Hyperparameters and Configuration
    model_weights = "yolov5s-cls.pt"  # Small model for fast baseline
    epochs = "50"                     # Total training epochs
    img_size = "224"                  # YOLOv5 classification standard input size
    batch_size = "64"                 # Adjust down to 32 or 16 if your GPU runs out of memory
    project_dir = "wildlife_species_classifier"
    experiment_name = "baseline_run_01"

    # Construct the command
    command = [
        sys.executable, yolov5_train_script,
        "--model", model_weights,
        "--data", dataset_path,
        "--epochs", epochs,
        "--img", img_size,
        "--batch-size", batch_size,
        "--project", project_dir,
        "--name", experiment_name
    ]

    print("\nStarting YOLOv5 Training...")
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_weights}")
    print("-" * 40)

    # Execute the training loop and stream the output directly to your VS Code terminal
    try:
        process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
        process.communicate()
        
        if process.returncode == 0:
            print("\nTraining completed successfully!")
            print(f"Results saved to: {project_dir}/{experiment_name}")
        else:
            print(f"\nTraining stopped or failed with return code {process.returncode}")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Stopping gracefully...")
        process.kill()

if __name__ == "__main__":
    run_yolo_training()
