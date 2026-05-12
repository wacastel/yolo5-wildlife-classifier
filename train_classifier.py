import subprocess
import sys
import os

def run_augmented_yolo_training():
    """
    Executes the YOLOv5 classification training loop with Albumentations
    and Dropout enabled to prevent background overfitting.
    """
    # Define paths
    yolov5_train_script = os.path.join("yolov5", "classify", "train.py")
    dataset_path = os.path.abspath("wildlife_yolo_dataset")
    
    # Check if YOLOv5 exists
    if not os.path.exists(yolov5_train_script):
        print(f"Error: Could not find {yolov5_train_script}. Make sure the yolov5 repo is cloned here.")
        sys.exit(1)

    # Hyperparameters and Configuration
    model_weights = "yolov5s-cls.pt"  
    epochs = "50"                     
    img_size = "224"                  
    batch_size = "64"                 
    project_dir = "wildlife_species_classifier"
    experiment_name = "augmented_run_02"
    dropout_rate = "0.3"  # 30% dropout to force feature learning

    # Construct the command
    command = [
        sys.executable, yolov5_train_script,
        "--model", model_weights,
        "--data", dataset_path,
        "--epochs", epochs,
        "--img", img_size,
        "--batch-size", batch_size,
        "--project", project_dir,
        "--name", experiment_name,
        "--dropout", dropout_rate
    ]

    print("\nStarting YOLOv5 Augmented Training...")
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_weights}")
    print(f"Dropout Rate: {dropout_rate}")
    print("-" * 40)

    # Execute the training loop and stream the output directly to your VS Code terminal
    try:
        process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
        process.communicate()
        
        if process.returncode == 0:
            print("\nAugmented training completed successfully!")
            print(f"Results saved to: {project_dir}/{experiment_name}")
        else:
            print(f"\nTraining stopped or failed with return code {process.returncode}")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Stopping gracefully...")
        process.kill()

if __name__ == "__main__":
    run_augmented_yolo_training()
