import os
import shutil

def fix_empty_folders(dataset_dir):
    splits = ['train', 'val', 'test']
    
    # Get the list of all valid species from the train directory
    train_dir = os.path.join(dataset_dir, 'train')
    species_list = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    for species in species_list:
        # Find a donor image for this species
        donor_img = None
        for split in splits:
            species_dir = os.path.join(dataset_dir, split, species)
            if os.path.exists(species_dir):
                images = [f for f in os.listdir(species_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    donor_img = os.path.join(species_dir, images[0])
                    break
        
        if not donor_img:
            continue
            
        # Check all splits and copy the donor image if the folder is empty
        for split in splits:
            species_dir = os.path.join(dataset_dir, split, species)
            os.makedirs(species_dir, exist_ok=True) # Ensure the directory exists
            
            images = [f for f in os.listdir(species_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(images) == 0:
                fix_filename = "dummy_fix.jpg"
                print(f"Fixing empty folder: {split}/{species} -> Copied 1 image to satisfy PyTorch.")
                shutil.copy(donor_img, os.path.join(species_dir, fix_filename))

if __name__ == "__main__":
    fix_empty_folders('./wildlife_yolo_dataset')
