import os
import shutil
from collections import defaultdict

def clean_dataset_global(dataset_dir, min_images=50):
    splits = ['train', 'val', 'test']
    species_global_counts = defaultdict(int)
    
    # 1. Calculate global image counts for each species
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        for species in os.listdir(split_dir):
            species_dir = os.path.join(split_dir, species)
            if os.path.isdir(species_dir):
                images = [f for f in os.listdir(species_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                species_global_counts[species] += len(images)
                
    # 2. Determine which species pass the global threshold
    species_to_keep = {species for species, count in species_global_counts.items() if count >= min_images}
    species_to_remove = set(species_global_counts.keys()) - species_to_keep
    
    print(f"Keeping {len(species_to_keep)} species. Removing {len(species_to_remove)} minority species.")
    for s in species_to_remove:
        print(f" - Removing '{s}' (Global count: {species_global_counts[s]})")
        
    # 3. Apply the cleaning globally
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        for species in os.listdir(split_dir):
            species_dir = os.path.join(split_dir, species)
            if not os.path.isdir(species_dir):
                continue
                
            # Remove species folder if it failed the global threshold
            if species in species_to_remove:
                shutil.rmtree(species_dir)
                
    # 4. Sync directory structures (PyTorch requirement)
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        if os.path.exists(split_dir):
            for species in species_to_keep:
                # Ensures an empty class folder exists even if a specific split has 0 images for it
                os.makedirs(os.path.join(split_dir, species), exist_ok=True)
                
    print("\nDataset cleaning complete! All splits now have synced directory structures.")

if __name__ == "__main__":
    clean_dataset_global('./wildlife_yolo_dataset', min_images=50)
