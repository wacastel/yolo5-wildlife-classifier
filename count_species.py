import os
from collections import defaultdict

def count_species(dataset_dir):
    species_counts = defaultdict(int)

    if not os.path.exists(dataset_dir):
        print(f"Error: Directory '{dataset_dir}' not found.")
        return
        
    # Traverse the YOLO splits
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        # Iterate through each species folder
        for species in os.listdir(split_dir):
            species_dir = os.path.join(split_dir, species)
            
            # Verify it is a directory
            if os.path.isdir(species_dir):
                # Count valid image files inside the species directory
                num_images = len([f for f in os.listdir(species_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                species_counts[species] += num_images
                
    # Format and display the results
    print(f"{'Species':<30} | {'Image Count':<10}")
    print("-" * 45)

    total_images = 0
    # Sort the dictionary by count in descending order
    for species, count in sorted(species_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{species:<30} | {count:<10}")
        total_images += count
        
    print("-" * 45)
    print(f"{'Total':<30} | {total_images:<10}")

if __name__ == "__main__":
    count_species('./wildlife_yolo_dataset')
