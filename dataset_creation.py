import os
import json
import random
import requests
from io import BytesIO
from PIL import Image

def setup_yolo_directories(base_dir, categories, splits=["train", "val", "test"]):
    """Creates the YOLOv5-cls directory structure."""
    for split in splits:
        for cat in categories.values():
            os.makedirs(os.path.join(base_dir, split, cat), exist_ok=True)

def crop_and_save_image(image_url, bbox_norm, save_path, padding=0.05):
    """Downloads an image into memory, crops it, pads to a square, and saves it."""
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        
        img_w, img_h = img.size
        x_norm, y_norm, w_norm, h_norm = bbox_norm
        
        # Convert normalized coordinates to absolute pixels
        x_abs = x_norm * img_w
        y_abs = y_norm * img_h
        w_abs = w_norm * img_w
        h_abs = h_norm * img_h
        
        # Apply padding buffer
        pad_w = w_abs * padding
        pad_h = h_abs * padding
        
        left = max(0, x_abs - pad_w)
        top = max(0, y_abs - pad_h)
        right = min(img_w, x_abs + w_abs + pad_w)
        bottom = min(img_h, y_abs + h_abs + pad_h)
        
        cropped_img = img.crop((left, top, right, bottom))
        
        # --- NEW: Pad to square with black pixels ---
        crop_w, crop_h = cropped_img.size
        max_dim = max(crop_w, crop_h)
        
        # Create a new black square image
        square_img = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
        
        # Calculate paste coordinates to center the crop
        offset_x = (max_dim - crop_w) // 2
        offset_y = (max_dim - crop_h) // 2
        
        # Paste the cropped image onto the black square
        square_img.paste(cropped_img, (offset_x, offset_y))
        
        # Save the final square image
        square_img.save(save_path, "JPEG")
        return True
    except Exception as e:
        print(f"Failed to process {image_url}: {e}")
        return False

def build_dataset(cct_metadata_path, md_results_path, output_dir, max_images=5000):
    # Azure Base URL for CCT images
    CCT_BLOB_BASE = "https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images/"
    
    # 1. Load Metadata
    print("Loading CCT Metadata and MegaDetector results...")
    with open(cct_metadata_path, 'r') as f:
        cct_data = json.load(f)
        
    with open(md_results_path, 'r') as f:
        md_data = json.load(f)

    # 2. Map categories and images
    categories = {cat['id']: cat['name'].replace(" ", "_") for cat in cct_data['categories']}
    image_meta = {img['id']: img for img in cct_data['images']}
    
    setup_yolo_directories(output_dir, categories)

    # 3. Create Location-Based Split
    unique_locations = list(set([img['location'] for img in cct_data['images']]))
    random.seed(42)
    random.shuffle(unique_locations)
    
    train_split = int(len(unique_locations) * 0.7)
    val_split = int(len(unique_locations) * 0.85)
    
    locations_train = set(unique_locations[:train_split])
    locations_val = set(unique_locations[train_split:val_split])
    
    def get_split(loc_id):
        if loc_id in locations_train: return "train"
        if loc_id in locations_val: return "val"
        return "test"

    # 4. Map MegaDetector detections to ground truth annotations
    print("Processing images and generating crops...")
    processed_count = 0
    
    # Map MD results by image ID for fast lookup (stripping the .jpg extension)
    md_lookup = {img['file'].replace('.jpg', ''): img.get('detections', []) for img in md_data['images']}

    for ann in cct_data['annotations']:
        if processed_count >= max_images:
            break
            
        img_id = ann['image_id']
        cat_id = ann['category_id']
        
        # Skip empty images or classes we don't care about
        if cat_id not in categories or categories[cat_id] == "empty":
            continue
            
        img_info = image_meta.get(img_id)
        if not img_info:
            continue
            
        detections = md_lookup.get(img_id, [])
        if not detections:
            continue
            
        # Get the highest confidence animal detection (category '1' in MD)
        animal_detections = [d for d in detections if d['category'] == '1']
        if not animal_detections:
            continue
            
        best_detection = max(animal_detections, key=lambda x: x['conf'])
        
        # Only use high confidence MD crops to train the species classifier
        if best_detection['conf'] < 0.8:
            continue
            
        split = get_split(img_info['location'])
        species_name = categories[cat_id]
        save_filename = f"{img_id}_crop.jpg"
        save_path = os.path.join(output_dir, split, species_name, save_filename)
        
        # Build URL and execute crop
        image_url = CCT_BLOB_BASE + img_info['file_name']
        if crop_and_save_image(image_url, best_detection['bbox'], save_path):
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} images...")

    print(f"Finished! Generated {processed_count} bounding box crops.")

if __name__ == "__main__":
    CCT_METADATA = 'caltech_images_20210113.json'
    MD_RESULTS = 'caltech-camera-traps_mdv5a.0.0_results.json'
    OUTPUT_DIR = './wildlife_yolo_dataset'
    
    print("Starting dataset build...")
    build_dataset(CCT_METADATA, MD_RESULTS, OUTPUT_DIR)
