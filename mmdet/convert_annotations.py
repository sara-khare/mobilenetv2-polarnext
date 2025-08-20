import os
import cv2
import json
import numpy as np
from PIL import Image
from glob import glob

def convert_masks_to_bboxes(mask_path, class_map):
    """
    Converts a single semantic segmentation mask image into bounding boxes.

    Args:
        mask_path (str): The file path to the semantic segmentation mask image.
        class_map (dict): A dictionary mapping pixel values to class names.
                          e.g., {0: 'background', 1: 'tree', 2: 'road', ...}

    Returns:
        list: A list of dictionaries, where each dictionary represents a
              bounding box annotation for a detected object.
              e.g., [{'bbox': [x, y, w, h], 'class_name': 'car'}]
    """
    if not os.path.exists(mask_path):
        print(f"Mask file not found: {mask_path}")
        return []

    # Use PIL to open the image and explicitly convert it to a single-channel grayscale ('L')
    mask = np.array(Image.open(mask_path).convert('L'))
    unique_pixels = np.unique(mask)
    annotations = []

    for pixel_value in unique_pixels:
        if pixel_value == 0:  # Skip background
            continue

        class_mask = np.zeros_like(mask, dtype=np.uint8)
        class_mask[mask == pixel_value] = 255
        
        # Use cv2 to find contours on the binary mask
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Check for a minimum area to filter out noise
            if cv2.contourArea(contour) < 20:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            class_name = class_map.get(pixel_value, 'unknown')
            
            bbox_annotation = {
                'bbox': [x, y, w, h], # COCO format is [x_min, y_min, width, height]
                'class_name': class_name,
                # The fix is here: explicitly cast the NumPy uint8 type to a standard Python integer.
                'class_id': int(pixel_value) 
            }
            annotations.append(bbox_annotation)
            
    return annotations

# --- Main execution block ---
if __name__ == '__main__':
    # Define a simple class map based on the UAVid pixel-to-class mapping
    uavid_class_map = {
        0: 'background',
        1: 'building',
        2: 'clutter',
        3: 'tree',
        4: 'low vegetation',
        5: 'car',
        6: 'pavement',
        7: 'road',
    }

    # Define the base directory for your dataset
    dataset_base_path = os.path.expanduser('~/dataset/UAVid/uavid_val')
    
    # Initialize the COCO-style data structure
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': [],
    }

    # Populate the categories list
    for class_id, class_name in uavid_class_map.items():
        if class_id != 0: # Skip background
            coco_data['categories'].append({
                'id': class_id,
                'name': class_name,
                'supercategory': 'uavid'
            })
    
    image_id_counter = 0
    annotation_id_counter = 0

    # Get a list of all 'seq' directories
    seq_dirs = sorted(glob(os.path.join(dataset_base_path, 'seq*')))

    # Loop through all the sequence directories
    for seq_dir in seq_dirs:
        print(f"Processing sequence directory: {os.path.basename(seq_dir)}")
        
        # Get a list of all image and label paths for the current sequence
        image_files = sorted(glob(os.path.join(seq_dir, 'Images', '*.png')))
        label_files = sorted(glob(os.path.join(seq_dir, 'Labels', '*.png')))
        
        # Ensure image and label counts match
        if len(image_files) != len(label_files):
            print(f"Warning: Mismatch between images ({len(image_files)}) and labels ({len(label_files)}) in {seq_dir}. Skipping.")
            continue
            
        # Loop through all image/label pairs
        for image_path, label_path in zip(image_files, label_files):
            # Read image dimensions to add to COCO data
            img = Image.open(image_path)
            width, height = img.size
            img.close()
            
            # Create the image entry for the COCO data
            coco_data['images'].append({
                'id': image_id_counter,
                'file_name': os.path.relpath(image_path, dataset_base_path),
                'width': width,
                'height': height
            })
            
            # Convert masks to bounding boxes for the current label file
            bounding_box_annotations = convert_masks_to_bboxes(label_path, uavid_class_map)
            
            # Add the annotations to the COCO data
            for bbox_anno in bounding_box_annotations:
                coco_data['annotations'].append({
                    'id': annotation_id_counter,
                    'image_id': image_id_counter,
                    'category_id': bbox_anno['class_id'],
                    'bbox': bbox_anno['bbox'],
                    'area': bbox_anno['bbox'][2] * bbox_anno['bbox'][3],
                    'iscrowd': 0 # Assuming not crowd for simplicity
                })
                annotation_id_counter += 1
            
            image_id_counter += 1
            
    # Save the final JSON file
    output_path = os.path.expanduser('~/dataset/UAVid/uavid_val_coco.json')
    print(f"\nSaving annotations to {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=4)
        
    print("Annotation file created successfully!")
