from PIL import Image 
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon

import os
import numpy as np
import tqdm
import datetime
import json

category2id = {'car':          33,
               'motorbicycle': 34, 
               'bicycle':      35,
               'person':       36, 
               'truck':        38,
               'bus':          39,
               'tricycle':     40,
              }
id2category = {}
for cat, _id in category2id.items():
    id2category[_id] = cat

def create_coco(img_path, label_path, destination):
    label_path.sort()
    img_path.sort()
    
    # create images list
    image_list = create_image_list(img_path)
    
    # create annotation list
    object2color, class_instance = create_object2color(label_path, id2category)
    category_ids = create_color2category(class_instance, object2color)
    
    is_crowd = 0

    # These ids will be automatically increased as we go
    annotation_id = 0

    # Create the annotations
    annotations = []
    for img_id, path in tqdm.tqdm(enumerate(label_path)):
        label = Image.open(path)
        label = np.array(label)
        mask = do_mask_image(label, class_instance[img_id], object2color)
        mask = Image.fromarray(np.uint8(mask))
        sub_masks = create_sub_masks(mask)
        for color, sub_mask in sub_masks.items():
            category_id = category_ids[img_id][color]
            annotation = create_sub_mask_annotation(sub_mask, img_id, category_id, annotation_id, is_crowd)
            annotations.append(annotation)
            annotation_id += 1
            
    data = {}
    data['info'] = {}
    data['licenses'] = []
    data['images'] = image_list
    data['annotations'] = annotations
    data['categories'] = [
        {"supercategory": "vehicle", "id": 33,"name": "car"},
        {"supercategory": "vehicle", "id": 34,"name": "motorbicycle"},
        {"supercategory": "vehicle", "id": 35,"name": "bicycle"},
        {"supercategory": "vehicle", "id": 38,"name": "truck"},
        {"supercategory": "vehicle", "id": 39,"name": "bus"},
        {"supercategory": "vehicle", "id": 40,"name": "tricycle"},

        {"supercategory": "person", "id": 36,"name": "person"},
    ]
    with open(destination, 'w') as f:
        json.dump(data, f, indent=4)

def create_image_list(img_path):
    image_list = []
    for img_id, path in enumerate(img_path):
        image_list += [{
            'id': img_id,
            'license': 0,
            'coco_url': 'https://www.google.com/',
            'flickr_url': 'https://www.google.com/',
            'width': 3384,
            'height': 2710,
            'file_name': path.split('/')[-1],
            'date_captured': datetime.datetime.now().replace(microsecond=0).isoformat(' '),
        }]
    return image_list


def create_object2color(label_path, id2category):
    
    class_instance = {}
    class_summary = {}
    for _id in id2category.keys():
        class_summary[_id] = set()
    
    for img_id, path in enumerate(label_path):
        label = Image.open(path)
        label = np.array(label)
        label_class = np.ndarray.astype(label/1000, np.int32)
        label_instance = np.mod(label, 1000)
        
        class_instance[img_id] = []
        for class_id in id2category.keys():
            i, j = np.where(label_class==class_id)
            instance_set = set(label_instance[i, j])
            
            class_summary[class_id] = class_summary[class_id].union(instance_set)
            for instance_id in instance_set:
                class_instance[img_id] += [(class_id, instance_id)]
            
    object2color = {}
    color_set = set()
    def generate_color(color_set):
        while True:
            r, g, b = np.random.randint(256, size=3)
            if not (r, g, b) in color_set:
                break
        return (r, g, b)
    
    for class_id, instances_id in class_summary.items():
        for instance_id in instances_id:
            r, g, b = generate_color(color_set)
            object2color[(class_id, instance_id)] = (r, g, b)
            
    
    return object2color, class_instance

def do_mask_image(label, class_instance, object2color):
    label_class = np.ndarray.astype(label/1000, np.int32)
    label_instance = np.mod(label, 1000)
    
    h, w = label.shape
    image_mask = np.zeros([h, w, 3], np.int32)
    for class_id, instance_id in class_instance:
        mask = np.zeros([h, w, 3], np.int32)
        r, g, b = object2color[(class_id, instance_id)]
        
        intersect = (label_class==class_id)*(label_instance==instance_id)
        mask[:, :, 0] = r*intersect
        mask[:, :, 1] = g*intersect
        mask[:, :, 2] = b*intersect
        
        image_mask += mask
        
    return image_mask

def create_color2category(class_instance, object2color):
    category_ids = {}

    for img_id, pair_list in class_instance.items():
        category_ids[img_id] = {}
        for class_id, instance_id in pair_list:
            r, g, b = object2color[(class_id, instance_id)]
            category_ids[img_id][str((r, g, b))] = class_id
    
    return category_ids

def create_sub_masks(mask_image):
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))[:3]

            # If the pixel is not black...
            if pixel != (0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                    # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))
                    
                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks



def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        try:
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
        except:
            continue
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation