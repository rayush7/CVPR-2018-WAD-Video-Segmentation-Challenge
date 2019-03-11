#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN Training on Kaggle WAD Competition 2018

# In[33]:


import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw


# In[34]:


# Set the ROOT_DIR variable to the root directory of the Mask_RCNN git repo
ROOT_DIR = './'
assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist.'

# Import mrcnn libraries
sys.path.append(ROOT_DIR) 
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib


# ## Set up logging and pre-trained model paths

# In[35]:


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR,"Pretrained_Model","mask_rcnn_coco.h5")

print(COCO_MODEL_PATH)

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# ## Configuration
# Define configurations for training on the Kaggle WAD Dataset

# In[36]:


class WADConfig(Config):
    """Configuration for training on the cigarette butts dataset.
    Derives from the base Config class and overrides values specific
    to the cigarette butts dataset.
    """
    # Give the configuration a recognizable name
    NAME = "WAD"

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 5

    # Number of classes (including background)
    NUM_CLASSES = 1 + 7  # background + 1 (7 classes in our dataset)

    # All of our training images are 512x512
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 832

    # You can experiment with this number to see if it improves training
    STEPS_PER_EPOCH = 10

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 5
    
    # Can also use resnet101
    BACKBONE = 'resnet50'

    # Set Region Proposal Anchor Scales
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50 
    POST_NMS_ROIS_INFERENCE = 500 
    POST_NMS_ROIS_TRAINING = 1000 
    
config = WADConfig()
config.display()


# # Define the dataset
# Generic COCO Like Dataset Loader Class

# In[37]:


class CocoLikeDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """
    def load_data(self, annotation_json, images_dir):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
        # Load json from file
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()
        
        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
                return
            
            self.add_class(source_name, class_id, class_name)
        
        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)
        
        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                
                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]
                
                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )
                
    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []
        
        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        
        return mask, class_ids


# # Create the Training and Validation Datasets

# In[38]:


dataset_train = CocoLikeDataset()
dataset_train.load_data('/home/ayush/Instance_Segmentation/all/Sample_Dataset/wad_sample_train.json', '/home/ayush/Instance_Segmentation/all/Sample_Dataset/sample_train_color')
dataset_train.prepare()

dataset_val = CocoLikeDataset()
dataset_val.load_data('/home/ayush/Instance_Segmentation/all/Sample_Dataset/wad_sample_val.json', '/home/ayush/Instance_Segmentation/all/Sample_Dataset/sample_val_color')
dataset_val.prepare()


# ## Display a few images from the training dataset

# In[39]:


dataset = dataset_train
image_ids = np.random.choice(dataset.image_ids, 4)
#print(dataset.class_names)
#print(image_ids)

for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    #visualize.display_top_masks(image, mask, class_ids, dataset.class_names)


# # Create the Training Model and Train

# In[14]:


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# In[15]:


# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)


# ## Training
# 
# Train in two stages:
# 
# 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass layers='heads' to the train() function.
# 
# 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass layers="all to train all layers.
# 
# 

# In[16]:


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
#start_train = time.time()
#model.train(dataset_train, dataset_val, 
#            learning_rate=config.LEARNING_RATE, 
#            epochs=2, 
#            layers='heads')
#end_train = time.time()
#minutes = round((end_train - start_train) / 60, 2)
#print('Training took {minutes} minutes')


# In[11]:


# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
start_train = time.time()
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=8, 
            layers="all")
end_train = time.time()
minutes = round((end_train - start_train) / 60, 2)
print('Training took {minutes} minutes')


# # Prepare to run Inference
# Create a new InferenceConfig, then use it to create a new model.

# In[22]:


#class InferenceConfig(WADConfig):
#    GPU_COUNT = 1
#    IMAGES_PER_GPU = 1
#    IMAGE_MIN_DIM = 640
#    IMAGE_MAX_DIM = 832
#    DETECTION_MIN_CONFIDENCE = 0.7
    

#inference_config = InferenceConfig()


# In[23]:


# Recreate the model in inference mode
#model = modellib.MaskRCNN(mode="inference", 
#                          config=inference_config,
#                          model_dir=MODEL_DIR)


# In[24]:


# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
#model_path = model.find_last()

# Load trained weights (fill in path to trained weights here)
#assert model_path != "", "Provide path to trained weights"
#print("Loading weights from ", model_path)
#model.load_weights(model_path, by_name=True)


# # Run Inference
# Run model.detect() on Test Images

# In[25]:


#import skimage
#real_test_dir = '/home/ayush/Instance_Segmentation/all/Sample_Dataset/sample_test'
#image_paths = []
#for filename in os.listdir(real_test_dir):
#    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
#        image_paths.append(os.path.join(real_test_dir, filename))

#for image_path in image_paths:
#    img = skimage.io.imread(image_path)
#    img_arr = np.array(img)
#    results = model.detect([img_arr], verbose=1)
#    r = results[0]
#    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
#                                dataset_val.class_names, r['scores'], figsize=(5,5))


# In[ ]: