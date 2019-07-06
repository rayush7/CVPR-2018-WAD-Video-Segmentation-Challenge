from skimage.transform import resize

def resize_image(img, crop_size=None, down_scale=None):
    '''
    This function is used to resize the given image.
    
    Args:
        img        : image to be cropped in numpy array.
        label      : label of image to be cropped in numpy array.
        crop_size  : cropping size [top, bottom, left, right]
        down_scale : the scale used to down sample the image
    Returns:
        rescale image and label
    '''
    
    crop_top, crop_down, crop_left, crop_right = crop_size
    img = img[crop_top:-crop_down, crop_left:-crop_right]
    
    h, w, ch = img.shape
    if not (h%down_scale==0 and w%down_scale==0):
        raise ValueError('Down sample fail!')
    else:
        h = int(h/down_scale)
        w = int(w/down_scale)
        
    img = resize(img, (h, w, ch), anti_aliasing=True, mode='constant')
    return img

def resize_label(label, crop_size=None, down_scale=None):
    crop_top, crop_down, crop_left, crop_right = crop_size
    label = label[crop_top:-crop_down, crop_left:-crop_right]
    
    h, w = label.shape
    if not (h%down_scale==0 and w%down_scale==0):
        raise ValueError('Down sample fail!')
    
    
    label = label[::down_scale, ::down_scale]
    return label