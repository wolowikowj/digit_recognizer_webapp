# Functions File

import numpy as np
from PIL import Image

def preprocess_img(img_path, new_width, new_height):
    img = Image.open(img_path).convert('L')
    new_img = img.resize(size = (new_width, new_height), resample=Image.Resampling.LANCZOS)
    img_array =np.array(new_img)

    return img_array

