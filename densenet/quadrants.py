import numpy as np
import pickle

# Deserializes CIFAR data.
def unpickle(data):
    with open(file, 'rb') as f:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Returns an 8x8 quadrant of a 32x32 input image (img).
# img is a 3x32 array, with each row containing values of 1 of 3 color channels
def get_quadrant(img):
    dim = 8
    quadrant = dim * np.random.randint(4)
    img = img[:,[quadrant, quadrant+7]]
    return img

# Converts all data samples to quadrants
def quadrant(data):
    for rgb_mat in data:
        rgb_mat = get_quadrant(rgb_mat)
    return data
