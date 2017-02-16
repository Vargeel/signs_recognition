import pandas
from PIL import Image
import numpy as np

dat=pandas.read_csv('data.csv')

names=dat['image_paths']

input_size = 64

for path in names:
    im = Image.open(path)
    dimensions = im.size
    max_dim = np.argmax(dimensions)
    if max_dim == 0: # we resize the data with the max dimension being 64 pixels
        h = input_size
        w = int(input_size * dimensions[1]/dimensions[0])
    else:
        h = int(input_size * dimensions[0]/dimensions[1])
        w = input_size

    imTmp = np.asarray(im.resize((h, w), Image.ANTIALIAS)).astype('float32')
    image_temp = Image.fromarray(imTmp.astype('uint8'))
    random_bg = np.random.random_integers(0,255,(input_size,input_size,3))
    background = Image.fromarray(random_bg.astype('uint8')) # we create a salt and pepper background
    offset = ((input_size - h) / 2, (input_size - w) / 2)
    background.paste(image_temp, offset) # we paste the image on the salt and pepper background
    background.save(path)