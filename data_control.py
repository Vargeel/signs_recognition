import os
import pandas
import numpy as np
root = '../'
targets = []
image_names = []
for path, subdirs, files in os.walk(root):
    for name in files:
        if name[-3:]=='jpg': # we keep only the images
            target = np.zeros(10)
            target[int(path[-5:])-1] = 1 # we get the target through the path structure
            targets.append(target)
            image_names.append(os.path.join(path, name))


df = pandas.DataFrame({'image_paths': image_names})
df.to_csv('data.csv')
np.save('targets',targets)