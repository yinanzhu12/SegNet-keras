import imageio
import numpy as np
from keras.utils import to_categorical
import config as cf

def read_image_batch(data_path,batch_size):
    while True:
        image_list=open(data_path).readlines()
        l=len(image_list)
        num_batch=l//batch_size
        if num_batch*batch_size<l:
            num_batch+=1
        for i in range(num_batch):
            batch_set=image_list[batch_size*i:min(batch_size*(i+1),l)]
            batch_set=[bs.strip().split() for bs in batch_set]
            X=np.array([imageio.imread(line[0][1:]) for line in batch_set])
            labels=np.array([imageio.imread(line[1][1:]) for line in batch_set])
            y=to_categorical(labels,cf.num_classes)
            yield tuple((X, y))