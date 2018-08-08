import argparse
from keras.models import Model
from keras.layers import Activation,Input
from custom_layers import MaxPoolingWithIndices,UpSamplingWithIndices,CompositeConv
import config as cf
from util import read_image_batch

def main(args):
    inputs=Input(shape=cf.image_shape)
    x=CompositeConv(inputs,2,4)
    x,argmax1=MaxPoolingWithIndices(pool_size=2,strides=2)(x)
    x=CompositeConv(x,2,4)
    x,argmax2=MaxPoolingWithIndices(pool_size=2,strides=2)(x)
    x=UpSamplingWithIndices()([x,argmax2])
    x=CompositeConv(x,2,4)
    x=UpSamplingWithIndices()([x,argmax1])
    x=CompositeConv(x,2,[4,cf.num_classes])
    y=Activation('softmax')(x)
    
    my_model=Model(inputs=inputs,outputs=y)
    my_model.compile(cf.optimizer,loss=cf.loss_function,metrics=cf.metrics)
    my_model.load_weights(cf.model_path+args.model_name)
    test_data=read_image_batch(cf.test_set_path,10)
    my_model.evaluate_generator(test_data,steps=(cf.test_set_size+1)//cf.batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",default='my_model')
    args = parser.parse_args()
    main(args)
