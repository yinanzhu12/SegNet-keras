import argparse
from keras.models import Model
from keras.layers import Activation,Input
from custom_layers import MaxPoolingWithIndices,UpSamplingWithIndices,CompositeConv
import config as cf
from util import read_image_batch
from create_model import create_model

def main(args):
   
    my_model=create_model()
    my_model.compile(cf.optimizer,loss=cf.loss_function,metrics=cf.metrics)
    my_model.load_weights(cf.model_path+args.model_name)
    test_data=read_image_batch(cf.test_set_path,10)
    my_model.evaluate_generator(test_data,steps=(cf.test_set_size+1)//cf.batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",default='my_model')
    args = parser.parse_args()
    main(args)