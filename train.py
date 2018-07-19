import argparse
from keras.models import Model
from keras.layers import Activation,Input
from custom_layers import MaxPoolingWithIndices,UpSamplingWithIndices,CompositeConv
import config as cf
from util import read_image_batch

def main(args):
    inputs=Input(shape=cf.image_shape)
    x=CompositeConv(inputs,2,cf.num_features)
    x,argmax1=MaxPoolingWithIndices(pool_size=2,strides=2)(x)
    x=CompositeConv(x,2,cf.num_features)
    x,argmax2=MaxPoolingWithIndices(pool_size=2,strides=2)(x)
    x=UpSamplingWithIndices()([x,argmax2])
    x=CompositeConv(x,2,cf.num_features)
    x=UpSamplingWithIndices()([x,argmax1])
    x=CompositeConv(x,2,[cf.num_features,cf.num_classes])
    y=Activation('softmax')(x)
    my_model=Model(inputs=inputs,outputs=y)
    my_model.compile(cf.optimizer,loss=cf.loss_function,metrics=cf.metrics)
    
    val_data=read_image_batch(cf.val_set_path,cf.batch_size)
    train_data=read_image_batch(cf.training_set_path,cf.batch_size)
    my_model.fit_generator(train_data,
                           steps_per_epoch=(cf.training_set_size+1)//cf.batch_size,
                           epochs=cf.epochs,validation_data=val_data,
                           validation_steps=(cf.val_set_size+1)//cf.batch_size)
    my_model.save_weights(cf.model_path+args.model_name)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",default='my_model')
    args = parser.parse_args()
    main(args)