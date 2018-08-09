from keras.models import Model
from keras.layers import Activation,Input
from custom_layers import MaxPoolingWithIndices,UpSamplingWithIndices,CompositeConv
import config as cf

def create_model():
    inputs=Input(shape=cf.image_shape)
    x=CompositeConv(inputs,2,32)
    x,argmax1=MaxPoolingWithIndices(pool_size=2,strides=2)(x)
    
    x=CompositeConv(x,3,64)
    x,argmax2=MaxPoolingWithIndices(pool_size=2,strides=2)(x)
    
    x=CompositeConv(x,3,256)
    x,argmax3=MaxPoolingWithIndices(pool_size=2,strides=2)(x)
    
    x=UpSamplingWithIndices()([x,argmax3])
    x=CompositeConv(x,3,[256,256,64])
    
    x=UpSamplingWithIndices()([x,argmax2])
    x=CompositeConv(x,3,[64,64,32])
    
    x=UpSamplingWithIndices()([x,argmax1])
    x=CompositeConv(x,2,[32,cf.num_classes])
    
    y=Activation('softmax')(x)
    my_model=Model(inputs=inputs,outputs=y)
    
    return my_model