from keras.models import Model
from keras.layers import Activation,Input,ZeroPadding2D,Cropping2D
from custom_layers import MaxPoolingWithIndices,UpSamplingWithIndices,CompositeConv
import config as cf

def create_model():
    inputs=Input(shape=cf.image_shape)

    x = ZeroPadding2D(cf.padding)(inputs)

    x=CompositeConv(x,2,64)
    x,argmax1=MaxPoolingWithIndices(pool_size=2,strides=2)(x)
    
    x=CompositeConv(x,2,64)
    x,argmax2=MaxPoolingWithIndices(pool_size=2,strides=2)(x)
    
    x=CompositeConv(x,3,64)
    x,argmax3=MaxPoolingWithIndices(pool_size=2,strides=2)(x)

    x=CompositeConv(x,3,64)
    x,argmax4=MaxPoolingWithIndices(pool_size=2,strides=2)(x)

    x=CompositeConv(x,3,64)
    x,argmax5=MaxPoolingWithIndices(pool_size=2,strides=2)(x)

    x=UpSamplingWithIndices()([x,argmax5])
    x=CompositeConv(x,3,64)

    x=UpSamplingWithIndices()([x,argmax4])
    x=CompositeConv(x,3,64)

    x=UpSamplingWithIndices()([x,argmax3])
    x=CompositeConv(x,3,64)

    x=UpSamplingWithIndices()([x,argmax2])
    x=CompositeConv(x,2,64)
    
    x=UpSamplingWithIndices()([x,argmax1])
    x=CompositeConv(x,2,[64,cf.num_classes])

    x=Activation('softmax')(x)

    y=Cropping2D(cf.padding)(x)
    my_model=Model(inputs=inputs,outputs=y)
    
    return my_model