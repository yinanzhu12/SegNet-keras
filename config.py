training_set_path='SegNet/CamVid/train.txt'
val_set_path='SegNet/CamVid/val.txt'
test_set_path='SegNet/CamVid/test.txt'
model_path='models/'

num_classes=13
image_shape=(360,480,3)
padding=((12,12),(16,16))
batch_size=5
epochs=100
training_set_size=367
val_set_size=101
test_set_size=233
optimizer='sgd'
loss_function='categorical_crossentropy'
metrics=['accuracy']