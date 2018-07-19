training_set_path='SegNet/CamVid/train.txt'
val_set_path='SegNet/CamVid/val.txt'
test_set_path='SegNet/CamVid/test.txt'
model_path='models/'

num_classes=13
image_shape=(360,480,3)
num_features=6
batch_size=10
epochs=2
training_set_size=367
val_set_size=101
test_set_size=233
optimizer='sgd'
loss_function='categorical_crossentropy'
metrics=['accuracy']