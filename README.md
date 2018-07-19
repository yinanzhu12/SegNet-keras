# SegNet-keras-implementation
An implementation of SegNet(https://arxiv.org/abs/1511.00561) in keras

The repository doesn't contain dataset, please prepare and set up it in config.py. A nicely normalized and cleaned dataset can be downloaded from https://github.com/alexgkendall/SegNet-Tutorial.

python train.py --model_name [name of model to be saved]
  
python test.py --model_name [name of model to read]

Currently the model (defined in train.py) is a toy version of the model proposed in the paper, with much fewer parameters so that it can be tested on a laptop. Will scale up and move it to the cloud in the future
