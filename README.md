# Text Task
## Usage example

run python extract_data.py to create train/validation/test set named data_split.p

Simply run python main.py and adjust parameters to train or test

# Image Task
## Usage example

With images and pre-trained AlexNet model file in the directory, we can train and test models using the following snippet.

~~~python
from main import *

# train simple cnn from scratch
train_X, train_y, \
valid_X, valid_y, \
test_X, test_y = build_default_dataset((128, 128, 3))

train(train_X, train_y, valid_X, valid_y,
      net_name='simple', learning_rate=1e-3, 
          dirname='simple-cnn', n_epoch=30, batch_size=128)
      

# train last a few fc layers of AlexNet   
train_X, train_y, \
valid_X, valid_y, \
test_X, test_y = build_default_dataset((227, 227, 3))

train(train_X, train_y, valid_X, valid_y,
      net_name='an_fc3', learning_rate=1e-3, 
      dirname='simple-cnn', n_epoch=30, batch_size=128)
      
# resume training         
train(train_X, train_y, valid_X, valid_y, restored_model_file='model-2000',
      net_name='an_fc3', learning_rate=1e-3, 
      dirname='simple-cnn', n_epoch=30, batch_size=128)

# test
test(test_X, test_y, restored_model_file='runs/an_fc3/checkpoints/model-3750', net_name='an_fc3')  
~~~

## Using TensorBoard

`tensorboard --logdir runs/simple-cnn --port 8090`
