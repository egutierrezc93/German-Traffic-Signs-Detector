# LeNet Model

This model is implemented in tensorflow with this important details:

- Xavier initialization is used
- RELU activation function is used instead of sigmoid
- Cost is modified to deal with class imbalance
- Data augmentation is used to improve the model's ability to learn
- AdamOptimizer is used to minimize the cost
- The learning rate is 0.001
- minibatches of size 128 each are used during the training phase

The model architecture in lenet is as follows:

POOL -> CONV -> POOL -> CONV -> FC -> FC

Where:

- CONV: input volume size is 32x32x3 and out volume is 28x28x6. Each filter is of size 5x5 with a stride of 1
- POOL: input volume size is 28x28x6 and output volume is 14x14x6. The kernel is of size 2x2
- CONV: input volume size is 14x14x6 and output volume is 10x10x16. Each filter is of size 5x5 with a stride of 1
- POOL: input volume size is 10x10x16 and output volume is 5x5x16. The kernel is of size 2x2
- FC: the last volume is flattened in a 400 unidirectional vector and connected to a FC layer with 120 neurons
- FC: The previous 120 neurons are connected to a FC layer with 84 neurons
- To make predictions the previous layer is connected to a softmax output layer with 43 neurons corresponding to each traffic sign
