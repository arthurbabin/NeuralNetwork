# Hand Made Neural Networks

## Artificial Neural Network with selected number of layers (ANN V2)

### ANN with 3 hidden layers to separate 2 circles

```python
anV2_circle = ANN_V2(hiddenLayers=[16,16,16],nIterations=2000,learningRate=0.1)
anV2_circle.train(Xcircle, ycircle)
```

![GIF Demo ANN V2](https://raw.github.com/arthurbabin/NeuralNetwork/main/images/gif/ANN_V2_Circle.gif)

<img src="https://github.com/arthurbabin/NeuralNetwork/blob/main/images/gif/ANN_V2_Circle.gif" width=1000>

### ANN with 5 hidden layers to separate 2 moons

```python
anV2_moon = ANN_V2(hiddenLayers=[16,16,16,16,16],nIterations=3000,learningRate=0.1)
anV2_moon.train(Xmoon, ymoon)
```

![GIF Demo ANN V2](https://raw.github.com/arthurbabin/NeuralNetwork/main/images/gif/ANN_V2_Moon.gif)

## Artificial Neural Network with 1 hidden layer (ANN V1)

### ANN with 1 hidden layer to separate 2 circles

```python
anV1_circle = ANN_V1(nbNeuronsHiddenLayer=32, nIterations=2000, learningRate=0.1)
anV0_circle.train(Xcircle, ycircle)
```

![GIF Demo ANN V1](https://raw.github.com/arthurbabin/NeuralNetwork/main/images/gif/ANN_V1_Circle.gif)


## Artificial Neuron Implementation

### Simple artificial neuron to separate two blobs

```python
anV0_blob = ArtificialNeuron(learningRate=0.1,nIterations=100)
anV0_blob.train(Xblob, yblob)
```

![GIF Demo Single Neuron](https://raw.github.com/arthurbabin/NeuralNetwork/main/images/gif/AN_Blob.gif)
