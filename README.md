# Hand Made Neural Networks
## Artificial Neuron Implementation


![GIF Demo Single Neuron](https://raw.github.com/arthurbabin/NeuralNetwork/main/images/gif/singleNeuron.gif)

### Usage
```python
### VARIABLES ###
nFeatures = 2
nSamples = 1000

### BUILD DATASET
X,y = make_blobs(
        n_samples=nSamples, 
        n_features=nFeatures, 
        centers=2,
        random_state=7845
        )
y = y.reshape((y.shape[0],1))

### CREATE AND TRAIN ARTIFICIAL NEURON
an = ArtificialNeuron(nIterations=10000, learningRate=0.1)
an.train(X,y)

### DISPLAY THE RESULTS 
print(an.accuracyScore(an.predict(X),y))
an.plotDecisionBoundary(X,y)
```

## Artificial Neural Network with 1 hidden layer (ANN V1)

![GIF Demo ANN V1](https://raw.github.com/arthurbabin/NeuralNetwork/main/images/gif/ANN_V1.gif)
