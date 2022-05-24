# Hand Made Neural Networks
<br>
## Artificial Neuron Implementation
<img src="https://github.com/arthurbabin/checkersAI/blob/main/screenshots/example.png?raw=true" width="300" height="324"/>
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
an = ArtificialNeuron(nIterations=10000)
an.train(X,y)

### DISPLAY THE RESULTS 
print(an.accuracyScore(an.predict(X),y))
an.plotDecisionBoundary(X,y)
```

