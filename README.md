# Hand Made Neural Networks
## Artificial Neuron Implementation
<img src="https://github.com/arthurbabin/NeuralNetwork/blob/main/images/AN_decisionBoundary.png?raw=true" width="800" height="450"/>
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
an = ArtificialNeuron(nIterations=10000)
an.train(X,y)

### DISPLAY THE RESULTS 
print(an.accuracyScore(an.predict(X),y))
an.plotDecisionBoundary(X,y)
```

```ruby
require 'redcarpet'
markdown = Redcarpet.new("Hello World!")
puts markdown.to_html
```

