import numpy as np
import tqdm
from abstractModel import AbstractModel

class ANN_V2(AbstractModel):
    params = {}
    gradients = {}

    def __init__(self,hiddenLayers=[16,16],learningRate=0.1,nIterations=200):
        """Initialize the dims corresponding to the number of neurons for
        each hidden layer, the weights and the bias for the neuron"""
        self.learningRate = learningRate
        self.nIterations = nIterations
        self.dims = list(hiddenLayers) #still need to insert X.shape[1] and y.shape[1]
        np.random.seed(0)

    def __str__(self):
        return f"ANN V2"

    def initParams(self):
        """
        Initialize the weights and bias for the neural network
        according to its dimensions
        """
        params = {}
        for layer in range(1,len(self.dims)):
            params[f"W{layer}"] = np.random.randn(self.dims[layer],self.dims[layer-1])
            params[f"b{layer}"] = np.zeros((self.dims[layer],1))

        self.params = params

    def forwardPropagation(self,X):
        """
        Save the activation A of each layer in a dict
        """
        activations = {"A0":X.T}

        assert len(self.params)//2+1 == len(self.dims)

        for layer in range(1,len(self.dims)):
            Z = self.params[f"W{layer}"].dot(activations[f"A{layer-1}"]) + self.params[f"b{layer}"]
            A = 1/(1+np.exp(-Z))
            assert A.shape == (self.dims[layer],X.shape[0])
            activations[f"A{layer}"] = A

        return activations
        
    def backPropagation(self,y,activations):
        """Save the log loss gradient"""
        m = y.shape[0]
        gradients = {}
        dZ = activations[f"A{len(self.dims)-1}"] - y.T

        for layer in range(len(self.dims)-1,0,-1):
            gradients[f"dW{layer}"] = 1/m * dZ.dot(activations[f"A{layer-1}"].T)
            gradients[f"db{layer}"] = 1/m * np.sum(dZ, axis=1, keepdims=True)

            if layer>1:
                dZpara = np.dot(self.params[f"W{layer}"].T,dZ)
                dZ = dZpara*activations[f"A{layer-1}"]*(1-activations[f"A{layer-1}"])

        self.gradients = gradients

        for layer in range(1,len(self.dims)):
            assert self.gradients[f"dW{layer}"].shape == (self.dims[layer],self.dims[layer-1])
            assert self.gradients[f"db{layer}"].shape == (self.dims[layer],1)

    def gradientDescent(self):
        """Implements the gradient descent algorithm
        according to the log loss function and save the new values
        of the parameters"""

        for key in self.params.keys():
            self.params[key] = self.params[key] - self.learningRate * self.gradients[f"d{key}"]
            
        for layer in range(1,len(self.dims)):
            assert self.params[f"W{layer}"].shape == (self.dims[layer],self.dims[layer-1])
            assert self.params[f"b{layer}"].shape == (self.dims[layer],1)

    def predict(self,X,strict=True):
        """Return the output vector of the ArtificialNeuron
        for the given input vector X"""
        activations = self.forwardPropagation(X)
        A = activations[f"A{len(self.dims)-1}"]

        if strict:
            return (A > 0.5).T
        else:
            return A.T

    def train(self,X,y,plotEvolution=False,plotDecisionBoundary=False):
        """The training algorithm of the ANN, the 
        elementary unit in an artificial neural network"""

        #Add dimensions of inputs and outputs in the list of dimensions
        self.dims.insert(0, X.shape[1])
        self.dims.append(y.shape[1])

        #Initialize the parameters dict
        self.initParams()

        lossValues = []
        accValues = []

        for i in tqdm.tqdm(range(self.nIterations)):
            activations = self.forwardPropagation(X) #Compute the output of the model
            self.backPropagation(y,activations) #update values of gradients 
            self.gradientDescent() #update values of parameters

            if i%10==0:
                loss = self.logLoss(activations[f"A{len(self.dims)-1}"],y)
                lossValues.append(loss)
                acc = self.accuracyScore(y,self.predict(X)) 
                accValues.append(acc)

            if plotDecisionBoundary:
                self.savePlot(X,y,name=f"{i}.png",iteration=i+1)

        if plotEvolution:
            plt.subplot(1,2,1)
            plt.plot(lossValues,label="Loss evolution during training")
            plt.subplot(1,2,2)
            plt.plot(accValues, label="Accuracy evolution during training")
            plt.legend()
            plt.show()
