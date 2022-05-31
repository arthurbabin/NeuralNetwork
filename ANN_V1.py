import numpy as np
from sklearn.datasets import make_circles
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
from abstractModel import AbstractModel

class ANN_V1(AbstractModel):
    params = {}
    gradients = {}

    def __init__(self,nbNeuronsHiddenLayer=3,learningRate=0.1,nIterations=200):
        """Initialize the dims corresponding to the number of neurons for
        each hidden layer, the weights and the bias for the neuron"""
        self.learningRate = learningRate
        self.nIterations = nIterations
        self.dim = nbNeuronsHiddenLayer

    def __str__(self):
        return f"ANN V1"

    def initParams(self,n0,n1,n2):
        np.random.seed(0)


        #First Layer
        self.params["W1"] = np.random.randn(n1,n0)
        self.params["b1"] = np.zeros((n1,1))

        #Second Layer
        self.params["W2"] = np.random.randn(n2,n1)
        self.params["b2"] = np.zeros((n2,1))

    def forwardPropagation(self,X):
        """
        Save the activation A of each layer
        """
        Z1 = self.params["W1"].dot(X.T)+self.params["b1"]
        A1 = 1/(1+np.exp(-Z1)) #Sigmoid activation
        Z2 = self.params["W2"].dot(A1)+self.params["b2"]
        A2 = 1/(1+np.exp(-Z2))

        assert A1.shape == (self.dim,X.shape[0])
        assert A2.shape == (1,X.shape[0])

        activations = {
                'A1':A1,
                'A2':A2
                }
        return activations
        
    def backPropagation(self,X,y,activations):
        """Save the log loss gradient"""
        m = y.shape[0]
        gradients = {}
        gradients["dZ2"] = activations["A2"]-y.T
        gradients["dW2"] = 1/m * gradients["dZ2"].dot(activations["A1"].T)
        gradients["db2"] = 1/m * np.sum(gradients["dZ2"],axis=1, keepdims=True)

        gradients["dZ1"] = np.dot(self.params["W2"].T,gradients["dZ2"])*activations["A1"]*(1-activations["A1"])
        gradients["dW1"] = 1/m * gradients["dZ1"].dot(X)
        gradients["db1"] = 1/m * np.sum(gradients["dZ1"],axis=1, keepdims=True)

        self.gradients = gradients

        assert self.gradients["dZ1"].shape == (self.dim,X.shape[0])
        assert self.gradients["dZ2"].shape == (1,X.shape[0])
        assert self.gradients["dW1"].shape == (self.dim,X.shape[1])
        assert self.gradients["dW2"].shape == (1,self.dim)
        assert self.gradients["db1"].shape == (self.dim,1)
        assert self.gradients["db2"].shape == (1,1)

    def gradientDescent(self,X):
        """Implements the gradient descent algorithm
        according to the log loss function and save the new values
        of the parameters"""
        self.params["W1"] = self.params["W1"] - self.learningRate * self.gradients["dW1"]
        self.params["W2"] = self.params["W2"] - self.learningRate * self.gradients["dW2"]
        self.params["b1"] = self.params["b1"] - self.learningRate * self.gradients["db1"]
        self.params["b2"] = self.params["b2"] - self.learningRate * self.gradients["db2"]

        assert self.params["W1"].shape == (self.dim,X.shape[1])
        assert self.params["W2"].shape == (1,self.dim)
        assert self.params["b1"].shape == (self.dim,1)
        assert self.params["b2"].shape == (1,1)

    def predict(self,X,strict=True):
        """Return the output vector of the ArtificialNeuron
        for the given input vector X"""
        activations = self.forwardPropagation(X)
        A2 = activations["A2"]

        if strict:
            return (A2 > 0.5).T
        else:
            return A2.T

    def train(self,X,y,plotEvolution=False,plotDecisionBoundary=False):
        """The training algorithm of the ANN, the 
        elementary unit in an artificial neural network"""

        n0 = X.shape[1] #number of inputs
        n1 = self.dim #number of neurons
        n2 = y.shape[1] #number of outputs

        self.initParams(n0,n1,n2)

        lossValues = []
        accValues = []

        for i in tqdm.tqdm(range(self.nIterations)):
            activations = self.forwardPropagation(X) #Compute the output of the model
            self.backPropagation(X,y,activations) #update values of gradients 
            self.gradientDescent(X) #update values of parameters

            if i%10==0:
                loss = self.logLoss(activations["A2"],y)
                lossValues.append(loss)
                accValues.append(self.accuracyScore(y,self.predict(X)))

            if plotDecisionBoundary:
                self.savePlot(X,y,name="{i}.png", iteration=i+1)

        if plotEvolution:
            plt.subplot(1,2,1)
            plt.plot(lossValues,label="Loss evolution during training")
            plt.subplot(1,2,2)
            plt.plot(accValues, label="Accuracy evolution during training")
            plt.legend()
            plt.show()

if __name__=="__main__":
    ### VARIABLES ###
    nSamples = 1000
    noise = 0.1
    factor = 0.3

    ### BUILD DATASET
    X,y = make_circles(
            n_samples=nSamples, 
            noise=noise, 
            factor=factor,
            random_state=0
            )
    y = y.reshape((y.shape[0],1))
    
    ### CREATE AND TRAIN ARTIFICIAL NEURON
    an = ANN_V1(nbNeuronsHiddenLayer=32,nIterations=1000,learningRate=0.1)
    an.train(X,y,plotDecisionBoundary=False)

    ### DISPLAY THE RESULTS 
    print(an.accuracyScore(an.predict(X),y))

