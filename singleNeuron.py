import numpy as np
from sklearn.datasets import make_blobs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
from abstractModel import AbstractModel

class ArtificialNeuron(AbstractModel):

    def __init__(self,learningRate=0.1,nIterations=200):
        """Initialize the weights and the bias for the neuron"""
        self.learningRate = learningRate
        self.nIterations = nIterations

    def __str__(self):
        return f"Artificial Neuron"

    def model(self,X):
        """
        Save the output of the ArtificialNeuron
        - inputs X
        - weights W
        - bias b
        """
        try:
            Z = X.dot(self.W)+self.b
            A = 1/(1+np.exp(-Z)) #Sigmoid activation
            return A
        except:
            raise RuntimeError("Train the ArtificialNeuron first")
        
    def logLoss(self,A,y):
        """Compute the log loss between the output of the model and y"""
        return (1/len(y)) * np.sum(-y*np.log(A) - (1-y)*np.log(1-A))

    def logLossGradient(self,A,X,y):
        """Save the log loss gradient"""
        self.dW = 1/A.shape[0] * np.dot(X.T,A-y)
        self.db = 1/A.shape[0] * np.sum(A-y)

    def gradientDescent(self):
        """Implements the gradient descent algorithm for W and b
        according to the log loss function and save the new values
        of the weights W and the bias b"""
        self.W = self.W - self.learningRate * self.dW
        self.b = self.b - self.learningRate * self.db

    def predict(self,X,strict=True):
        """Return the output vector of the ArtificialNeuron
        for the given input vector X"""
        A = self.model(X)
        if strict:
            return A > 0.5
        else:
            return A

    def train(self,X,y,plotLogLossEvolution=False, plotDecisionBoundary=False):
        """The training algorithm of the artificial Neuron, the 
        elementary unit in an artificial neural network"""

        self.W = np.random.randn(X.shape[1],1)
        self.b = np.random.randn(1)

        lossValues = []
        
        for i in tqdm.tqdm(range(self.nIterations)):
            A = self.model(X) #Compute the output of the model
            loss = self.logLoss(A,y)
            lossValues.append(loss)
            self.logLossGradient(A,X,y) #update values of dW and db
            self.gradientDescent() #update values of W and b
            if plotDecisionBoundary:
                self.savePlot(X,y,name=f"{i}.png",iteration=i+1)

        if plotLogLossEvolution:
            plt.plot(lossValues)
            plt.show()
    

    def plotDecisionBoundary(self,X,y,name="singleNeuronBoundary.png",iteration=0):
        """Plot the input data on a bidimensional graph, colored depending
        on the corresponding output y and with a line representing the
        decision boundary of the ArtificialNeuron"""
        

        plt.rcParams.update({"text.color":"#f5f5f5",
            "axes.labelcolor":"#f5f5f5",
            "axes.facecolor":"#000",
            "axes.edgecolor":"#f5f5f5",
            "grid.color":"#f5f5f5",
            "xtick.color":"#f5f5f5",
            "ytick.color":"#f5f5f5",
            "figure.facecolor":"#000",
            })

        min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
        min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1
        x1grid = np.arange(min1, max1, 0.01)
        x2grid = np.arange(min2, max2, 0.01)
        xx, yy = np.meshgrid(x1grid, x2grid)
        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
        grid = np.hstack((r1,r2))
        yhat = self.predict(grid,strict=False)
        yhat = yhat[:, 0]
        zz = yhat.reshape(xx.shape)
        f = plt.figure()
        f.set_figwidth(16)
        f.set_figheight(9)
        c = plt.contourf(xx, yy, zz, cmap='seismic', levels=4, vmin=0, vmax=1)
        plt.colorbar(c,label="Real Output")

        sns.scatterplot(
                x=X[:,0],
                y=X[:,1],
                hue=y.flatten(),
                palette="seismic",
                s=16,
                edgecolor="black"
                )

        plt.xlabel("feature n°1")
        plt.ylabel("feature n°2")
        plt.legend(title="Expected output")
        plt.title(f"Decision Boundary of the Artificial Neuron (accuracy={self.accuracyScore(self.predict(X),y)}, iteration={iteration})")
        f.savefig(name,format="png")
        plt.close()

if __name__=="__main__":
    ### VARIABLES ###
    nFeatures = 2
    nSamples = 1000

    ### BUILD DATASET
    X,y = make_blobs(
            n_samples=nSamples, 
            n_features=nFeatures, 
            centers=2,
            random_state=0
            )
    y = y.reshape((y.shape[0],1))
    
    ### CREATE AND TRAIN ARTIFICIAL NEURON
    an = ArtificialNeuron(nIterations=100)
    an.train(X,y,plotDecisionBoundary=True)

    ### DISPLAY THE RESULTS 
    print(an.accuracyScore(an.predict(X),y))
       


