import numpy as np
from sklearn.datasets import make_blobs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class ArtificialNeuron:

    def __init__(self,learningRate=0.1,nIterations=200):
        """Initialize the weights and the bias for the neuron"""
        self.learningRate = learningRate
        self.nIterations = nIterations

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

    def predict(self,X):
        """Return the output vector of the ArtificialNeuron
        for the given input vector X"""
        A = self.model(X)
        return A > 0.5

    def train(self,X,y,plotLogLossEvolution=False):
        """The training algorithm of the artificial Neuron, the 
        elementary unit in an artificial neural network"""

        self.W = np.random.randn(X.shape[1],1)
        self.b = np.random.randn(1)

        lossValues = []
        
        for i in range(self.nIterations):
            A = self.model(X) #Compute the output of the model
            loss = self.logLoss(A,y)
            lossValues.append(loss)
            self.logLossGradient(A,X,y) #update values of dW and db
            self.gradientDescent() #update values of W and b

        if plotLogLossEvolution:
            plt.plot(lossValues)
            plt.show()
    
    def accuracyScore(self,y,yPredicted):
        """Return the accuracy score of the predictions yPredicted
        compared to the know outcomes y"""
        n = len(y) #the number of predictions/outcomes
        s = 0
        for i in range(n):
            s += int(yPredicted[i]==y[i])
        return s/n

    def plotDecisionBoundary(self,X,y):
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
        yhat = self.predict(grid)
        yhat = yhat[:, 0]
        zz = yhat.reshape(xx.shape)
        f = plt.figure()
        f.set_figwidth(16)
        f.set_figheight(9)
        c = plt.contourf(xx, yy, zz, cmap='RdBu')
        plt.colorbar(c)

        sns.scatterplot(
                x=X[:,0],
                y=X[:,1],
                hue=y.flatten(),
                palette="Set1",
                s=16,
                edgecolor="black"
                )

        plt.xlabel("feature n°1")
        plt.ylabel("feature n°2")
        plt.legend(title="output")
        plt.title(f"Decision Boundary of the Artificial Neuron with a precision of {self.accuracyScore(self.predict(X),y)}")
        plt.show()


        

if __name__=="__main__":
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
       


