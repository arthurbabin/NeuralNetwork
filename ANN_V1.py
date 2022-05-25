import numpy as np
from sklearn.datasets import make_circles
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class ANN_V1:
    params = {}
    gradients = {}

    def __init__(self,dims,learningRate=0.1,nIterations=200):
        """Initialize the dims corresponding to the number of neurons for
        each hidden layer, the weights and the bias for the neuron"""
        self.learningRate = learningRate
        self.nIterations = nIterations
        self.dims = dims

    def forwardPropagation(self,X):
        """
        Save the activation A of each layer
        """
        Z1 = self.params["W1"].dot(X.T)+self.params["b1"]
        A1 = 1/(1+np.exp(-Z1)) #Sigmoid activation
        Z2 = self.params["W2"].dot(A1)+self.params["b2"]
        A2 = 1/(1+np.exp(-Z2))

        assert A1.shape == (self.dims[0],X.shape[0])
        assert A2.shape == (1,X.shape[0])

        activations = {
                'A1':A1,
                'A2':A2
                }
        return activations
        
    def logLoss(self,A,y):
        """Compute the log loss between the output of the model and y"""
        return (1/len(y)) * np.sum(-y*np.log(A) - (1-y)*np.log(1-A))

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

        assert self.gradients["dZ1"].shape == (self.dims[0],X.shape[0])
        assert self.gradients["dZ2"].shape == (1,X.shape[0])
        assert self.gradients["dW1"].shape == (self.dims[0],X.shape[1])
        assert self.gradients["dW2"].shape == (1,self.dims[0])
        assert self.gradients["db1"].shape == (self.dims[0],1)
        assert self.gradients["db2"].shape == (1,1)

    def gradientDescent(self):
        """Implements the gradient descent algorithm
        according to the log loss function and save the new values
        of the parameters"""
        self.params["W1"] = self.params["W1"] - self.learningRate * self.gradients["dW1"]
        self.params["W2"] = self.params["W2"] - self.learningRate * self.gradients["dW2"]
        self.params["b1"] = self.params["b1"] - self.learningRate * self.gradients["db1"]
        self.params["b2"] = self.params["b2"] - self.learningRate * self.gradients["db2"]

        assert self.params["W1"].shape == (self.dims[0],X.shape[1])
        assert self.params["W2"].shape == (1,self.dims[0])
        assert self.params["b1"].shape == (self.dims[0],1)
        assert self.params["b2"].shape == (1,1)

    def predict(self,X):
        """Return the output vector of the ArtificialNeuron
        for the given input vector X"""
        activations = self.forwardPropagation(X)
        A2 = activations["A2"]

        return (A2 > 0.5).T

    def initParams(self,n0,n1,n2):
        np.random.seed(0)


        #First Layer
        self.params["W1"] = np.random.randn(n1,n0)
        self.params["b1"] = np.zeros((n1,1))

        #Second Layer
        self.params["W2"] = np.random.randn(n2,n1)
        self.params["b2"] = np.zeros((n2,1))


    def train(self,X,y,plotEvolution=False,plotDecisionBoundary=False):
        """The training algorithm of the ANN, the 
        elementary unit in an artificial neural network"""

        n0 = X.shape[1] #number of inputs
        n1 = self.dims[0] #number of neurons
        n2 = y.shape[1] #number of outputs

        self.initParams(n0,n1,n2)

        lossValues = []
        accValues = []

        for i in range(self.nIterations):
            activations = self.forwardPropagation(X) #Compute the output of the model
            self.backPropagation(X,y,activations) #update values of gradients 
            self.gradientDescent() #update values of parameters

            if i%10==0:
                loss = self.logLoss(activations["A2"],y)
                lossValues.append(loss)
                accValues.append(self.accuracyScore(y,self.predict(X)))

            if plotDecisionBoundary:
                self.plotDecisionBoundary(X,
                        y,
                        name=f"images/plots/{i}.png",
                        iteration=i+1)

        if plotEvolution:
            plt.subplot(1,2,1)
            plt.plot(lossValues,label="Loss evolution during training")
            plt.subplot(1,2,2)
            plt.plot(accValues, label="Accuracy evolution during training")
            plt.legend()
            plt.show()

    
    def accuracyScore(self,y,yPredicted):
        """Return the accuracy score of the predictions yPredicted
        compared to the know outcomes y"""
        assert y.shape == yPredicted.shape
        n = len(y) #the number of predictions/outcomes
        s = 0
        for i in range(n):
            s += int(np.array_equal(yPredicted[i],y[i]))
        return s/n

    def plotDecisionBoundary(self,X,y,name="decisionBoundary",iteration=None):
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
        c = plt.contourf(xx, yy, zz, cmap='seismic',vmin=0,vmax=1)
        plt.colorbar(c,label="Predicted Output")

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
        plt.legend(title="output")
        plt.title(f"Decision Boundary of the ANN (accuracy={self.accuracyScore(self.predict(X),y)}, iteration={iteration})")
        f.savefig(name,format="png")
        plt.close()

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
    an = ANN_V1([32],nIterations=1000,learningRate=0.1)
    an.train(X,y,plotDecisionBoundary=True)

    ### DISPLAY THE RESULTS 
    print(an.accuracyScore(an.predict(X),y))

