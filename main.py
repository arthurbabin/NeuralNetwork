import numpy as np
from sklearn.datasets import make_circles, make_moons
import matplotlib.pyplot as plt
from ANN_V1 import *
from ANN_V2 import *
from singleNeuron import *
from decisionBoundary import *

def trainAndResults(model,X,y,iterationPlotRate=False):
    """
    Train model then display the resulting accuracy and decision boundary
    """
    print(f"\n{model} training...")
    model.iterationPlotRate = iterationPlotRate
    model.train(X,y,plotDecisionBoundary=bool(iterationPlotRate))
    accuracy = model.accuracyScore(model.predict(X),y)
    print(f"Result accuracy = {accuracy}\n")
    plotDecisionBoundary(model,X,y,path=f"images/{model}.png")

if __name__ == "__main__":
    ### VARIABLES ###
    nSamples = 1000
    noise_circle = 0.1
    noise_moon = 0.1
    factor = 0.3

    ### BUILD DATASETS
    Xblob,yblob = make_blobs(n_samples=nSamples,n_features=2,centers=2,random_state=0)
    yblob = yblob.reshape((yblob.shape[0],1))
    
    Xcircle,ycircle = make_circles(n_samples=nSamples,noise=noise_circle,factor=factor,random_state=0)
    ycircle = ycircle.reshape((ycircle.shape[0],1))

    Xmoon,ymoon = make_moons(n_samples=nSamples,noise=noise_moon, random_state=0)
    ymoon = ymoon.reshape((ymoon.shape[0],1))
    
    ### CREATE MODELS

    #Simple artificial neuron to separate two blobs
    anV0_blob = ArtificialNeuron(learningRate=0.1,nIterations=100)

    #ANN with 1 hidden layer to separate 2 circles
    anV1_circle = ANN_V1(nbNeuronsHiddenLayer=32, nIterations=2000, learningRate=0.1)

    #ANN with 3 hidden layers to separate 2 circles
    anV2_circle = ANN_V2(hiddenLayers=[16,16,16],nIterations=2000,learningRate=0.1)

    #ANN with 5 hidden layers to separate 2 moons
    anV2_moon = ANN_V2(hiddenLayers=[16,16,16,16,16],nIterations=3000,learningRate=0.1)

    ### TRAIN MODELS AND DISPLAY THE DECISION BOUNDARY
    #trainAndResults(anV0_blob,Xblob,yblob,iterationPlotRate=1)
    trainAndResults(anV1_circle,Xcircle,ycircle,iterationPlotRate=10)
    trainAndResults(anV2_circle,Xcircle,ycircle,iterationPlotRate=10)
    #trainAndResults(anV2_moon,Xmoon,ymoon,iterationPlotRate=10)

#    anV0_blob.saveGif("AN_Blob")
    anV1_circle.saveGif("ANN_V1_Circle")
    anV2_circle.saveGif("ANN_V2_Circle")
#    anV2_moon.saveGif("ANN_V2_Moon")
