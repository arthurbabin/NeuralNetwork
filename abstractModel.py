import numpy as np
from abc import ABC, abstractmethod, abstractclassmethod
from decisionBoundary import plotDecisionBoundary
import os
from PIL import Image

class AbstractModel(ABC):
    """
    Abstract Class for the models
    """
    iterationPlotRate = 0
    def accuracyScore(self,y,yPredicted):
        """Return the accuracy score of the predictions yPredicted
        compared to the know outcomes y"""
        assert y.shape == yPredicted.shape
        n = y.shape[0] #the number of predictions/outcomes
        s = 0
        for i in range(n):
            s += int(np.array_equal(yPredicted[i],y[i]))
        return s/n

    def logLoss(self,A,y):
        """Compute the log loss between the output of the model and y"""
        return (1/y.shape[0]) * np.sum(-y.T*np.log(A) - (1-y.T)*np.log(1-A))


    def savePlot(self,X,y,name="0.png",iteration=0,show=False):
        if iteration % self.iterationPlotRate == 0:
            plotDecisionBoundary(self,X,y,path=f"images/plots/{self}/{iteration}.png",iteration=iteration)

    def saveGif(self,name="evolution",duration=100, loop=0):
        """
        Concat all the plot images into a gif to illustrate the evolution
        through the training of the model
        """
        images = []
        originFiles = os.listdir(f"./images/plots/{self}/")
        imageFiles = []
        for file in originFiles:
            if file.endswith(".png"):
                imageFiles.append(file)

        for file in imageFiles:
                im = Image.open(f"./images/plots/{self}/{file}")
                images.append(im)

        n = len(imageFiles)
        im = Image.open(f"./images/plots/{self}/{imageFiles[-1]}")
        for i in range(n):
            images.append(im)

        if images:
            images[0].save(
                    f"./images/gif/{name}.gif",
                    save_all=True,
                    append_images = images[1:],
                    optimize = False,
                    duration=duration,
                    loop=loop)

    @abstractmethod
    def predict(self,X,strict=True):
        pass

    @abstractmethod
    def gradientDescent(self):
        pass

    @abstractmethod
    def train(self,X,y,plotLogLossEvolution=False, plotDecisionBoundary=False):
        pass
    

