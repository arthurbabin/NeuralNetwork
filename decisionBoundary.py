import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def plotDecisionBoundary(model,X,y,path="",colorLevels=64,iteration=1):
    """
    Plot the input data [X] on a bidimensional graph, colored depending
    on the corresponding output [y] and with a line representing the
    decision boundary of the [model].
    Also save to figure to [path] in png format if specified.
    """
    

    textColor = "9F98DF"
    bgColor = "030115"
    plt.rcParams.update({
        "font.family":"DejaVu Sans",
        "text.color":textColor,
        "axes.labelcolor":textColor,
        "axes.facecolor":bgColor,
        "axes.edgecolor":textColor,
        "axes.titlesize":15,
        "axes.labelsize":13,
        "axes.labelpad":8,
        "grid.color":textColor,
        "xtick.color":textColor,
        "ytick.color":textColor,
        "figure.facecolor":bgColor,
        "legend.title_fontsize":13,
        "legend.fontsize":13,
        })

    min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
    min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1
    x1grid = np.arange(min1, max1, 0.01)
    x2grid = np.arange(min2, max2, 0.01)
    xx, yy = np.meshgrid(x1grid, x2grid)
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    grid = np.hstack((r1,r2))
    yhat = model.predict(grid,strict=False)
    yhat = yhat[:, 0]
    zz = yhat.reshape(xx.shape)
    f = plt.figure()
    f.set_figwidth(14)
    f.set_figheight(9)
    c = plt.contourf(xx, yy, zz, cmap='seismic', levels=colorLevels, vmin=0, vmax=1)
    plt.clim(0,1)
    plt.colorbar(c,label="Predicted Output")

    sns.scatterplot(
            x=X[:,0],
            y=X[:,1],
            hue=y.flatten(),
            palette="seismic",
            s=30,
            edgecolor="black"
            )

    plt.xlabel("feature n°1")
    plt.ylabel("feature n°2")
    plt.legend(title="Real output")
    plt.title(f"Decision Boundary of the {str(model)}\naccuracy={'%.3f' % model.accuracyScore(model.predict(X),y)} Iteration n°{iteration}")
    if path:
        f.savefig(path,format="png")
    else:
        plt.show()
    plt.close()
