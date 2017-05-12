from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as st



def kde(x, y, xlabel='x', ylabel='y', ax=None, contours=True):
    """
    Create a kernel density estimation visualization of a 2d point pattern

    Parameters
    ---------

    x : ndarray
        of x-axis values

    y : ndarray
        of y-axis values

    xlabel : str
             The label for the x-axis

    ylabel : str
             The lable for the y-axis

    ax : object
         A Matplotlib axis object to insert the plot into. If
         none provided, the current axis is grabbed

    contours : bool
               If true (default) add contours.
    """
    if ax == None:
        ax = plt.gca()
   
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # Contour plot
    if contours:
        cfset = ax.contourf(xx, yy, f, cmap='Blues')
        cset = ax.contour(xx, yy, f, colors='k')
        ax.clabel(cset, inline=1, fontsize=10)
    else:
        ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
    
    # Contour plot
    # Label plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax
