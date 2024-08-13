import os
import pandas as pd
from . import plotters

currentPath = os.path.dirname(os.path.realpath(__file__))
testResultsDir = os.path.join(currentPath, '__test_results__')

def test_plotHeatmapFromDf():
    """ Saves a plot in the folder given by the varaible testResultsDir. The 
    plot should be of a heat map with four squares. Each square on the bottom 
    should be lighter than the square directly above it. Each square on the 
    left should be lighter than the square directly to the right.
    """
    cbarMode = 'auto'
    cbarSpec = {'cCenter': 0, 
                'cMap': 'RdBu_r', 
                'addCBar': True}
    df = pd.DataFrame()
    df.loc[1, 1] = 0
    df.loc[2, 1] = 10
    df.loc[1, 2] = 50
    df.loc[2, 2] = 100

    fig = plotters.plotHeatmapFromDf(df, cbarMode=cbarMode, cbarSpec=cbarSpec)
    plotters.saveFigAndClose(
        os.path.join(testResultsDir, 'plotHeatmapFromDf.pdf'),
        figsToSave = fig)
