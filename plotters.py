""" Tools for plotting, including for plotting several related plots as part 
of larger subplots.

"""
import pandas as pd


class PlotData():
    """ Stores and manipulates the data for a single subplot element. E.g. the
    data for a single series (that could later become part of a multiple
    series subplot) or for a single heatmap.

    ATTRIBUTES
    data: dataframe. Data to plot. Potentially modified through use of the 
        methods.

    TODO
    - Add capability to subtract one dataframe from another to produce new 
    series
    """

    def __init__(self, data: pd.DataFrame):
        """
        INPUT
        data: dataframe. Data to plot. If wish to perform statistical tests
            then should provide data without averaging over the relevant
            aspect of the data. E.g. provide data for each participant if 
            which to compute staitstics over participants.
        """
        pass