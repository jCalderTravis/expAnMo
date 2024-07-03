""" Tools for plotting. Especially helpful for plotting several related plots
in rows, as part of larger subplots.
"""
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as pltColours
import matplotlib.cm as mplCm
import pandas as pd
import numpy as np
from . import helpers as helpKit


class HeatmapPlotter(Plotter):
    """ Makes various related heatmap plots.

    ATTRIBUTES
    cBarCenter: None | scalar. Same as input to constructor.
    cMin: scalar. Stores the minimum that should be used on the colour bar. An
        error is raised if this value has not yet been set through a call to
        prepare for plotting.
    cMax: scalar. Same as cMin but for max
    """

    def __init__(self, cBarCentre=None):
        """
        INPUT
        cBarCentre: None | scalar. If not none, ensure the colour bar is 
            centred on this value.
        """
        self.cBarCentre = cBarCentre
        self.cMin = None
        self.cMax = None
    
    @property
    def cMin(self):
        assert self._cMin is not None
        return self._cMin

    @cMin.setter
    def cMin(self, value):
        self._cMin = value

    @property
    def cMax(self):
        assert self._cMax is not None
        return self._cMax

    @cMax.setter
    def cMax(self, value):
        self._cMax = value


    def prepareForPlotting(self, plotKwargsList):
        """ Computre shared colour bar bounds prior to plotting.

        plotKwargsList: list of dict. See comments on this version of the 
            function in the base class.
        """
        super().prepareForPlotting(plotKwargsList)
        self.setCbarRange(plotKwargsList)
    

    def mkPlotSpec(self):
        """ Produce details of how to annotate and decorate the plots.

        OUTPUT
        plotSpec: dict. Contains plot specifications. Required keys are...
            xLabel: str | None.
            yLabel: str | None.
            cLabel: str | None. Label for the colourbar
            cNorm: matplotlib normalise class. Describes the way in 
                which to scale the data for all colourmaps in the 
                plots
            cMin, cMax: scalar. Passed to cNorm as vmin and vmax. 
                Sets min and max of the colourbar scale
            cMap: str. Colourmap to use
            addCBar: bool. Whether to add a colourbar
        May optionally contain a request to add labeled vertical 
        lines to the plots.
            vLines: dict. Keys are strings to be used as labels, 
                and values are scalars giving the x-location in 
                data coordinates for the vertical line
            addVLabels: bool. Whether to add labels to the vertical 
                lines. 
        """
        plotSpec = dict()
        plotSpec['xLabel'] = ''
        plotSpec['yLabel'] = ''
        plotSpec['cLabel'] = ''
        plotSpec['cNorm'] = pltColours.Normalize
        plotSpec['cMin'] = self.cMin
        plotSpec['cMax'] = self.cMax
        plotSpec['cMap'] = 'RdBu_r'
        plotSpec['addCBar'] = True

        return plotSpec
        

    def findColourData(self, **pltKwargs):
        """ Finds the data that should be plotted on a heatmap, when 
        .makePlot() is called using the same plotKwargs. Should not call 
        directly. Instead call safeFindColourData which has the same 
        inputs and outputs but runs additional checks.
        
        INPUT
        **pltKwargs: (optional) Additional keyword arguments to specify plot
            to produce. Match those that are passed to .makePlot().

        OUTPUT
        colourData: pandas dataframe. Values are the colour values due to be
            plotted.
        """
        raise NotImplementedError
    

    def safeFindColourData(self, **pltKwargs):
        """ Same as the method .findColourData() but also checks that the 
        retrieved data is in range.
        """
        colourData = self.findColourData(**pltKwargs)
        if len(colourData) == 0:
            below = False
            above = False
        else:
            below = np.min(colourData.to_numpy()) < self.cMin
            above = np.max(colourData.to_numpy()) > self.cMax

        if below or above:
            raise ValueError('Tring to plot data outside colour bar limits')
        
        return colourData
    

    def setCbarRange(self, pltKwargsList):
        """ For a range of heatmap plots requested using different keyword 
        arguments, find the corresponding plot data. Accross all plot data
        find the highest and lowest values that would be plotted in the 
        heatmaps. Store these highest and lowest values for use in future when
        actually making the plots, so that a consistent colour bar can be 
        set for all plots.

        INPUT
        pltKwargsList: list of dict. See comments for prepareForPlotting. 

        ACTIONS
        Sets self.cMin and self.cMax
        """
        allMin = []
        allMax = []

        for theseKwargs in pltKwargsList:
            colourData = self.findColourData(**theseKwargs)
            if len(colourData) != 0:
                allMin.append(np.min(colourData.to_numpy()))
                allMax.append(np.max(colourData.to_numpy()))

        if len(allMin) == len(allMax) == 0:
            allMin = [-1]
            allMax = [1]
            
        cValMin = np.min(allMin)
        cValMax = np.max(allMax)

        # WORKING HERE -- moving accross
        if self.cBarCentre is not None:
            self.cMin, self.cMax = findCenteredScale(cValMin, cValMax,
                                                        self.cBarCentre)
        else:
            self.cMin = cValMin
            self.cMax = cValMax


def findColourBarSpec(plotSpec):
    """ Find requested properties for the colour bar using the full plot
    specification.

    INPUT
    plotSpec: dict. Of the format produced by HeatmapPlotter.mkPlotSpec()
    
    OUTPUT
    cbarSpec. dict. A subset of the dict produced by .mkPlotSpec(). 
        Specifically the keys are cMap, addCBar, cNorm, cMin, and cMax.
    """
    toKeep = ['cMap', 'addCBar', 'cNorm', 'cMin', 'cMax']
    return {k: v for k, v in plotSpec.items() if k in toKeep}


def makeColourMapper(plotSpec):
    """ Returns the matplotlib ScalarMappable that is to be used for 
    mapping scalars to colours.

    INPUT
    plotSpec: dict. Of the format produced by HeatmapPlotter.mkPlotSpec()
    
    OUTPUT
    scalarMappable: matplotlib ScalarMappable instance.
    """
    Normaliser = plotSpec['cNorm']
    cbarNorm = Normaliser(vmin=plotSpec['cMin'], vmax=plotSpec['cMax'])
    scalarMappable = mplCm.ScalarMappable(norm=cbarNorm, 
                                            cmap=plotSpec['cMap'])
    return scalarMappable


def findCenteredScale(lower, upper, centre):
    """ Find a scale that runs from at least lower to upper, but is extended
    (as little as possible) to ensure that it is centered on centre
    """
    assert upper > lower
    maxDiff = max(np.absolute([upper-centre, lower-centre]))

    newMin = centre - maxDiff
    newMax = centre + maxDiff

    assert(newMin <= lower)
    assert(newMax >= upper)
    return newMin, newMax


class PlotCoordinator():
    """ Class for coordinating the plotting of multiple related and unrelated
    plots. Also takes care of details such as shared colourbars.

    ATTRIBUTES
    plotters: Same as input to __init__
    isFirstPlot: List of bool. As long as plotters. True if this plotter has 
        not yet been used.
    firstAx: List. As long as plotters. Elements are axis, and specifically 
        the first axis plotted using each plotter. Elelments are None, where
        a plotter has not yet been used. Useful for setting up axis sharing.
    fig: None | figure. The figure we are plotting onto.
    """

    def __init__(self, plotters):
        """ 
        INPUT
        plotters: list of SeriesPlotter or HeatmapPlotter instances. Each 
            instance stores the code and data for creating one type of plot. 
        """
        assert isinstance(plotters, list)
        if len(plotters) == 0:
            raise ValueError('Not given anything to plot')
        self.plotters = plotters
        self.isFirstPlot = [True] * len(plotters)
        self.firstAx = [None] * len(plotters)
        self.fig = None


    def plotSubplotRows(self, colKwargs=None, colLabels=None, 
                        xKeepLabels=False):
        """ Plot rows of subplots, one row for each plotter (that was passed 
        to the PlotCoordinator constructor).
    
        INPUT
        colKwargs: None | list of dict. If None, plot a single column of 
            subplots. If list, plots as many subplot columns as the length
            of the list. The contents of each dict are passed as keyword 
            arguments to the .makePlot() method of each plotter to determine 
            the specific plot drawn in each column. E.g. the contents of the 
            second dict are passed as keyword arguments to the .makePlot()
            method of the third plotter to determine what is plotted in the 
            second subpolot column of the the third subplot row.
        colLabels: None | list of str. List as long as colKwargs. Detemines
            the title for each subplot column.
        xKeepLabels: bool. If true, never supress tick labels on the x-axis,
            even if these are duplicates of the x-labels on adjacent plots.

        OUTPUT
        fig: The figure that was plot to.
        """
        if (colKwargs is not None) and (colLabels is not None):
            assert len(colKwargs) == len(colLabels)
            nCols = len(colKwargs)
        elif (colKwargs is None) and (colLabels is None):
            colKwargs = [dict()]
            colLabels = []
            nCols = 1
        else:
            raise AssertionError('Not sure whether to plot clusters or not!')
        
        for thisPlotter in self.plotters:
            thisPlotter.prepareForPlotting(colKwargs)

        isHeatmap = [self.findIfHeatmap(thisPlotter.mkPlotSpec())
                     for thisPlotter in self.plotters]

        headerHeight = 1.5
        footerHeight = 0.7
        figHeight = headerHeight + footerHeight + (1.5*len(self.plotters))
        edges = 1.8
        figWidth = edges + nCols + 1.2
        edgesFrac = edges / (2 * figWidth)
        self.fig = plt.figure(figsize=[figWidth, figHeight])
        weights = [1]*(nCols+1) # Extra column for colourbar / legend
        if np.all(isHeatmap):
            weights[-1] = 0.3
        else:
            weights[-1] = 0.5 # Legends are often wider
        top = 1 - (headerHeight / figHeight)
        bottom = footerHeight / figHeight 
        grid = GridSpec(len(self.plotters), nCols+1, width_ratios=weights, 
                            figure=self.fig, hspace=1, top=top, bottom=bottom, 
                            left=edgesFrac, right=1-edgesFrac) 
                            # Extra column for any possible colourbars

        for iPlotter, thisPlotter in enumerate(self.plotters):
            for iCol, thisColKwargs in enumerate(colKwargs):
                self.addplot(iPlotter, grid[iPlotter, iCol], thisColKwargs,
                             xKeepLabels=xKeepLabels)

            plotSpec = self.plotters[iPlotter].mkPlotSpec()

            self.addSharedXLabel(grid[iPlotter, 0:nCols], plotSpec)
            if self.findIfHeatmap(plotSpec) and plotSpec['addCBar']:
                self.addColourBar(grid[iPlotter, nCols], plotSpec)
            if self.findIfMultipleSeries(plotSpec) and plotSpec['addLegend']:
                self.addLegend(grid[iPlotter, nCols], plotSpec)

        if nCols > 1:
            for iClst, thisClstLabel in enumerate(colLabels):
                self.addSharedTitle(grid[:, iClst], thisClstLabel, 
                                            rotation=90, fontweight='bold')

        return self.fig
    

    def addplot(self, plotterIdx, gridSection, pltKwargs, xKeepLabels=False):
        """ Add an axis, and a specific plot to the area defined by 
        gridSection. For each individual plotter object, all plots created 
        using this function will share the same axes if their attribute
        shareAxesOk is True. In this case Y-ticks and labels on all but the 
        first plot will be hidden.

        INPUT
        plotterIdx: The index of the plotter object in self.plotters that is
            to be used for creating the plot.
        gridSection: A section of a grid produced using
            matplotlib.gridspec.GridSpec. 
        pltKwargs: Additional keyword arguments to pass to the .makePlot()
            method of the plotter
        xKeepLabels: bool. If true, never supress tick labels on the x-axis,
            even if these are duplicates of the x-labels on adjacent plots.
        """
        shareAxes = self.plotters[plotterIdx].shareAxesOk

        if self.isFirstPlot[plotterIdx]:
            assert self.firstAx[plotterIdx] is None
            thisAx = self.fig.add_subplot(gridSection)
            self.firstAx[plotterIdx] = thisAx
        elif shareAxes: 
            thisAx = self.fig.add_subplot(gridSection, 
                                            sharex=self.firstAx[plotterIdx],
                                            sharey=self.firstAx[plotterIdx])
        else:
            thisAx = self.fig.add_subplot(gridSection)
            
        plotSpec = self.plotters[plotterIdx].mkPlotSpec()
        if plotSpec is None:
            raise ValueError('No plot spec was returned')
        plotSpec = deepcopy(plotSpec)

        plotSpec['xLabel'] = None
        if not self.isFirstPlot[plotterIdx]:
            plotSpec['yLabel'] = None
            plotSpec['addVLabels'] = False
        plotSpec['addCBar'] = False
        plotSpec['addLegend'] = False

        self.plotters[plotterIdx].makePlot(axis=thisAx, 
                                            plotSpec=plotSpec,
                                            **pltKwargs)
        
        if (not self.isFirstPlot[plotterIdx]) and shareAxes:
            checkTickLabelsMatch(
                self.firstAx[plotterIdx].get_yticklabels(),
                thisAx.get_yticklabels()
            )
            checkTickLabelsMatch(
                self.firstAx[plotterIdx].get_xticklabels(),
                thisAx.get_xticklabels()
            )
            plt.setp(thisAx.get_yticklabels(), visible=False)
            if not xKeepLabels:
                plt.setp(thisAx.get_xticklabels(), visible=False)

        self.isFirstPlot[plotterIdx] = False

    
    def addSharedTitle(self, gridSection, titleTxt, rotation=0,
                        fontweight='normal'):
        """ Add a title to the region defined by gridSection

        INPUT
        gridSection: A section of a grid produced using
            matplotlib.gridspec.GridSpec, under which the label will be 
            placed
        titleTxt: str.
        rotation: scalar. Roation in degrees of the text.
        """
        titleAx = self.addInvisibleSubplot(gridSection)
        titleAx.set_title(titleTxt, 
                            rotation=rotation,
                            fontweight=fontweight)


    def addSharedXLabel(self, gridSection, plotSpec):
        """ Add a x-label to the region defined by gridSection

        INPUT
        gridSection: A section of a grid produced using
            matplotlib.gridspec.GridSpec, under which the label will be 
            placed
        plotSpec: dict. Contains plot specifications. Produced by the 
            mkPlotSpec() method of plotter object that we are adding an
            x-label for. See the __init__() method for more details.
        """
        if 'xLabel' not in plotSpec:
            return
        xLabelAx = self.addInvisibleSubplot(gridSection)
        xLabelAx.set_xlabel(plotSpec['xLabel'])


    def addColourBar(self, gridSection, plotSpec):
        """ Add a colourbar to the region defined by gridSection

        INPUT
        gridSection: A section of a grid produced using
            matplotlib.gridspec.GridSpec, under which the label will be 
            placed
        plotSpec: dict. Contains plot specifications. Produced by the 
            mkPlotSpec() method of plotter object that we are adding a 
            colourbar for. See the __init__() method for more details.
        """
        scalarMappable = makeColourMapper(plotSpec)
        cbarAx = self.fig.add_subplot(gridSection)
        cbar = self.fig.colorbar(scalarMappable, cax=cbarAx, fraction=0.005)
        cbar.set_label(plotSpec['cLabel'])

       def addInvisibleSubplot(self, gridSection):
        """ 
        INPUT
        gridSection: A section of a grid produced using
            matplotlib.gridspec.GridSpec. The invisible subplot axes will 
            occupy this space

        OUTPUT
        invisAx: matplotlib axes.
        """
        invisAx = addInvisibleSubplotToFig(self.fig, gridSection=gridSection)
        return invisAx
    

    def findIfHeatmap(self, plotSpec):
        """ Infer from attributes whether this is a heatmap. Raise exception
        if detect unexpected combinations.

        INPUT
        plotSpec: dict. Contains plot specifications. Produced by the 
            mkPlotSpec() method of plotter object that we are investigating. 
            See the __init__() method for more details.

        OUTPUT
        isHeatmap: bool.
        """
        containsKeys = self.findIfPlotSpecKeys(plotSpec,
                                ['cNorm', 'cMin', 'cMax', 'cMap', 'addCBar'])

        if containsKeys == 'all':
            isHeatmap = True
        elif containsKeys == 'none':
            isHeatmap = False
        else:
            raise AssertionError('Not sure if this is a heatmap or not')

        return isHeatmap
        

    def findIfMultipleSeries(self, plotSpec):
        """ Infer from attributes whether this plot contains multiple series. 
        Raise exception if detect unexpected combinations.

        INPUT
        plotSpec: dict. Contains plot specifications. Produced by the 
            mkPlotSpec() method of plotter object that we are investigating. 
            See the __init__() method for more details.

        OUTPUT
        containsSeries: bool.
        """
        containsKeys = self.findIfPlotSpecKeys(plotSpec,
                                        ['sLabels', 'sColours', 'addLegend'])

        if containsKeys == 'all':
            containsSeries = True
        elif containsKeys == 'none':
            containsSeries = False
        else:
            raise AssertionError('Not sure if this contains series or not')

        return containsSeries


    def findIfPlotSpecKeys(self, plotSpec, keysToCheck):
        """ Find whether a plot spec conatains all, some or none of the 
        specified keys. 

        INPUT
        plotSpec: dict. Contains plot specifications. Produced by the 
            mkPlotSpec() method of plotter object that we are investigating. 
            See the __init__() method for more details.
        keys: list of str. Keys to check

        OUTPUT
        containsKeys: str. 'all', 'some', 'none'
        """
        matches = np.isin(keysToCheck, list(plotSpec.keys()))

        if np.all(matches):
            containsKeys = 'all'
        elif not np.any(matches):
            containsKeys = 'none'
        else:
            containsKeys = 'some'
        return containsKeys



def plotHeatmapFromDf(df, unevenAllowed=False, plotFun='pcolormesh', 
                        xLabel=None, yLabel=None, cLabel=None,
                        xtickVals=None, ytickVals=None,
                        ax=None, cbarMode='auto', 
                        cbarSpec=None):
    """ Plot a heatmap from a dataframe where the indicies and columns
    are numeric values.

    INPUT
    df: Pandas dataframe. Should have one column-level and one index-level, 
        both of which should contain numeric values. The index values will 
        form the y-values in the heatmap, and the column values will form the 
        x-values in the heatmap 
    unevenAllowed: boolean. Allow the values for the x or y axis to be 
        unevenly spaced? The plot will take the spacing into account.
    plotFun: string. Either 'pcolormesh' or 'imshow'. Determines which 
        matplotlib function to use for plotting. Must use 'pcolormesh' if  
        unevenAllowed=True
    xLabel and yLabel: str | None
    cLabel: string | None. Sets label for colour bar
    xtickVals and ytickVals: numpy arrays specifying the locations of the 
        ticks (on the same scale as the data)
    ax: axis to plot on to. If none provided, creates new figure.
    cbarMode: 'auto' | 'predef'. Whether to automatically determine the 
        colourbar range to include all data, or to use a predefined scale
    cbarSpec: dict. Required keys depend on cbarMode. Always require...
            cMap: str. Colourmap to use
            addCBar: bool. Whether to add a colourbar
        If cbarmode is 'auto' then addionally require...
            cCenter: scalar | None. If a scalar the colour bar will be 
                centered around this value.
        If cbarMode is 'predef' then additonally require...
            cNorm: matplotlib normalise class. Describes the way in which
                to scale the data for all colourmaps in the plots
            cMin, cMax: scalar. Passed to cNorm as vmin and vmax. Sets
                min and max of the colourbar scale

    OUTPUT
    Returns the figure produced/modified
    vmin: Lowest value on the resulting colour bar scale
    vmax: Highest value on the resulting colour bar scale
    """
    if cbarSpec is None:
        assert cbarMode == 'auto'
        cbarSpec = {'cCenter': None, 
                    'cMap': 'RdBu_r', 
                    'addCBar': True}

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    specs = list(cbarSpec.keys())
    specs.sort()
    if cbarMode == 'auto':
        assert specs == ['addCBar', 'cCenter', 'cMap']
    elif cbarMode == 'predef':
        assert specs == ['addCBar', 'cMap', 'cMax', 'cMin', 'cNorm']
    else:
        raise ValueError('Unexpected input')

    if unevenAllowed and (plotFun != 'pcolormesh'):
        raise ValueError('Must use plotFun=\'pcolormesh\' if '+ 
                         'unevenAllowed=True')

    df = df.sort_index(axis=0)
    df = df.sort_index(axis=1)

    yVals = df.index.to_numpy()
    xVals = df.columns.to_numpy()
    cVals = df.to_numpy()

    yDiffs = np.diff(yVals)
    xDiffs = np.diff(xVals)
    yEven = np.allclose(yDiffs, yDiffs[0], rtol=0, atol=10e-8)
    xEven = np.allclose(xDiffs, xDiffs[0], rtol=0, atol=10e-8)
    allEven = yEven and xEven
    if (not allEven) and (not unevenAllowed):
        raise AssertionError('Either the columns or the rows are '+
                         ' unequally spaced.')

    # Work out colour bar range
    if cbarMode == 'predef':
        vmin = cbarSpec['cMin']
        vmax = cbarSpec['cMax']
        assert cbarSpec['cMax'] > cbarSpec['cMin']

        cbarNormClass = cbarSpec['cNorm']
        cbarNorm = cbarNormClass(vmin=vmin, vmax=vmax)

    elif cbarMode == 'auto':
        minCVal = np.amin(cVals)
        maxCVal = np.amax(cVals)
        if cbarSpec['cCenter'] is not None:
            vmin, vmax = findCenteredScale(minCVal, maxCVal, 
                                                   cbarSpec['cCenter'])
        else:
            vmin = minCVal
            vmax = maxCVal

        cbarNorm = pltColours.Normalize(vmin=vmin, vmax=vmax)
    else:
        raise ValueError('Unrecognised input')

    # Do the plotting ...
    # TODO I always find orientation/order of labels on colourmaps so confusing.
    # Need to check charefully I got the order correct.
    if plotFun == 'imshow':
        axIm = ax.imshow(cVals, origin='lower', 
                            cmap=cbarSpec['cMap'], 
                            norm=cbarNorm)

        # How far along the axis should each tick go, as a fraction
        # TODO Not to confident that the ticks have been set correctly 
        computeFrac = lambda tickVals, allVals : ((tickVals - np.min(allVals)) 
                                        / (np.max(allVals) - np.min(allVals)))
        if xtickVals is not None:
            xtickFrac = computeFrac(xtickVals, xVals)
        if ytickVals is not None:
            ytickFrac = computeFrac(ytickVals, yVals) 

        # Convert fraction into the units used by imshow (number of data 
        # points along)
        computePos = lambda tickFrac, allVals: tickFrac * (len(allVals)-1)
        if xtickVals is not None:
            xtickPos = computePos(xtickFrac, xVals) 
        if ytickVals is not None:
            ytickPos = computePos(ytickFrac, yVals) 
        
        if xtickVals is not None:    
            ax.set_xticks(xtickPos) 
            ax.set_xticklabels(xtickVals)
        if ytickVals is not None:
            ax.set_yticks(ytickPos) 
            ax.set_yticklabels(ytickVals)

    elif plotFun == 'pcolormesh':
        axIm = ax.pcolormesh(xVals, yVals, cVals, 
                                shading='nearest', 
                                cmap=cbarSpec['cMap'], 
                                norm=cbarNorm)
        
        if xtickVals is not None:    
            ax.set_xticks(xtickVals) 
        if ytickVals is not None:
            ax.set_yticks(ytickVals)
    else:
        raise ValueError('Requested plotFun not recognised.') 

    if cbarSpec['addCBar']:
        colourBar = plt.colorbar(axIm, ax=ax)
        if cLabel is not None:
            colourBar.set_label(cLabel)

    if xLabel is not None:
        ax.set_xlabel(xLabel) 
    if yLabel is not None:
        ax.set_ylabel(yLabel) 

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig, vmin, vmax


def addInvisibleSubplotToFig(fig, gridSection=None, pos=None):
    """ Make an invisible subplot. Useful for adding shared axes and legends.
    
    INPUT
    fig: Figure to add the subplot to.
    gridSection: A section of a grid produced using
        matplotlib.gridspec.GridSpec. The invisible subplot axes will 
        occupy this space. One of gridSection or pos must be provided.
    pos: The position for the subplot to occupy specified with three integers
        (nrows, ncols, index). Could also be a three digit integer. One of 
        gridSection or pos must be provided.

    OUTPUT
    invisAx: matplotlib axes.

    EXAMPLE USAGE
    ax = addInvisibleSubplotToFig(fig, pos)
    ax.set_xlabel(label)
    """
    if gridSection is not None:
        assert pos is None
        locSpecifier = [gridSection]
    else:
        assert pos is not None
        locSpecifier = pos

    invisAx = fig.add_subplot(*locSpecifier, frameon=False)
    invisAx.tick_params(labelcolor='none', which='both',
                            top=False, bottom=False, 
                            left=False, right=False)
    return invisAx


def applyDefaultAxisProperties(axis):
    """ Apply some defaults to the appearance of the axes

    INPUT
    axis: The axis we want to modify
    """
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.axhline(color='k', linewidth=1)
            

def checkTickLabelsMatch(tickLabels1, tickLabels2):
    """ Check two sets of tick labels are the same. Can find the tick labels
    on a matplotlib axis using axis.get_xticklabels() or axis.get_yticklabels()
    """
    assert len(tickLabels1) == len(tickLabels2)
    for this1, this2 in zip(tickLabels1, tickLabels2):
        assert this1.get_text() == this2.get_text()
        assert this1.get_position() == this2.get_position()