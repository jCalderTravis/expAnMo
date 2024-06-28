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


class DataToPlot():
    """ Stores data for plots, and facilitates the standardisation and 
    preparation of plot data. Can be used to produce a dataset with the 
    format required for OneSeriesSetPlotter.

    ATTRIBUTES
    pltData: list of pandas dataframes. The data to be plotted but not 
        yet in a standard format. Index of each frame is arbitrary. Columns
        match the columns in the dataframes passed to the constructor 
        as the input rawPltData. Some columns may be missing, if the associated
        varaibles have been averaged out. May have additional column 'xGroup'
        if grouping has been performed.
    """

    def __init__(self, rawPltData: pd.DataFrame | list[pd.DataFrame]):
        """
        INPUT
        rawPltData: pandas dataframe (plot one series), or n-long list of 
            dataframes (plot n series). Index is ignored. Should have three or 
            columns corresponding to (a) the case to which the data point
            belongs, (b) x-value, (c) y-value.
        """
        if not isinstance(rawPltData, list):
            assert isinstance(rawPltData, pd.DataFrame)
            rawPltData = [rawPltData]

        self.pltData = []
        for thisRawPltData in rawPltData:
            thisRawPltData = thisRawPltData.reset_index(drop=True)
            self.pltData.append(thisRawPltData)
            
    
    def findStandardPltData(self, naming: dict | list[dict],
                            allowMissing: bool = False) -> pd.DataFrame:
        """ Return plot data in a standardised format, after checking that it 
        matches the attribute description in the comments on the class 
        OneSeriesSetPlotter. Data must first have been grouped, creating an
        'xGroup' column. An error is raised if this is not present.

        INPUT
        naming: dict (plot one series, or use the same setting for all series),
            or n-long list of dict (plot n series). Specifies the meaning of 
            the columns in self.pltData. Values give column names in 
            self.pltData as strings, and the keys are...
                case: Column contining the case of each data point (if present 
                    in rawPltData) as scalars
                xValue: Column contining the x-value data as scalars
                yValue: Column contining the y-value data as scalars
            If keys are not provided, then default names will be assumed
            ('case', 'xValue', 'yValue'). Data from other columns is not 
            returned, apart from the 'xGroup' column.
        allowMissing: bool. If True, allow cases to be missing data points from
            one or more groups.

        OUTPUT
        standardPltData: list of pandas dataframes. The data to be plotted in 
            standard format. Index of each frame is arbitrary. Has the 
            following four columns: 
                case: scalars. The independent case to which the data belongs
                xValue: scalars. X-data
                yValue: scalars. Y-data
                xGroup: scalars. The plotting group to which each data point 
                    belongs. 
            This class contains methods for processing the data so that each 
            case eventually contains one and only one data point from each 
            group. Only once this has been achieved may pltData be retrieved 
            using self.pltData. This condition can be somewhat relaxed by
            setting allowMissing to True.
        """
        if isinstance(naming, dict):
            naming = [deepcopy(naming) for _ in range(len(self.pltData))]
        else:
            assert len(naming) == len(self.pltData)

        standardPltData = []
        for thisPltData, thisNaming in zip(self.pltData, naming):
            
            if 'xGroup' not in thisPltData:
                raise ValueError('Grouping must first be applied before '
                                 'standardised plot data can be retrieved.')
            assert np.all(np.isin(list(thisNaming.keys()), ['case', 
                                                            'xValue',
                                                            'yValue']))

            colsToKeep = list(thisNaming.values()) + ['xGroup']
            thisPltData = deepcopy(thisPltData.loc[:, colsToKeep])

            standardPltData.append(self._standardiseOneDataframe(thisPltData, 
                                                                 thisNaming))

        checkPltData(standardPltData, allowMissing=allowMissing)
        return standardPltData


    def _standardiseOneDataframe(self, pltData: pd.DataFrame, naming: dict):
        """ Standardise one dataframe, by changing the naming of the columns 
        to standard names.

        INPUT
        pltData: pandas dataframe. 
        naming: dict. One of the naming dicts passed as the naming input to 
            the method findStandardPltData.
        """
        defaultNaming = {
            'case': 'case',
            'xValue': 'xValue',
            'yValue': 'yValue',
            'xGroup': 'xGroup'
        }
        assert np.all(np.isin(list(naming.keys()), list(defaultNaming.keys())))
        for thisKey, thisValue in defaultNaming.items():
            if thisKey not in naming:
                naming[thisKey] = thisValue

        assert len(set(naming.values())) == len(naming.values())
        assert set(naming.values()) == set(pltData.columns)

        renamePlan = {v: k for k, v in naming.items()}
        pltData = pltData.rename(columns=renamePlan, errors='raise')
        assert set(pltData.columns) == set(defaultNaming.keys())

        return pltData
    


    


def checkPltData(pltData: list[pd.DataFrame], allowMissing=False):
    """ Run some checks that self.pltData matches the attribute 
    description in the comments on the class OneSeriesSetPlotter.

    INPUT
    pltData: list of pandas dataframes. 
    allowMissing: bool. If True, allow cases to be missing data points from
        one or more groups.
    """
    for thisData in pltData:
        if set(thisData.columns) != {'case', 'xValue', 'yValue', 'xGroup'}:
            raise ValueError('Dataframe does not have the standard columns')

        errorMsg = 'Each case should contain 1 data point for each group'
        
        cases = np.unique(thisData['case'])
        groups = np.unique(thisData['xGroup'])
        count = thisData.groupby(['case', 'xGroup']).count()

        for thisCase in cases:
            thisCount = count.loc[(thisCase,), :]

            if allowMissing:
                fulfilled = np.all(np.isin(
                    thisCount.index.get_level_values('xGroup'),
                    groups
                ))
            else:
                fulfilled = np.array_equal(
                    thisCount.index.get_level_values('xGroup'),
                    groups
                )
            if not fulfilled:
                raise ValueError(errorMsg)

        if not np.all(count.values == 1):
            raise ValueError(errorMsg)


class OneSeriesSetPlotter():
    """ Stores code and data from making one specific plot of sets of series.
    Especially helpful when writting subclasses of SeriesPlotter, which 
    coordinates the plotting of multiple related series plots.

    ATTRIBUTES
    pltData: list of pandas dataframe. Same as pltData input to the 
        constructor.
    """

    def __init__(self, 
                 pltData: pd.DataFrame | list[pd.DataFrame], 
                 hLine: bool | float = False,
                 allowMissing: bool = False):
        """
        INPUT
        pltData: list of pandas dataframe. Length of list corresponds to the 
            number of series to plot. Index is arbitrary. Has the following 
            columns: 
                case: scalars. The independent case to which the data belongs
                xValue: scalars. X-data
                yValue: scalars. Y-data
                xGroup: scalars. The plotting group to which each data point 
                    belongs. Each case contains one and only one data point 
                    from each group (unless allowMissing is True). 
                    x- and y-values on plots are deteremined 
                    by averaging within each group. Groups are also used for 
                    computing error bars and statisics (where the ordering of 
                    groups -- inferred through the scalars with which they are 
                    denoted -- is assumed to be meaningful). Statistics 
                    cannot be performed if allowMissing is True.
            The class DataToPlot can be used to get data into this format.
        hLine: scalar or False. If not False, then adds a horizonal line at the 
            value given (in data coordinates)
        allowMissing: bool. If True, allow cases to be missing data points from
            one or more groups.
        """
        self.hLine = hLine
        self.allowMissing = allowMissing
        if not isinstance(pltData, list):
            pltData = [pltData]
        self.pltData = pltData

        self.checkPltData()
    

    def checkPltData(self):
        """ Run some checks that self.pltData matches the attribute 
        description in the comments on the class.
        """
        checkPltData(self.pltData, allowMissing=self.allowMissing)
            

    def prepareData(self, runSig=True):
        """ Convert data to the format required for the plotting function.

        INPUT
        runSig: bool. If true, run significance tests.
        """
        self.checkPltData()

        prepData = []
        for seriesIdx, thisData in enumerate(self.pltData):
            
            # WORKING HERE -- moving this over

            if runSig:
                pVal = self.runSigTests(seriesIdx)
                assert pVal.shape == yMean.shape
                sig = pVal < 0.05

                if not np.any(sig):
                    print(
                        'Series at index {} had no significant points'.format(
                            seriesIdx))

                sigCol = {'SIG': sig}
            else:
                sigCol = {}

            prepData.append(pd.DataFrame({
                'X': xMean, 
                'Y': yMean, 
                'ERROR': ySem,
                **sigCol
            }))
        return prepData
    

    def runSigTests(self, seriesIdx):
        """ Run threshold-free cluster-based permutation testing to compute
        significance. Adjacency for the cluster-based test is based on the 
        values of the xGroup variable. Groups are ordered based on the value of
        xValue, and the group one above and one below a specific group are 
        considered adjacent to it.

        INPUT
        seriesIdx: int. The index of the series in the list self.pltData
            that want to run the statistics for. The series will be compared
            to zero.

        OUTPUT
        pVal: 1D numpy array. As long as the number of xGroups. Gives
            the p-value associated with that point.
        """
        import mne

        if self.allowMissing:
            raise NotImplementedError('Currently missing data is not allowed')

        self.checkPltData()
        thisSeries = self.pltData[seriesIdx]
        numXGroups = len(np.unique(thisSeries['xGroup']))

        thisSeries = thisSeries.set_index(['case', 'xGroup'], 
                                          verify_integrity=True)
        thisSeries = thisSeries.sort_index()
        
        yData = thisSeries['yValue']
        yData = yData.unstack('xGroup')

        xData = thisSeries['xValue']
        xData = xData.unstack('xGroup')

        helpKit.checkDfLevels(yData, indexLvs=['case'], colLvs=['xGroup'])
        helpKit.checkDfLevels(xData, indexLvs=['case'], colLvs=['xGroup'])
        assert yData.index.equals(xData.index)
        assert yData.columns.equals(xData.columns)
        assert not np.any(pd.isnull(yData))
        assert not np.any(pd.isnull(xData))
        
        xPos = xData.mean(skipna=False)
        diff = xData.subtract(xPos)
        if not np.all(np.isclose(diff.values, 0, rtol=0)):
            raise ValueError('For statistical testing, for each group, the '+
                                'x-value of the group must be the same for '+
                                'all cases.')
        if not np.all(np.diff(
            xData.columns.get_level_values('xGroup')) > 0):
            raise AssertionError('Bug. Axis should be sorted.')
        
        if not np.all(np.diff(xPos.values) > 0):
            raise AssertionError('Data not ordered. See comment.')
            # A possible cause is that xGroup is differently ordered to xValue.
            # This causes the ambigious situation where it is unclear which 
            # data points should be counted as "neighbours" for cluster based 
            # permutation testing.

        _, _, pVal, _ = mne.stats.permutation_cluster_1samp_test(
                                        yData.values,
                                        threshold={'start':0, 'step':0.005},
                                        adjacency=None)
        assert pVal.shape == (numXGroups,)
        
        return pVal


    def makePlot(self, ax, plotSpec, runSig=True):
        """ Make the plot

        INPUT
        ax: Axis to plot onto
        plotSpec: dict. Contains plot specifications. Required keys are...
                xLabel: str | None.
                yLabel: str | None.
            If the plot contains multiple series the following keys 
            are also required...
                sLabels: list of str. Labels for each series plotted
                sColours: list of str. Colours for each series named in 
                    sLabels
                addLegend: bool.
            May optionally contain a request to add labeled vertical 
            lines to the plots.
                vLines: dict. Keys are strings to be used as labels, 
                    and values are scalars giving the x-location in 
                    data coordinates for the vertical line
                addVLabels: bool. Whether to add labels to the vertical 
                    lines. 
        runSig: bool. If true, run significance tests.
        """
        prepData = self.prepareData(runSig=runSig)

        if len(prepData) > 1:
            assert 'sLabels' in plotSpec
            assert 'sColours' in plotSpec

        plotSpec = deepcopy(plotSpec)
        specDefaults = {
            'sLabels': '',
            'sColours': 'tab:blue',
            'addLegend': False,
            'vLines': {},
            'addVLabels': False
        }
        for thisKey, thisVal in specDefaults.items():
            if thisKey not in plotSpec:
                plotSpec[thisKey] = thisVal

        plotLineWithError(prepData, 
                          sColours=plotSpec['sColours'],
                          sLabels=plotSpec['sLabels'],
                          xLabel=plotSpec['xLabel'],
                          yLabel=plotSpec['yLabel'],
                          addLegend=plotSpec['addLegend'],
                          hLine=self.hLine,
                          ax=ax)
        
        if 'vLines' in plotSpec:
            addVLines(ax, plotSpec['vLines'], plotSpec['addVLabels'])


class Plotter():
    """ Stores code and data for creating multiple related plots. 

    ATTRIBUTES
    shareAxesOk: bool. If true, it is ok if repeated calls to makePlot create
        plots on axes that are shared across those plots.
    """
    shareAxesOk = True

    def prepareForPlotting(self, plotKwargsList):
        """ Run any calculations needed before plotting can take place. E.g.
        computation of shared colour bar bounds.

        pltKwargsList: list of dict. If a single plot is about to be made it 
            will be passed a 1-length list, where the only element is
            the dict pltKwargs, that is about to be passed to the 
            makePlot() method. If N plots are about to be made, it is
            passed an N-length list of pltKwargs dicts, one for each
            of the upcoming plots. Helpful if we want to set the 
            colourbar range on the fly.
        """
        pass


    def makePlot(self, axis, plotSpec, **pltKwargs):
        """ Function plots to the axis, with different related plots
        optionally requested through pltKwargs.

        INPUT
        axis: axis to plot onto
        plotSpec: dict. The output of .mkPlotSpec() specifying plotting 
            details. Possibly modified to suppress labels or colourbars, as 
            appropriate.
        **pltKwargs: (optional) Additional keyword arguments to specify plot
            to produce
        """
        raise NotImplementedError


    def mkPlotSpec(self):
        """ Produce details of how to annotate and decorate the plots.

        OUTPUT
        plotSpec: dict. Contains plot specification to be used by .makePlot()
        """
        raise NotImplementedError


class SeriesPlotter(Plotter):
    """ Makes various related plots, that are all plots of sets of series of 
    data.
    """

    def mkPlotSpec(self):
        """ Produce details of how to annotate and decorate the plots.

        OUTPUT
        plotSpec: dict. Contains plot specifications. Required keys are...
                xLabel: str | None.
                yLabel: str | None.
            If the plot contains multiple series the following keys 
            are also required...
                sLabels: list of str. Labels for each series plotted
                sColours: list of str. Colours for each series named in 
                    sLabels
                addLegend: bool.
            May optionally contain a request to add labeled vertical 
            lines to the plots.
                vLines: dict. Keys are strings to be used as labels, 
                    and values are scalars giving the x-location in 
                    data coordinates for the vertical line
                addVLabels: bool. Whether to add labels to the vertical 
                    lines. 
        """
        raise NotImplementedError


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


    def addLegend(self, gridSection, plotSpec):
        """ Add a legend to the region defined by gridSection

        INPUT
        gridSection: A section of a grid produced using
            matplotlib.gridspec.GridSpec, under which the label will be 
            placed
        plotSpec: dict. Contains plot specifications. Produced by the 
            mkPlotSpec() method of plotter object that we are adding a 
            legend for. See the __init__() method for more details.
        """
        legAx = self.addInvisibleSubplot(gridSection)

        allLines = []
        allLineSpecs = zip(plotSpec['sLabels'], plotSpec['sColours'])
        for thisLabel, thisColour in allLineSpecs:
            allLines.append(legAx.plot([], [], label=thisLabel, 
                            color=thisColour))
        legAx.legend(frameon=False, loc='upper left')


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


def plotLineWithError(pltData: pd.DataFrame | list[pd.DataFrame], 
                        sColours='tab:blue', lineStyles='-', 
                        sLabels='', xLabel=None, yLabel=None, 
                        addLegend=False, hLine=False, ax=None,
                        xLogScale=False, yLogScale=False, sigPos=None):
    """ Line plot with shaded error.

    INPUT
    pltData: Pandas dataframe, or list to plot several series. Each
        contain columns with the following names (different dataframes
        are permited to have different combinations of the optional columns):
            "X": Contains x-values
            "Y": (optional) Contains y-values 
            "ERROR": (optional) Error shading will be plotted between Y+ERROR 
                and Y-ERROR
            "SIG": (optional) Contains boolean values. Where True a line 
                will be plotted indicating significance.
    sColours: colour specification or list as long as list given for pltData,
        Determines colour of each series.
    lineStyles: line specification or list as long as list given for pltData.
        Determines line style of each series.
    sLabels: string or list as long as list given for pltData. Determines
        the label for each series.
    xLabel and yLabel: strings. Labels for axes 
    addLegend: boolean. Whether to display legend.
    hLine: scalar or False. If not False, then adds a horizonal line at the 
        value given
    ax: axis to plot on to (optional)
    xLogScale: bool. Whether to use log scale on x-axis
    yLogScale: bool. Whether to use log scale on y-axis
    sigPos: None | scalar | list as long as pltData. May only be scalar if 
        pltData is a dataframe. Gives the height in data coordinates at which 
        to plot the significance bars. If None, a guess is made as to a good
        place to put the signficance lines.

    OUTPUT
    fig: The matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if isinstance(pltData, pd.DataFrame):
        numSeries = 1
        pltData = [pltData]
    else: 
        assert isinstance(pltData, list)
        numSeries = len(pltData)

    toCheck = [sColours, lineStyles, sLabels]
    for iIn, thisInput in enumerate(toCheck):
        if not isinstance(thisInput, list):
            toCheck[iIn] = [thisInput] * numSeries
        elif (len(thisInput) == 1) and (numSeries > 1):
            toCheck[iIn] = thisInput * numSeries
        else:
            assert len(thisInput) == numSeries
    sColours, lineStyles, sLabels = toCheck

    if sigPos is None:
        minData = np.min([np.min(thisData['Y']) for thisData in pltData])
        maxData = np.max([np.max(thisData['Y']) for thisData in pltData])
        rangeData = maxData - minData
        sigPos = np.linspace(minData - (rangeData/5),
                             minData - (rangeData/10),
                             num=len(pltData))
        
    elif np.ndim(sigPos) == 0:
        assert isinstance(pltData, pd.DataFrame)
        sigPos = [sigPos]
    assert len(sigPos) == len(pltData)

    for thisData in pltData:
        if not np.all(np.diff(thisData['X'])>0):
            raise ValueError('X-data must be in assending order')

    plotHasSig = False
    for iSeries, seriesData in enumerate(pltData):
        if 'Y' in seriesData.columns:
            ax.plot(seriesData['X'].to_numpy(), seriesData['Y'].to_numpy(), 
                    label=sLabels[iSeries], color=sColours[iSeries], 
                    linestyle=lineStyles[iSeries])
            
        if 'ERROR' in seriesData.columns:
            ax.fill_between(seriesData['X'].to_numpy(), 
                    seriesData['Y'].to_numpy()-seriesData['ERROR'].to_numpy(), 
                    seriesData['Y'].to_numpy()+seriesData['ERROR'].to_numpy(), 
                    color=sColours[iSeries], alpha=0.5)
        
        if 'SIG' in seriesData.columns:
            plotHasSig = True
            sigRegions = findSigEnds(seriesData['X'], seriesData['SIG'])
   
            thisLabel = sLabels[iSeries]
            if 'Y' in seriesData.columns:
                # In this case the corresponding series has already had a 
                # label added during the above call to ax.plot. We can suppress
                # this label from appearing a second time in the legend by
                # using an underscore.
                thisLabel = '_' + thisLabel

            for thisStart, thisEnd in sigRegions:
                ax.plot([thisStart, thisEnd], 
                        [sigPos[iSeries]]*2,
                        label=thisLabel,
                        color=sColours[iSeries],
                        linewidth=3)
                
                # Only add to the legend a maximum of one time
                thisLabel = '_'

    if xLogScale:
        ax.set_xscale('log')
        if plotHasSig:
            raise NotImplementedError('See comment')
            # Significance lines are ploted from halway before the curent point
            # to halway to the following point. But this is done before the
            # transformation to log scale, so it will no longer be correct
            # afterwards.
    if yLogScale:
        ax.set_yscale('log')

    if xLabel is not None:
        ax.set_xlabel(xLabel)
    if yLabel is not None:
        ax.set_ylabel(yLabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if hLine is not False:
        ax.axhline(y=hLine, color='k', linestyle='--', linewidth=1)
    if addLegend:
        ax.legend(frameon=False)

    return fig


def findSigEnds(xVals, sig):
    """ From a boolean mask indicating significance find the start and end
    points at which lines should be drawn to indicate significance.

    For a cluster of significant points, significance lines are requested
    from halfway between the first significant point and the preceeding 
    non-significnat point, up to halfway between the last significant point
    and the following non-significant point. If the first or last points
    are significant, lines are only requested up to the first or last point.

    INPUT
    xVals: array of scalar.
    sig: array of bool. Indicates whether correspinding points in xVals are 
        significant.

    OUTPUT
    sigRegions: list of 2-length tuples. Each tuple gives the start and end
        of significant region in x-coordinates.
    """
    xVals = np.asarray(xVals)
    sig = np.asarray(sig)

    assert xVals.shape == sig.shape
    assert np.ndim(sig) == 1
    
    sig = np.concatenate([[0], sig, [0]])
    changes = np.diff(sig)
    assert np.all(np.isin(changes, [0, 1, -1]))
    assert np.all(np.isin(np.cumsum(changes), [0, 1]))

    # Create vector of the first point, the last point, and all the midpoints
    midpoints = np.concatenate([[xVals[0]], 
                                (xVals[:-1]+xVals[1:])/2,
                                [xVals[-1]]]) 
    assert midpoints.shape == changes.shape

    starts = midpoints[changes == 1]
    ends = midpoints[changes == -1]
    assert starts.shape == ends.shape
    sigRegions = list(zip(starts, ends))

    return sigRegions



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


def alsoPlotVLines(plotFun):
    """ Decorates a plotting methods, such that it now also plots labeled
    vertical lines where requested.

    INPUT
    plotFun: Method which performs plotting, in the same form as the 
        makePlot() method of the plotter objects that are passed to the 
        __init__() method of PlotCoordinator (see detailed comments there).
        Note it should be a method, not a function, meaning that it accepts 
        the following keyword arguments...
            self: Calling object instance
            axis: axis to plot onto
            plotSpec: dict. Same dict of the format described in 
                PlotCoordinator.__init__().
        May additionaly accept...
            **pltKwargs: Additional keyword arguments.

    OUTPUT
    inner: Decorated plotFun, that accepts the same inputs as before, and 
        returns the same output as before.
    """

    def inner(self, axis, plotSpec, **pltKwargs):
        output = plotFun(self, axis=axis, plotSpec=plotSpec, **pltKwargs)
        if 'vLines' in plotSpec:
            addVLines(axis, plotSpec['vLines'], plotSpec['addVLabels'])
        return output
    return inner


def addVLines(ax, vLines, addLabels):
    """ Add labeled vertical lines to an axis
    
    INPUT
    ax: axis
    vLines: dict. Keys are strings to be used as labels, and values are 
        scalars giving the x-location in data coordinates for the vertical 
        line
    addLabels: bool. Whether to add labels to the vertical lines.
    """
    if vLines is None:
        return

    mixTransform = ax.get_xaxis_transform()

    for label, xPos in vLines.items():
        ax.axvline(xPos, linestyle='--', color='k', linewidth=1)

        if addLabels:
            ax.text(xPos, 0.5, label, transform=mixTransform,
                    rotation='vertical',
                    horizontalalignment='right',
                    fontsize=2)
            

def checkTickLabelsMatch(tickLabels1, tickLabels2):
    """ Check two sets of tick labels are the same. Can find the tick labels
    on a matplotlib axis using axis.get_xticklabels() or axis.get_yticklabels()
    """
    assert len(tickLabels1) == len(tickLabels2)
    for this1, this2 in zip(tickLabels1, tickLabels2):
        assert this1.get_text() == this2.get_text()
        assert this1.get_position() == this2.get_position()