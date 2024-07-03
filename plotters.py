""" Tools for plotting, including for plotting several related plots as part 
of larger subplots.

"""
from copy import deepcopy
import numpy as np
import pandas as pd
import scipy.stats as spStats
import matplotlib.pyplot as plt
from . import helpers


def _makeCarefulProperty(name: str, keys: list[str]):
        """ Make a property which is a dict, and which can only be used in a 
        specific way without an error being raised. Specifically, the property
        will raise an error if...
            - Called before being set
            - Set again after already being set
            - Called but the keys of the returned dict don't match the 
            input 'keys'

        INPUT
        name: str. Name to use for the underying attribute. A '_' will be 
            prepended.
        keys: list[str]. The keys that expect in the dict returned when calling
            this property.
        """
        attrName = '_'+name

        @property
        def thisProp(self):
            value = getattr(self, attrName)
            if value is None:
                raise ValueError('The {} attribute has not yet been '+
                                 'set.'.format(name))
            if set(value.keys()) != set(keys):
                raise ValueError('The {} attribute does not have the '+
                                 'expected keys.'.format(name))
            return value
        
        @thisProp.setter
        def thisProp(self, value):
            oldValue = getattr(self, attrName)
            if oldValue is not None:
                raise ValueError('The {} attribute cannot be set '+
                                 'twice.'.format(name))
            if set(value.keys()) != set(keys):
                raise ValueError('The {} attribute does not have the '+
                                 'expected keys.'.format(name))
            setattr(self, attrName, value)

        return thisProp


class Formatter():
    """ Stores and manipulates a dataset. Helpful for getting a dataset into
    the format required for making a specific plot.

    ATTRIBUTES
    data: pandas dataframe storing the current version of the dataset
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        INPUT
        data: pandas dataframe. The data that we want to manipulate. The 
            index will be ignored. Only a deepcopy of the input dataframe
            will be maniptulated.
        """
        data = deepcopy(data)
        self.data = data.reset_index(drop=True) 


    def average(self, within: str | list[str],
                    keep: list[str] = None, 
                    drop: list[str] = None,
                    checkEqual: bool = True):
        """ Average the data within specified groups, and optionally dropping
        some columns.

        within: str | list[str]. The names of one or more columns. 
            Averaging is performed seperately for each unique combination of 
            these variables (i.e. we average within the groups defined by
            these variables).
        keep: None | list[str]. List of column names. If provided, checks that 
            all columns in the dataset match list(within) + keep + drop. 
            Provide (possibly empty) lists for both keep and drop or neither of 
            them.
        drop: None | list[str]. List of column names. If provided, these 
            columns will be dropped. Provide (possibly empty) lists for both 
            keep and drop or neither of them.
        checkEqual: bool. If true, check that there are the same number
            of cases in each group.
        """
        if not isinstance(within, list):
            assert isinstance(within, str)
            within = [within]

        if (keep is None) and (drop is None):
            pass
        elif (keep is not None) and (drop is not None):
            expectCols = within + keep + drop
            assert len(expectCols) == len(set(expectCols)), 'Duplicates'
            expectCols = set(expectCols)
            assert expectCols == set(self.data.columns), 'Unexpected columns'

            self.data = self.data.drop(columns=drop)
        else:
            raise ValueError('Provide both ''keep'' and ''drop'' or neither.')

        grouped = self.data.groupby(within)
        if checkEqual:
            checkGroupsEqual(grouped)
        avData = grouped.mean()

        helpers.checkDfLevels(avData, indexLvs=within)
        avData = avData.reset_index(allow_duplicates=False)
        assert len(set(avData.columns)) == len(avData.columns)
        if keep is not None:
            assert set(avData.columns) == set(within + keep)

        self.data = avData


    def bin(self, strategy: str, 
                opts: dict, 
                binName: str,
                sepFor: None | str | list[str] = None):
        """ Bin data and add a new column to the data, givng the bin to
        which each case belongs.

        INPUT
        strategy: str. Binning strategy. Options are...
            'col': The values in a column directly as the bins. 
            'specifiedEdges': The x-values are binned based on specified bin
                edges.
        opts: dict. Depends on the value of strategy. The keys and 
            values for each strategy option are as follows.
                'col': Keys and values...
                    'col': The name of the column to use directly as the bins
                'specifiedEdges': Keys and values...
                    'col': The name of the column upon which the binning is 
                        to be conducted.
                    'edges': Sequence of scalars giving the bin edges to use.
        binName: str. The name of the column that will be added to the data
            giving the bin to which each case belongs.
        sepFor: None | str | list[str]. Name or names of columns in the 
            dataset. Bining is performed seperately for each unique 
            combination of these variables.

        TODO Add option to bin by percentile See groupby(...)[var].rank(pct=true)
        """
        startCases = len(self.data)
        startCols = deepcopy(list(self.data.columns))

        if sepFor is None: 
            self.data = self._binAll(self.data, strategy, opts, binName)
        else:
            if isinstance(sepFor, str):
                sepFor = [sepFor]
            assert isinstance(sepFor, list)

            grouped = self.data.groupby(sepFor)
            processed = []
            for _, group in grouped:
                thisProcessed = self._binAll(group, strategy, opts, binName)
                processed.append(thisProcessed)

            processed = pd.concat(processed, verify_integrity=True)
            assert len(processed) == len(startCases)
            assert list(processed.columns) == startCols + [binName]
            self.data = processed


    def _binAll(self, data: pd.DataFrame, strategy: str, opts: dict, 
                  binName: str):
        """ Perform binning of all input data. Similar functionality to .bin()
        except operates on the input, not on self.data, and cannot perform 
        binning seperately for different subsets of the data.

        INPUT
        data: pandas dataframe to perform the binning on. Data is deepcopied
            to ensure no mutation of the input data.
        strategy, opts, binName: Same as input to the method .bin()

        OUTPUT
        data: pandas dataframe. Copy of the input data with the binning 
            perormed and the results of the binning stored in a new column
            with name binName.
        """
        data = deepcopy(data)
        if binName in data.columns:
            raise ValueError('Binning data cannot be assigned to requested '+
                             'columns because it already exists.')

        if strategy == 'col':
            assert set(opts.keys()) == {'col'}
            data[binName] = data[opts['col']]
        
        elif strategy == 'specifiedEdges':
            assert set(opts.keys()) == {'col', 'edges'}
            relCol = opts['col']
            data[binName] = pd.cut(data[relCol],
                                    bins=opts['edges'],
                                    labels=False)
            assert not np.any(np.isnan(data[binName]))
        else:
            raise ValueError('Unrecognised binning option')
        
        return data
            

    def runSeriesStats(self, xVar: str, yVar: str, bin: str, 
                       obsID: None | str = None,
                       runPerm: bool = True,
                       checkEqual: bool = True,
                       checkSame: bool = True,
                       checkOrder: bool = True):
        """ Run statistics on series data.

        INPUT
        xVar: str. Name of column giving the values of the independent 
            variable.
        yVar: str. Name of column giving the values of the dependent variable.
        bin: str. Name of column giving the bin to which each datapoint 
            belongs. Averaging, and statistical tests are performed 
            for each bin. For example, this might give a binned version of the 
            xVar, or xVar itself, if xVar takes discrete values. Data will
            be returned in assending order of this binning variable, therefore
            it must be numeric. Bins which are adjacent acording to numeric
            ordering of this binning variable are also treated as ajdacent 
            for cluster-based permutation tests.
        obsID: None | str. Only needed if runPerm or checkSame are True. 
            In this case it should be the name of a column giving the 
            independent observation to which each datapoint belongs. 
            In these cases, bin takes on the more specific meaning of 
            identifying the different repeated measurements that make up
            each observeration. E.g. obsID could be a unique identifier for 
            each participant.
        runPerm: bool. If true, run threshold-free cluster based permutation
            tests.
        checkEqual: bool. If true, check that there are the same number
            of cases in each bin. Must be true if performing cluster based
            permutation tests. Must be true if runPerm is True.
        checkSame: bool. If str, should be the name of a column in the
            dataset giving the independent observation to which each 
            measurement belongs. In this case check that the same observations 
            are present in each comparison. Must be true if runPerm is True.
        checkOrder: bool. If true, check that when data is ordered following
            the values of the binning variable ('bin'), than the mean 
            x-value increases across bins.

        OUTPUT
        stats: dataframe. Gives the results of the statistical tests. 
            Index is arbitrary. Has the one row for each bin, and the 
            following columns...
                xVar+'_mean': Mean x-value of data in the bin
                yVar+'_mean': Mean y-value of data in the bin
                yVar+'_SEM': Standard error of the mean y-value
                yVar+'_sig': (Only present if runPerm is True.) bool. True if y-value
                    for this bin is significantly different from zero in 
                    threhsold-free cluster-based permutation test.
        """
        # WORKING HERE -- done but read through
        if runPerm:
            assert checkEqual
            assert checkSame
        if runPerm or checkSame:
            assert obsID

        data = deepcopy(self.data)
        grouped = data.groupby(bin)
        if checkEqual:
            checkGroupsEqual(grouped)
        if checkSame:
            checkSameMeasures(data, obsID, bin)

        xMean, _, xBin = groupedToMeanSd(grouped[xVar])
        yMean, ySem, xBinForY = groupedToMeanSd(grouped[yVar])
        assert np.array_equal(xBin, xBinForY)
        assert np.all(np.diff(xBin) > 0)
        if checkOrder:
            assert np.all(np.diff(xMean) > 0)

        stats = {
            (xVar+'_mean'): xMean,
            (yVar+'_mean'): yMean,
            (yVar+'_SEM'): ySem
        }

        if obsID is not None:
            pVals, xBinForP = runSeriesClstTest(self.data, xBin=bin,
                                                  yVar=yVar, obsID=obsID)
            assert np.array_equal(xBin, xBinForP)

            sig = pVals < 0.05
            assert sig.shape == stats['Y'].shape
            stats[yVar+'_sig'] = sig
            print('{} of {} bins significant.'.format(np.sum(sig), len(sig)))

        stats = pd.DataFrame(stats)
        return stats


    def runMultiStats(self,
                        depVar: str, 
                        avVars: None | str | list[str],
                        bin: str | list[str],
                        checkEqual: bool = True,
                        checkSame: bool = True,
                        fdr: bool = True):
        """ Run multiple statistical tests comparing values to zero, before
        performing FDR correction if requested.

        INPUT
        depVar: str. Name of the columns containing the dependent variable 
            that we want to perform statistics on.
        avVars: None | str | list[str]. Names of any columns containing 
            variables that we would like to average (within the bins defined 
            by bin) but not perform statisitcs on. E.g. could be helpful for 
            determining average values of independent variables.
        bin: str | list[str]. Name or names of columns in the 
            dataset. Statistics are performed seperately for each unique 
            combination of these variables. That is, each unique combination
            of these variables defines a bin of data, which which averaging
            and a statistical comparison is performed. Results of these 
            multiple individual statistical comparsions are corrected using 
            FDR correction if requested.
        checkEqual: bool. If true, check there are an equal number of cases
            in each comparison performed.
        checkSame: None | str. If str, should be the name of a column in the
            dataset giving the independent observation to which each 
            measurement belongs. In this case check that the same observations 
            are present in each comparison.
        fdr: bool. If true, peform False Discovery Rate correction.

        OUTPUT
        stats: dataframe. Gives the results of the statistical tests. 
            Index is arbitrary. Has the one row for each unique combination of
            the bin variables, and the following columns...
                depVar+'_mean': Gives the mean of the dependent variable within
                    the bin.
                depVar+'_sig': Bool. True for for significant points (FDR 
                    corrected if requested).
                Each of the avVars with the suffex '_mean': For each of the 
                    avVars gives the mean within the bin.
                Each of the bin vars: The binning variables.
        """
        # Really wrote very fast -- check carefully! WORKING HERE
        import mne

        assert isinstance(depVar, str)
        if avVars is None:
            allAvVars = [depVar]
        elif isinstance(avVars, str):
            allAvVars = [depVar, avVars]
        elif isinstance(avVars, list):
            allAvVars = [depVar] + avVars
        else:
            raise TypeError('Unsupported input')

        data = deepcopy(self.data)
        grouped = data.groupby(bin)
        if checkEqual:
            checkGroupsEqual(grouped)
        if checkSame:
            checkSameMeasures(data, obsID=checkSame, repMeas=bin)

        stats = data.groupby(bin).aggregate(lambda df: spStats.ttest_1samp(
                                                        df[depVar], 
                                                        popmean=0, 
                                                        axis=None).pvalue)
        assert isinstance(stats, pd.Series)
        stats = stats.rename('pValue').to_frame()
        helpers.checkDfLevels(stats, indexLvs=bin)
        assert list(stats.columns) == ['pValue']
        assert stats.dtypes['pValue'] == 'float'
        
        if fdr:
            stats[depVar+'_sig'], _ = mne.stats.fdr_correction(
                                                            stats['pValue'])
        else:
            stats[depVar+'_sig'] = stats['pValue'] < 0.05
        stats = stats.drop(columns='pValue')

        mean = grouped.mean()
        mean = mean[allAvVars]
        if isinstance(mean, pd.Series):
            mean = mean.to_frame()
        assert list(mean.columns) == allAvVars
        mean = mean.add_suffix('_mean', axis='columns')

        oldLen = len(stats)
        stats = pd.merge(mean, stats,
                            how='inner',
                            on=bin,
                            suffixes=(False, False),
                            validate='one_to_one')
        assert len(stats) == oldLen == len(mean)
        helpers.checkDfLevels(stats, indexLvs=bin)

        stats = stats.reset_index(allow_duplicates=False)
        expect = [thisCol+'_mean' for thisCol in allAvVars] + [depVar+'_sig']
        assert list(stats.columns) == (expect + bin)
        return stats


class Plotter():
    """ Stores and plots the data for one subplot. Only takes care of plotting
    things specific to the one subplot. Details which may be shared across 
    plots such as x- and y-axis labels, legends and colour bars, are not 
    added to the subplot. Instead the details of these features are stored
    as attributes. These details can be used to add the corresponding 
    features to the plots by through methods of the MultiPlotter class.

    ATTRIBUTES
    axisLabels: None | dict. Stores the labels for the x- and y-axes.
    legendSpec: None | dict. Stores a complete specification of the legend 
        associated with the plot.
    cBarSpec: None | dict. Stores a complete specification of the colour bar
        associated with the plot.
    """
    axisLabels = _makeCarefulProperty('axisLabels', ['xLabel', 'yLabel'])
    legendSpec = _makeCarefulProperty('legendSpec', ['label', 'colour'])
    cBarSpec = _makeCarefulProperty('cBarSpec', []) # WORKING HERE

    
    def __init__(self, plotSpec=None) -> None:
        """
        INPUT
        plotSpec: None | dict. Specificiations for the subplot as a whole.
            axisLabels: None | dict. Keys are...
                xLabel: str | None.
                yLabel: str | None. 
        """
        self._axisLabels = None
        self._legendSpec = None
        self._cBarSpec = None

        if plotSpec is not None:
            if plotSpec['axisLabels'] is not None:
                self.axisLabels = plotSpec['axisLabels']


    def plot(self, ax):
        """ Make the subplot.

        INPUT
        ax: Axis to plot onto
        """
        raise NotImplementedError
    

class SeriesPlotter(Plotter):
    """ Stores and plots the data for one or multiple series on a single
    subplot. Only takes care of plotting things specific to the one subplot. 
    Details which may be shared across plots such as x- and y-axis labels, 
    legends and colour bars, are not added to the subplot. Instead the details 
    of these features are stored as attributes. These details can be used to 
    add the corresponding features to the plots by through methods of the 
    MultiPlotter class.

    ATTRIBUTES
    seriesSpec: list of dict. Same as input to the constructor.
    axisLabels: None | dict. Stores the labels for the x- and y-axes.
    vLines: None | dict. Same as input to constructor plotSpec['vLines'].
    hLine: scalar | False. Same as input to constructor plotSpec['hLine'].
    legendSpec: dict. Stores a complete specification of the legend 
        associated with the plot. Keys are...
            'label': list of str. Labels for the entries in the legend.
            'colour': list of str. Colours for the entries in the legend.
    """

    def __init__(self, seriesSpec, plotSpec=None):
        """
        INPUT
        seriesSpec: list of dict. Each element of the list is a dict that 
            specifies a series to plot. Keys are...
                data: dataframe. The data to plot.
                x: str. The column in data giving the x-points to plot.
                y: str. The column in data giving the y-points to plot.
                posError: str. The column in data giving the length of the 
                    error bar from the data point to the top of the error bar.
                    Hence the total length of the error bar will be twice this
                    value.
                sig: str (optional). The column in the data giving boolean 
                    values where true indicates that a data point was 
                    significant.
                label: None | str. The label for the series in the legend.
                    None to exclude from legend.
                colour: str. The colour to plot the series in.
        plotSpec: None | dict. Specificiations for the subplot as a whole.
            axisLabels: None | dict. Keys are...
                xLabel: str | None.
                yLabel: str | None. 
            vLines: None | dict. Keys are...
                vLines: dict. Keys are strings to be used as labels, 
                    and values are scalars giving the x-location in 
                    data coordinates for the vertical line
                addVLabels: bool. Whether to add labels to the vertical 
                    lines. 
            hLine: scalar or False. If not False, then adds a horizonal line 
                at the value given (in data coordinates)
        # TODO -- change plotSpec to input args? And elsewhere
        # TODO -- check that seriesSpec has only permitted keys?
        """
        super().__init__(plotSpec)
        self.seriesSpec = seriesSpec
        self.legendSpec = self.findLegend()

        if plotSpec is not None:
            if plotSpec['vLines'] is not None:
                self.vLines = plotSpec['vLines']

            if plotSpec['hLine'] is not None:
                self.hLine = plotSpec['hLines']


    def findLegend(self):
        """ Using self.seriesSpec find the details of the legend to use.

        OUTPUT
        legendSpec: dict. Keys are...
            'label': list of str. Labels for the entries in the legend.
            'colour': list of str. Colours for the entries in the legend.
        """
        legendSpec = dict()
        for detail in ['label', 'colour']:
            legendSpec[detail] = [
                thisSeries[detail] for thisSeries in self.seriesSpec]

        legendSpec = [(thisL, thisC) for thisL, thisC 
                      in zip(legendSpec['label'], legendSpec['colour'])
                      if thisL is not None]
        legendSpec['label'], legendSpec['colour'] = zip(legendSpec)

        return legendSpec


    def plot(self, ax):
        """ Make the subplot.

        INPUT
        ax: Axis to plot onto
        """
        pltData = []
        sColours = []
        for thisSeries in self.seriesSpec:
            thisData = thisSeries['data']
            newNames = {
                thisSeries['x']: 'X',
                thisSeries['y']: 'Y',
                thisSeries['posError']: 'ERROR',
            }
            if 'sig' in thisSeries:
                newNames[thisSeries['sig']] = 'SIG'

            toKeep = list(newNames.keys())
            thisData = thisData.loc[:, toKeep]
            thisData = thisData.rename(columns=newNames)
            pltData.append(thisData)

            sColours.append(thisSeries['colour'])

        plotLineWithError(pltData, sColours, hLine=self.hLine, ax=ax)
        addVLines(ax, self.vLines['vLines'], self.vLines['addVLabels'])

    
    def addLegend(self, ax):
        """ Add a legend to a specified axis

        INPUT
        ax: The axis to add the legend to
        plotSpec: dict. Contains plot specifications. Produced by the 
            mkPlotSpec() method of plotter object that we are adding a 
            legend for. See the __init__() method for more details.
        """
        legSpec = self.legendSpec
        assert set(legSpec.keys()) == set(['label', 'colour'])

        allLines = []
        allLineSpecs = zip(legSpec['label'], legSpec['colour'])
        for thisLabel, thisColour in allLineSpecs:
            allLines.append(ax.plot([], [], label=thisLabel, color=thisColour))
        ax.legend(frameon=False, loc='upper left')
            

class ColourPlotter(Plotter):
    """ Stores and plots the data for one colour-based plot (e.g. a heatmap) 
    on a single subplot. Only takes care of plotting things specific to the 
    one subplot. Details which may be shared across plots such as x- and y-axis 
    labels, legends and colour bars, are not added to the subplot. Instead the 
    details of these features are stored as attributes. These details can be 
    used to add the corresponding features to the plots by through methods of 
    the MultiPlotter class.

    ATTRIBUTES
    colourData: dict. Same as input to the constructor.
    axisLabels: None | dict. Stores the labels for the x- and y-axes.
    cBarSpec: dict. Stores a complete specification of all the details 
        required to produce the colour bar associated with the plot. 
    cBarCenter: None | scalar. Same as input to constructor.
    """

    def __init__(self, colourData, plotSpec=None, cBarCentre=None):
        """
        INPUT
        colourData: dict. Contains the colour data to plot. Keys are...
                data: dataframe. The data to plot. All provided data will be
                    used to set the range of the colourbar.
                c: str. The column in the data giving the colour values that
                    will be plotted.
        plotSpec: None | dict. Specificiations for the subplot as a whole.
            axisLabels: None | dict. Keys are...
                xLabel: str | None.
                yLabel: str | None. 
        cBarCentre: None | scalar. If not none, ensure the colour bar is 
            centred on this value.
        """
        super().__init__(plotSpec)
        self.colourData = colourData

    
    def findColourRange(self):
        """ Find the smallest colour range that would include all colour 
        data and meet all requirements on the nature of the colour bar.
        """
        # WORKING HERE


def checkGroupsEqual(grouped):
    """ Check there are an equal number of cases in each group, after 
    performing grouping with pandas.

    INPUT
    grouped: pandas groupby instance.
    """
    counts = grouped.count()
    counts = np.unique(counts.to_numpy())
    if len(counts) != 1:
        raise ValueError('Number of cases being averages is not constant.')
    

def checkSameMeasures(data: pd.DataFrame, obsID: str, 
                      repMeas: str | list[str]):
    """ Check that we have the same set of repeared measurements for each 
    observation. An error is raised if an observation has either duplicated
    or missing reapeated measurements.

    INPUT
    data: dataframe. Index will be ignored.
    obsID: None | str. Name of the column in data giving the independent 
        observation to which each datpoint belongs.
    repMeas: str | list[str]. Name or names of columns in the 
        dataset. Each unique combination defines one repeated measurement.
        Code checks that each observeration has one data point for each of
        these measurements.
    """
    # WORKING HERE -- read through 
    data = deepcopy(data)
    data = data.sort_index(axis=0)
    data = data.sort_index(axis=1)

    assert isinstance(obsID, str)
    if isinstance(repMeas, str):
        repMeas = [repMeas]
    assert isinstance(repMeas, list)

    grouped = data.groupby([obsID]+repMeas)
    count = grouped.count()
    if not np.all(np.equal(count.to_numpy(), 1)):
        raise ValueError('An observation has duplicate measurements')
    
    targIdx = data.groupby(repMeas).count().index
    obs = np.unique(data[obsID])
    for thisObs in obs:
        thisCount = count.loc[(thisObs,), :]
        if not targIdx.equals(thisCount.index):
            raise ValueError('An observation is missing measurements')
    

def groupedToMeanSd(grouped, sort=True):
    """ Takes the grouped version (i.e. groupby applied) of a pandas
    dataframe with a single column (or a grouped pandas series) and computes 
    the mean and SEM for each group.

    INPUT
    grouped: grouped pandas dataframe.
    sort: bool. If True, return the results sorted in assending order of the
        grouping variable.

    OUTPUT
    mean: 1D array. Mean for each group of the values of the single column.
    sem: 1D array. SEM for each group of the values of the single column.
    group: 1D array. The value for each group of the grouping variable.
    """
    permitted = isinstance(grouped.mean(), pd.Series) or \
        len(grouped.mean().columns) == 1
    if not permitted:
        raise ValueError('Should be groupby of a series or dataframe with '+
                         'single column')

    meanDf = grouped.mean()
    if sort:
        meanDf = meanDf.sort_index()
    mean = meanDf.to_numpy()
    mean = helpers.removeDimIfNeeded(mean)
    group = meanDf.index.to_numpy()

    semDf = grouped.sem()
    if sort:
        semDf = semDf.sort_index()
    sem = semDf.to_numpy()
    sem = helpers.removeDimIfNeeded(sem)
    group2 = semDf.index.to_numpy()

    assert np.array_equal(group, group2)
    assert np.all([np.ndim(thisOut) == 1 
                  for thisOut in [mean, sem, group]])
    return mean, sem, group


def runSeriesClstTest(data: pd.DataFrame, xBin: str, yVar: str, obsID: str):
    """ Run threshold-free cluster-based permutation testing to compute
    significance. 

    INPUT
    data: dataframe. Index is ignored. Input data is not modified.
    xBin: str. Name of the column identifying the different repeated 
        measurements that make up each observeration (and that define
        the division of data into bins that are plotted at seperate points
        on the x-axis). Bin/measurements which are adjacent 
        acording to numeric ordering of this binning variable are also 
        treated as ajdacent for cluster-based permutation tests.
    yVar: str. Name of column giving the values of the dependent variable.
    obsID: str. Name of the column giving the independent observation to which 
        each datapoint belongs. E.g. obsID could be a unique identifier for 
        each participant.

    OUTPUT
    pVals: 1D numpy array. As long as the number of bins. Gives the p-value 
        associated with that point.
    xBinVals: 1D numpy array. For each value in pVals this variable gives
        the repeated measurement to which it corresponds. These values will
        be in assending order.
    """
    # WORKING HERE -- done but read through
    import mne
    data = deepcopy(data)

    numBins = len(np.unique(data[xBin]))
    numObs = len(np.unique(data[obsID]))

    data = data.loc[:, [xBin, yVar, obsID]]
    data = data.set_index([xBin, obsID], verify_integrity=True)
    data = data.sort_index()
    data = data.unstack(xBin)

    helpers.checkDfLevels(data, indexLvs=[obsID], colLvs=[xBin])
    
    indexVals = data.index.get_level_values(obsID).to_numpy()
    colVals = data.columns.get_level_values(xBin).to_numpy()
    for theseVals, thisExpect in zip([indexVals, colVals], [numObs, numBins]):
        assert len(theseVals) == len(np.unique(theseVals)) == thisExpect
        assert np.all(np.diff(theseVals) > 0)
    if np.any(pd.isnull(data)) or np.any(np.isnan(data.values)):
        raise ValueError('No data may be missing when performing permutation '+
                         'testing on a series.')

    _, _, pVals, _ = mne.stats.permutation_cluster_1samp_test(
                                    data.values,
                                    threshold={'start':0, 'step':0.005},
                                    adjacency=None)
    xBinVals = colVals

    assert pVals.shape == xBinVals.shape == (numBins,)
    return pVals, xBinVals


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
