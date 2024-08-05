""" Tools for plotting, including for plotting several related plots as part 
of larger subplots.

HISTORY
10.07.2024 Read through everything apart from plotLineWithError, findSigEnds,
    plotHeatmapFromDf
"""
from copy import deepcopy
import numpy as np
import pandas as pd
import scipy.stats as spStats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as mplCm
import matplotlib.colors as pltColours
import mne
from . import helpers


def _makeCarefulProperty(name: str, keys: list[str] = None):
    """ Make a property which can only be used in a specific way without 
    an error being raised. Specifically, the property will raise an 
    error if...
        - Called before being set
        - Set again after already being set
        - Called but the keys of the returned dict don't match the 
        input 'keys' (if keys is provided as an input)

    INPUT
    name: str. Name to use for the underying attribute. A '_' will be 
        prepended.
    keys: list[str]. Optional. The keys that expect in the dict returned 
        when calling this property.
    """
    attrName = '_'+name

    @property
    def thisProp(self):
        value = getattr(self, attrName)
        if value is None:
            raise ValueError(f'The {name} attribute has not yet been set.')
        if (keys is not None) and (set(value.keys()) != set(keys)):
            raise ValueError(f'The {name} attribute does not have the '+
                                'expected keys.')
        return value
    
    @thisProp.setter
    def thisProp(self, value):
        oldValue = getattr(self, attrName)
        if oldValue is not None:
            raise ValueError(f'The {name} attribute cannot be set twice.')
        if (keys is not None) and (set(value.keys()) != set(keys)):
            raise ValueError(f'The {name} attribute does not have the '+
                                'expected keys.')
        setattr(self, attrName, value)

    return thisProp


class Formatter():
    """ Stores and manipulates a dataset. Helpful for getting a dataset into
    the format required for making a specific plot.

    ATTRIBUTES
    data: pandas dataframe storing the current version of the dataset. Index
        is arbitrary.
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
            raise ValueError('Provide both \'keep\' and \'drop\' or neither.')

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


    def dBin(self, strategy: str, 
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
        """ Perform binning of all input data. Similar functionality to .dBin()
        except operates on the input, not on self.data, and cannot perform 
        binning seperately for different subsets of the data.

        INPUT
        data: pandas dataframe to perform the binning on. Data is deepcopied
            to ensure no mutation of the input data.
        strategy, opts, binName: Same as input to the method .dBin()

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
            data[binName] = pd.cut(data[opts['col']],
                                    bins=opts['edges'],
                                    labels=False)
            assert not np.any(np.isnan(data[binName]))
        else:
            raise ValueError('Unrecognised binning option')
        
        return data
            

    def runSeriesStats(self, xVar: str, yVar: str, dBin: str, 
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
        dBin: str. Name of column giving the bin to which each datapoint 
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
            In these cases, dBin takes on the more specific meaning of 
            identifying the different repeated measurements that make up
            each observeration. E.g. obsID could be a unique identifier for 
            each participant.
        runPerm: bool. If true, run threshold-free cluster based permutation
            tests.
        checkEqual: bool. If true, check that there are the same number
            of cases in each dBin. Must be true if runPerm is True.
        checkSame: bool. Check that the same observations (identified through
            obsID) are present in each comparison. Must be true if runPerm is 
            True.
        checkOrder: bool. If true, check that when data is ordered following
            the values of the binning variable ('dBin'), that the mean 
            x-value increases across bins.

        OUTPUT
        stats: dataframe. Gives the results of the statistical tests. 
            Index is arbitrary. Has the one row for each dBin, and the 
            following columns...
                xVar+'_mean': Mean x-value of data in the dBin
                yVar+'_mean': Mean y-value of data in the dBin
                yVar+'_SEM': Standard error of the mean y-value
                yVar+'_sig': (Only present if runPerm is True.) bool. True if 
                    y-value for this dBin is significantly different from zero 
                    in threhsold-free cluster-based permutation test.
        """
        if runPerm:
            assert checkEqual
            assert checkSame
        if runPerm or checkSame:
            assert obsID

        data = deepcopy(self.data)
        grouped = data.groupby(dBin)
        if checkEqual:
            checkGroupsEqual(grouped)
        if checkSame:
            checkSameMeasures(data, obsID, dBin)

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
            pVals, xBinForP = runSeriesClstTest(self.data, xBin=dBin,
                                                  yVar=yVar, obsID=obsID)
            assert np.array_equal(xBin, xBinForP)

            sig = pVals < 0.05
            assert sig.shape == stats[yVar+'_mean'].shape
            stats[yVar+'_sig'] = sig
            print(f'{np.sum(sig)} of {len(sig)} bins significant.')

        stats = pd.DataFrame(stats)
        return stats


    def runMultiStats(self,
                        depVar: str, 
                        dBin: str | list[str],
                        avVars: None | str | list[str] = None,
                        checkEqual: bool = True,
                        checkSame: str = None,
                        fdr: bool = True):
        """ Run multiple statistical tests comparing values to zero, before
        performing FDR correction if requested.

        INPUT
        depVar: str. Name of the columns containing the dependent variable 
            that we want to perform statistics on.
        avVars: None | str | list[str]. Names of any columns containing 
            variables that we would like to average (within the bins defined 
            by dBin) but not perform statisitcs on. E.g. could be helpful for 
            determining average values of independent variables.
        dBin: str | list[str]. Name or names of columns in the 
            dataset. Statistics are performed seperately for each unique 
            combination of these variables. That is, each unique combination
            of these variables defines a bin of data, within which averaging
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
                    the dBin.
                depVar+'_sig' or depVar+'_fdr_sig': Bool. True for for 
                    significant points (FDR corrected if requested).
                Each of the avVars with the suffex '_mean': For each of the 
                    avVars gives the mean within the dBin.
                Each of the dBin vars: The binning variables.
        """
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
        grouped = data.groupby(dBin)
        if checkEqual:
            checkGroupsEqual(grouped)
        if checkSame:
            checkSameMeasures(data, obsID=checkSame, repMeas=dBin)

        def tTestCol(df):
            assert np.ndim(df) == 1
            return spStats.ttest_1samp(df, popmean=0, axis=None).pvalue

        stats = data.groupby(dBin).agg(pValue=pd.NamedAgg(
            column=depVar, aggfunc=tTestCol))
        helpers.checkDfLevels(stats, indexLvs=[dBin])
        assert list(stats.columns) == ['pValue']
        assert stats.dtypes['pValue'] == 'float'
        
        if fdr:
            stats[depVar+'_fdr_sig'], _ = mne.stats.fdr_correction(
                                                            stats['pValue'])
        else:
            stats[depVar+'_sig'] = stats['pValue'] < 0.05
        stats = stats.drop(columns='pValue')

        mean = grouped[allAvVars].mean()
        if isinstance(mean, pd.Series):
            mean = mean.to_frame()
        assert list(mean.columns) == allAvVars
        mean = mean.add_suffix('_mean', axis='columns')

        assert len(stats) == len(mean)
        oldLen = len(stats)
        stats = pd.merge(mean, stats,
                            how='inner',
                            on=dBin,
                            suffixes=(False, False),
                            validate='one_to_one')
        assert len(stats) == len(mean) == oldLen
        helpers.checkDfLevels(stats, indexLvs=[dBin])

        stats = stats.reset_index(allow_duplicates=False)
        if fdr:
            sigCol = depVar + '_fdr_sig'
        else:
            sigCol = depVar + '_sig'
        if isinstance(dBin, str):
            dBin = [dBin]
        expect = [thisCol+'_mean' for thisCol in allAvVars] + [sigCol] + dBin
        assert set(stats.columns) == set(expect)
        assert len(np.unique(stats.columns)) == len(stats.columns)
        return stats


class Plotter():
    """ Stores and plots the data for one subplot.

    ATTRIBUTES
    axisLabels: None | dict. Stores the labels for the x- and y-axes.
    title: None | dict. Stores a complete specification of the plot title.
    ax: None | axis. Once plotting is performed, the axis used is stored here.
    """
    axisLabels = _makeCarefulProperty('axisLabels', ['xLabel', 'yLabel'])
    title = _makeCarefulProperty('title', ['txt', 'rotation', 'weight'])
    
    def __init__(self, xLabel=None, yLabel=None, 
                 titleTxt=None, titleRot=0, titleWeight='normal') -> None:
        """
        INPUT
        xLabel: str | None. Axis label.
        yLabel: str | None. Axis label.
        titleTxt: str | None. Text for title.
        titleRot: scalar. Rotation of title text.
        titleWeight: str. Specification of font weight.
        """
        self._axisLabels = None
        self._title = None
        self.axisLabels = {'xLabel': xLabel, 'yLabel': yLabel}
        self.title = {'txt': titleTxt, 'rotation': titleRot, 
                      'weight': titleWeight}
        self.ax = None


    def plot(self, ax):
        """ Make the subplot.

        INPUT
        ax: Axis to plot onto
        """
        raise NotImplementedError
    

    def addTitle(self, ax):
        """ Add a title to a specified axis.

        INPUT
        ax: The axis to add the title to 
        """
        ax.set_title(self.title['txt'],
                     rotation=self.title['rotation'],
                     fontweight=self.title['weight'])
    

class SeriesPlotter(Plotter):
    """ Stores and plots the data for one or multiple series on a single
    subplot.

    ATTRIBUTES
    ax: None | axis. Once plotting is performed, the axis used is stored here.
    seriesData: list of dataframe. Same as input to the 
        constructor but always a list.
    sColours: list of str. Same as input to the constructor.
    sLabels: list of str. Same as input to the constructor.
    axisLabels: None | dict. Stores the labels for the x- and y-axes.
    title: None | dict. Stores a complete specification of the plot title.
    vLines: None | dict. Same as input to constructor.
    hLine: scalar | False. Same as input to constructor.
    legendSpec: dict. Stores a complete specification of the legend 
        associated with the plot. Keys are...
            'label': list of str. Labels for the entries in the legend.
            'colour': list of str. Colours for the entries in the legend.
    """
    # Make sure the colours, labels and legend can only be set once, so that
    # they don't accidentaly come out of sync with each other (or with
    # the legends of other plots, if plotting multiple subplots)
    sColours = _makeCarefulProperty('sColours')
    sLabels = _makeCarefulProperty('sLabels')
    legendSpec = _makeCarefulProperty('legendSpec', ['label', 'colour'])

    def __init__(self, seriesData, sColours, sLabels, 
                 xLabel=None, yLabel=None,
                 vLines=None, hLine=None,
                 titleTxt=None, titleRot=0, titleWeight='normal'):
        """
        INPUT
        seriesData: dataframe | list of dataframe. Each element of the list is 
            a dataframe that specifies a series to plot. One dataframe to plot
            only a single series. Index of the dataframe is ignored. Columns 
            should be...
                X: str. The column giving the x-points to plot.
                Y: str. The column giving the y-points to plot.
                posErr: str. The column giving the length of the 
                    error bar from the data point to the top of the error bar.
                    Hence the total length of the error bar will be twice this
                    value.
                sig: str (optional). The column giving boolean values where 
                    true indicates that a data point was significant.
        sColours: list of str. List as long as seriesData, specifying the 
            colour to use for each series.
        sLabels: list of str. List as long as seriesData, specifying the label
            for each series in the legend
        xLabel: str | None. Axis label.
        yLabel: str | None. Axis label. 
        vLines: None | dict. Keys are...
            vLines: dict. Keys are strings to be used as labels, 
                and values are scalars giving the x-location in 
                data coordinates for the vertical line
            addVLabels: bool. Whether to add labels to the vertical 
                lines. 
        hLine: scalar or False. If not False, then adds a horizonal line 
            at the value given (in data coordinates)
        titleTxt: str | None. Text for title.
        titleRot: scalar. Rotation of title text.
        titleWeight: str. Specification of font weight.
        """
        self._sColours = None
        self._sLabels = None
        self._legendSpec = None
        super().__init__(xLabel, yLabel, titleTxt, titleRot, titleWeight)

        if isinstance(seriesData, pd.DataFrame):
            seriesData = [seriesData]
        assert isinstance(seriesData, list)
        self.seriesData = seriesData
        
        self.sColours = sColours
        self.sLabels = sLabels
        self.legendSpec = self.findLegend()
        self.vLines = vLines
        self.hLine = hLine


    def findLegend(self):
        """ Find the details of the legend to use.

        OUTPUT
        legendSpec: dict. Keys are...
            'label': list of str. Labels for the entries in the legend.
            'colour': list of str. Colours for the entries in the legend.
        """
        # This function isn't doing much at the moment but will become 
        # helpful when want to impliment the option of supressing particular
        # series from the legend.
        legendSpec = dict()
        legendSpec['label'] = self.sLabels
        legendSpec['colour'] = self.sColours

        return legendSpec


    def plot(self, ax):
        """ Make the subplot.

        INPUT
        ax: Axis to plot onto
        """
        for thisSeries in self.seriesData:
            assert np.all(np.isin(list(thisSeries.columns), 
                                  ['X', 'Y', 'posErr', 'sig']))

        plotLineWithError(self.seriesData, self.sColours, 
                            hLine=self.hLine, ax=ax,
                            xLabel=self.axisLabels['xLabel'],
                            yLabel=self.axisLabels['yLabel'])
        addVLines(ax, self.vLines['vLines'], self.vLines['addVLabels'])
        self.addLegend(ax)
        self.addTitle(ax)
        self.ax = ax

    
    def addLegend(self, ax):
        """ Add a legend to a specified axis

        INPUT
        ax: The axis to add the legend to
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
    on a single subplot.

    ATTRIBUTES
    cBarSpec: None | dict. Stores a final and complete specification of all the 
        details required to produce the colour bar associated with the plot. 
    axisLabels: None | dict. Stores the labels for the x- and y-axes.
    title: None | dict. Stores a complete specification of the plot title.
    ax: None | axis. Once plotting is performed, the axis used is stored here.
    colourData: dict. Same as input to the constructor.
    draftCBarSpec: None | dict. Stores a draft of the cBarSpec based on user
        requested settings. Should not be used for plotting, but rather a 
        finalised cBarSpec should be created using finaliseCBarSpec. Keys 
        are...
            cLabel: str | None. Label for the colour bar.
            cBarCenter: None | scalar. If not none, colour bar is to be 
                centred on this value.
    cBar: None or handle of the colourbar associated with the plot.
    """
    cBarSpec = _makeCarefulProperty('cBarSpec', ['cMap', 'cNorm', 'cMin', 
                                                 'cMax', 'cLabel'])

    def __init__(self, colourData, xLabel=None, yLabel=None, 
                 cLabel=None, cBarCentre=None,
                 titleTxt=None, titleRot=0, titleWeight='normal'):
        """
        INPUT
        colourData: dataframe. Contains the data to convert to colours and
            plot on the heatmap. All provided data will be plotted. Index is 
            arbitrary. Has the following columns:
                C: The values to convert to colours and plot. All provided data 
                    will be used to set the range of the colourbar.
        xLabel: str | None. Axis label.
        yLabel: str | None. Axis label.
        cLabel: str | None. Label for the colourbar.
        cBarCentre: None | scalar. If not none, ensure the colour bar is 
            centred on this value.
        titleTxt: str | None. Text for title.
        titleRot: scalar. Rotation of title text.
        titleWeight: str. Specification of font weight.
        """
        self._cBarSpec = None
        super().__init__(xLabel, yLabel, titleTxt, titleRot, titleWeight)
        self.colourData = colourData
        self.draftCBarSpec = {
            'cLabel': cLabel,
            'cBarCentre': cBarCentre
        }
        self.cBar = None


    def plot(self, ax):
        """ Make the subplot.

        INPUT
        ax: Axis to plot onto
        """
        raise NotImplementedError

    
    def findColourRange(self):
        """ Find the smallest colour range that would include all colour 
        data and meet all requirements on the nature of the colour bar.
        Defaults to -0.001 to 0.001 if there is no data to plot.

        OUTPUT
        cMin: scalar. Bottom of the smallest possible colour bar range.
        cMax: scalar. Top of the smallest possible colour bar range.
        cCenter: None | scalar. If not None, then gives the value that has
            been requested to be at the centre of the colour scale.
        """
        vals = self.colourData['C']

        if len(vals) > 1:
            cMin = np.min(vals)
            cMax = np.max(vals)
        
        elif len(vals) == 1:
            vals = vals.iloc[0]
            assert np.ndim(vals) == 0
            cMax = np.abs(vals)
            cMin = -cMax
        else:
            cMin = -0.001
            cMax = 0.001

        for lim in [cMin, cMax]:
            assert not np.isnan(lim)

        cBarCentre = self.draftCBarSpec['cBarCentre']
        if cBarCentre is not None:
            cMin, cMax = findCenteredScale(cMin, cMax, cBarCentre)
        return cMin, cMax, cBarCentre
    

    def finaliseCBarSpec(self, cMin, cMax):
        """ Finalise the cBarSpec attribute, using the draft version and the
        input arguments. Check that the specification is consistent with the 
        requested properties.

        INPUT
        cMin, cMax: scalar. The bottom and top of the finalised colour bar
            scale.
        """
        cBarSpec = dict()
        cBarSpec['cMap'] = 'RdBu_r'
        cBarSpec['cNorm'] = pltColours.Normalize
        cBarSpec['cMin'] = cMin
        cBarSpec['cMax'] = cMax
        cBarSpec['cLabel'] = self.draftCBarSpec['cLabel']

        self.checkCBarSpec(cBarSpec=cBarSpec, colourData=self.colourData['C'])
        self.cBarSpec = cBarSpec


    def checkCBarSpec(self, cBarSpec=None, colourData=None):
        """ Run a number of checks on the cBarSpec.

        INPUT
        cBarSpec: None | dict. The cBarSpec to check. If None, checks
            self.cBarSpec
        colourData: None | dataframe | pandas series. If provided, it is 
            checked that all the values in the dataframe are within the 
            colorbar range.
        """
        if cBarSpec is None:
            cBarSpec = self.cBarSpec

        assert set(cBarSpec.keys()) == set(['cMap', 'cNorm', 'cMin', 'cMax',
                                            'cLabel'])

        if self.draftCBarSpec['cBarCentre'] is not None:
            assert self.draftCBarSpec['cBarCentre'] == ((cBarSpec['cMax'] + 
                                                        cBarSpec['cMin']) /2)
            
        if (colourData is not None) and (len(colourData) > 0):
            assert np.min(colourData.to_numpy()) >= cBarSpec['cMin']
            assert np.max(colourData.to_numpy()) <= cBarSpec['cMax']


    def addColourBar(self, ax):
        """ Plot a colourbar

        INPUT
        ax: The axis to use for the colourbar
        """
        cBarSpec = self.cBarSpec
        assert set(cBarSpec.keys()) == set(['cMap', 'cNorm', 'cMin', 'cMax',
                                            'cLabel'])

        scalarMappable, _ = self.makeColourMapper()
        cbar = plt.colorbar(scalarMappable, cax=ax)
        cbar.outline.set_visible(False)
        cbar.set_label(cBarSpec['cLabel'])

        assert self.cBar is None, 'About to overwrite existing colourbar'
        self.cBar = cbar


    def removeColourBar(self):
        """ Remove the colourbar associated with the plot
        """
        assert self.cBar is not None
        self.cBar.remove()
        self.cBar = None


    def addColourBarOverPlot(self, ax):
        """ Create a new axis directly on top of the passed axis, and create
        a colourbar in this new axis.

        INPUT
        ax: The axis over which to create the colourbar
        """
        fig = ax.get_figure()
        cax = fig.add_axes(ax.get_position())
        self.addColourBar(cax)

    
    def makeColourMapper(self):
        """ Returns the matplotlib ScalarMappable that is to be used for 
        mapping scalars to colours.

        OUTPUT
        scalarMappable: matplotlib ScalarMappable instance.
        cBarNorm: The matplotlib normaliser instance used to create the 
            scalar mappable. E.g. an initalised instance of 
            pltColours.Normalize.
        """
        cBarSpec = self.cBarSpec
        assert set(cBarSpec.keys()) == set(['cMap', 'cNorm', 'cMin', 'cMax',
                                            'cLabel'])
        Normaliser = cBarSpec['cNorm']
        cBarNorm = Normaliser(vmin=cBarSpec['cMin'], vmax=cBarSpec['cMax'])
        scalarMappable = mplCm.ScalarMappable(norm=cBarNorm, 
                                                cmap=cBarSpec['cMap'])
        return scalarMappable, cBarNorm
    

class HeatmapPlotter(ColourPlotter):
    """ Stores and plots the data for one heatmap in a single subplot.

    ATTRIBUTES
    cBarSpec: None | dict. Stores a final and complete specification of all the 
        details required to produce the colour bar associated with the plot. 
    axisLabels: None | dict. Stores the labels for the x- and y-axes.
    title: None | dict. Stores a complete specification of the plot title.
    ax: None | axis. Once plotting is performed, the axis used is stored here.
    colourData: dict. Same as input to the constructor.
    draftCBarSpec: None | dict. Stores a draft of the cBarSpec based on user
        requested plotting. Should not be used for plotting, but rather a 
        finalised cBarSpec should be created using finaliseCBarSpec. Keys 
        are...
            cLabel: str | None. Label for the colour bar.
            cBarCenter: None | scalar. If not none, colour bar is to be 
                centred on this value.
    cBar: None or handle of the colourbar associated with the plot, if a 
        colourbar has been plotted.
    """

    def __init__(self, colourData, xLabel=None, yLabel=None, 
                    cLabel=None, cBarCentre=None,
                    titleTxt=None, titleRot=0, titleWeight='normal'):
        """
        INPUT
        colourData: dataframe. Contains the data to convert to colours and
            plot on the heatmap. All provided data will be plotted. Index is 
            arbitrary. Has the following columns:
                C: The values to convert to colours and plot. All provided data 
                    will be used to set the range of the colourbar.
                X: The x-position associated with each colour value
                Y: The y-position associated with each colour value
        xLabel: str | None. Axis label.
        yLabel: str | None. Axis label.
        cLabel: str | None. Label for the colourbar.
        cBarCentre: None | scalar. If not none, ensure the colour bar is 
            centred on this value.
        titleTxt: str | None. Text for title.
        titleRot: scalar. Rotation of title text.
        titleWeight: str. Specification of font weight.
        """
        super().__init__(colourData, xLabel, yLabel, cLabel, cBarCentre,
                         titleTxt, titleRot, titleWeight)

    def plot(self, ax):
        """ Make the subplot.

        INPUT
        ax: Axis to plot onto
        """
        data = deepcopy(self.colourData)
        data = data.pivot(columns='X', index='Y', values='C')
        for axis in [0, 1]:
            data = data.sort_index(axis=axis)
        assert not np.any(np.isnan(data.to_numpy()))
        helpers.checkDfLevels(data, indexLvs=['Y'], colLvs=['X'])

        self.checkCBarSpec(colourData=data)
        cBarSpec = dict()
        cBarSpec['addCBar'] = False
        cBarSpec['cMap'] = self.cBarSpec['cMap']
        _, cBarSpec['cNorm'] = self.makeColourMapper()

        plotHeatmapFromDf(data, unevenAllowed=True, plotFun='pcolormesh',
                            ax=ax, cbarMode='predef', cbarSpec=cBarSpec,
                            xLabel=self.axisLabels['xLabel'],
                            yLabel=self.axisLabels['yLabel'])
        self.addColourBarOverPlot(ax)
        self.addTitle(ax)
        self.ax = ax
        

class BrainPlotter(ColourPlotter):
    """ Stores and plots the data for one colour-based brain plot in a single 
    subplot.

    ATTRIBUTES
    cBarSpec: None | dict. Stores a final and complete specification of all the 
        details required to produce the colour bar associated with the plot. 
    axisLabels: None | dict. Stores the labels for the x- and y-axes.
    title: None | dict. Stores a complete specification of the plot title.
    ax: None | axis. Once plotting is performed, the axis used is stored here.
    colourData: dict. Same as input to the constructor.
    draftCBarSpec: None | dict. Stores a draft of the cBarSpec based on user
        requested plotting. Should not be used for plotting, but rather a 
        finalised cBarSpec should be created using finaliseCBarSpec. Keys 
        are...
            cLabel: str | None. Label for the colour bar.
            cBarCenter: None | scalar. If not none, colour bar is to be 
                centred on this value.
    cBar: None or handle of the colourbar associated with the plot, if a 
        colourbar has been plotted.
    fsDir: str. Same as input to constructor.
    azimuth: scalar. Same as input to constructor.
    elevation: scalar. Same as input to constructor.
    brainLabels: list of MNE label objects. Gives the label objects associated 
        with the loaded parceltation.
    """
    
    def __init__(self, colourData, fsDir, parc, xLabel=None, yLabel=None, 
                 cLabel=None, cBarCentre=None, 
                 titleTxt=None, titleRot=0, titleWeight='normal',
                 azimuth=0, elevation=0):
        """
        INPUT
        colourData: dataframe. Contains the data to convert to colours and
            plot on the heatmap. All provided data will be plotted. Index is 
            arbitrary. Has the following columns:
                Parc: The string names of the brain parcels to be coloured in.
                    Each must match the name of a left-hemisphiere label in
                    the parcelation specified by the input parc.    
                C: The values to convert to colours and plot. All provided data 
                    will be used to set the range of the colourbar, and will
                    later be plotted.
        fsDir: str. The freesurfer subjects directory. Will load and use the
            'fsaverage' brain.
        parc: str. The name of the labeled parceltation associated with the
            'fsaverage' brain to load.
        xLabel: str | None. Axis label.
        yLabel: str | None. Axis label.
        cLabel: str | None. Label for the colourbar.
        cBarCentre: None | scalar. If not none, ensure the colour bar is 
            centred on this value.
        titleTxt: str | None. Text for title.
        titleRot: scalar. Rotation of title text.
        titleWeight: str. Specification of font weight.
        azimuth: scalar. Angle for displaying the brain.
        elevation: scalar. Angle for displaying the brain.
        """
        super().__init__(colourData, xLabel, yLabel, 
                         cLabel, cBarCentre,
                         titleTxt, titleRot, titleWeight)
        self.fsDir = fsDir
        self.azimuth = azimuth
        self.elevation = elevation
        self.brainLabels = mne.read_labels_from_annot('fsaverage', parc,
                                                        subjects_dir=fsDir,
                                                        verbose='warning')
        

    def plot(self, ax):
        """ Make the subplot.

        INPUT
        ax: Axis to plot onto
        """
        Brain = mne.viz.get_brain_class()
        brain = Brain(
            'fsaverage',
            hemi='lh',
            surf='inflated',
            subjects_dir=self.fsDir,
            background='white',
        )
        mapper, _ = self.makeColourMapper()
        self.checkCBarSpec(colourData=self.colourData['C'])

        parcels = self.colourData['Parc']
        colours = self.colourData['C']

        for thisParc, thisCVal in zip(parcels, colours):
            isLeft = thisParc.startswith('L_') and thisParc.endswith('_ROI-lh') 
            if not isLeft:
                raise NotImplementedError('Code currently only plots '+
                                          'the left hemisphere')
            
            thisLabel = [label for label in self.brainLabels 
                            if label.name == thisParc]
            assert len(thisLabel) == 1
            thisLabel = thisLabel[0]

            thisColour = mapper.to_rgba(thisCVal)
            assert np.shape(thisColour) == (4,)
            brain.add_label(thisLabel, color=thisColour[:-1])
        
        brain.show_view(azimuth=self.azimuth, 
                        elevation=self.elevation,
                        distance=450)
        img = brain.screenshot()
        brain.close()

        ax.imshow(img)

        if self.axisLabels['xLabel']:
            ax.set_xlabel(self.axisLabels['xLabel'])
        if self.axisLabels['yLabel']:
            ax.set_ylabel(self.axisLabels['yLabel'])

        ax.set_frame_on(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        
        self.addColourBarOverPlot(ax) 
        self.addTitle(ax) 
        self.ax = ax


class MultiPlotter():
    """ Plots and coordinates the plottng of multiple subplots, including 
    shared axes, legends, labels and colourbars.

    ATTRIBUTES
    fig: The figure that we are plotting to
    grid: The GridSpec instance for plotting onto
    plots: list of dict: The list of all the plots to make. Each dict has the 
        following keys...
            plotter: An instance of a subclass of Plotter.
            row, col: scalar. Indicies of the first row and columns in the 
                underlying grid that the subplot should occupy
            endRow, endCol: scalar. One greater (consistent with normal
                python indexing) than the index of the final row and columns 
                in the underlying grid that the subplot should occupy
            tags: list. List of string or scalar tags that can be used to 
                refer to the plot
    shared: list of dict. Details all the shared properties. Each element
        specifies one "share". The dict element have keys...
            'property': str. Specifies which property to share. One of 'title',
                'xLabel', 'yLabel', 'xAxis', 'yAxis', 'legend', 'colourBar'
            'tags': list. List of string or scalar tags. Specifies the plots 
                that the "share" should apply to. All plots with any of the 
                tags in the list are included.
            'axis': None | axis instance. Only present if 'property' is 
                'xAxis' or 'yAxis'. In this case, the first time any item
                from the group of plots is plotted, its axis is stored here.
                It can then be refered to, to set up the requested axis 
                sharing.
            pos: None | str | dict. Not None only in the following cases...
                'property' is 'legend' or 'colourBar', then pos is a dict with 
                    keys 'row' and 'col' specifying the position of the legend 
                    or colour bar in the usual format (see comments for 
                    addPlot)
                'property' is 'title', 'xLabel', 'yLabel', 'xAxis', 'yAxis' and
                    want to override the default position of the shared label /
                    tick labels. Then pos is either (a) the string 'first' or
                    'last', determining whether to put the labels on the first
                    or last plot (as ordered in self.plots), or (b) a dict with 
                    keys 'row', 'col', 'endRow', 'endCol' to specify the 
                    position of the label in the usual format 
                    (see comments for addPlot)
    checkAllShare: bool. Same as input to the constructor.
    """

    def __init__(self, gridType: str, gridInfo: dict, 
                 checkAllShare: bool = True,
                 sizes: None | dict = None) -> None:
        """ Define the grid onto which subplots will later be placed.

        INPUT
        gridType: str. The type of grid to produce. Options are...
            'rightSpace': A regular grid with an extra column at the right
                hand side for colour bars or legends.
            'rightSpacePaired': Like right space but the columns are in 
                triplets with two larger and then one smaller. This can be used
                to generate plots with paired columns that are closer to 
                each other.
        gridInfo: dict. Detailed specification of the grid. Keys required
            depend on the gridType. For gridType='rightSpace' require...
                plotRows: The number of subplot rows
                plotCols: The number of subplot cols (not counting the column
                    for the legends and/or colour bars, or the columns
                    for creating spaces between pairs when using 
                    gridType='rightSpacePaired')
        checkAllShare: bool. If True, check at plotting time, that all plots
            are refered to in a specified legend or colour bar share.
        sizes: None | dict. Optionally provide a dict with any subset of the 
            following keys to override default sizing of the plots.
                subplotHeight: Height of each subplot
                headerHeight: Height of space above the top plot
                footerHeight: Height of space below the bottom plot
                subplotWidth: Width of each subplot
                extraColWidth: Width of the extra column in the 'rightSpace'
                    and 'rightSpacePaired' gridTypes
                leftEdge: Width of the space left of the leftmost plot
                rightEdge: Width of the space right of the rightmost plot
                gapWidth: Width of the smaller columns in the rightSpacePaired
                    grid type
        """
        self.checkAllShare = checkAllShare
        self.plots = []
        self.shared = []

        if sizes is None:
            sizes = dict()
        else:
            permittedSizes = ['subplotHeight', 'headerHeight', 'footerHeight',
                              'subplotWidth', 'extraColWidth',
                              'leftEdge', 'rightEdge',
                              'gapWidth']
            assert np.all(np.isin(list(sizes.keys()), permittedSizes))

        if gridType in ['rightSpace', 'rightSpacePaired']:
            assert set(gridInfo.keys()) == set(['plotRows', 'plotCols'])

            setIfMissing(sizes, 'subplotHeight', 1.5)
            setIfMissing(sizes, 'headerHeight', 1.1)
            setIfMissing(sizes, 'footerHeight', 0.3)

            setIfMissing(sizes, 'subplotWidth', 1)
            setIfMissing(sizes, 'extraColWidth', 0.1)
            setIfMissing(sizes, 'leftEdge', 1.5)
            setIfMissing(sizes, 'rightEdge', 0.9)

            if gridType == 'rightSpacePaired':
                numPairs = np.around(gridInfo['plotCols']/2, 10)
                if numPairs != int(numPairs):
                    raise ValueError('For paired columns there must be'
                                     'an even number of columns.')
                numPairs = int(numPairs)
                numGaps = numPairs
                setIfMissing(sizes, 'gapWidth', 0.5)
                gridKwargs = {'wspace': -0.1}
            else:
                assert gridType == 'rightSpace'
                numGaps = 0
                setIfMissing(sizes, 'gapWidth', 0)
                gridKwargs = {}

            figHeight = sizes['headerHeight'] + sizes['footerHeight'] + (
                sizes['subplotHeight']*gridInfo['plotRows'])
            figWidth = sizes['leftEdge'] + sizes['rightEdge'] + \
                sizes['extraColWidth'] + \
                (sizes['subplotWidth'] * gridInfo['plotCols']) + \
                (sizes['gapWidth'] * numGaps)

            leftFrac = sizes['leftEdge']/figWidth
            rightFrac = 1 - (sizes['rightEdge']/figWidth)
            topFrac = 1 - (sizes['headerHeight']/figHeight)
            bottomFrac = sizes['footerHeight']/figHeight

            self.fig = plt.figure(figsize=[figWidth, figHeight])

            if gridType == 'rightSpacePaired':
                weights = [sizes['subplotWidth'], sizes['subplotWidth'], 
                           sizes['gapWidth']] * numPairs
                weights.append(sizes['extraColWidth']) 
                # Extra column for cbar / legend
            else:
                assert gridType == 'rightSpace'
                weights = [sizes['subplotWidth']]*(gridInfo['plotCols']+1) 
                # Extra column for colourbar / legend
                weights[-1] = sizes['extraColWidth']

            self.grid = gridspec.GridSpec(gridInfo['plotRows'], 
                                            gridInfo['plotCols'] + numGaps + 1,
                                            left=leftFrac, bottom=bottomFrac, 
                                            right=rightFrac, top=topFrac,
                                            width_ratios=weights,
                                            **gridKwargs) 
        else:
            raise ValueError('Unrecognised option')


    def addPlot(self, plotter, row, col, endRow=None, endCol=None, tags=None):
        """ Store all the information for a subplot. Call prior to calling the 
        perform method, which performs the requested plotting.

        INPUT
        plotter: An instance of a subclass of Plotter. MultiPlotter takes care
            of deleting existing labels, colour bars, and legends that become
            shared. Therefore it is always safest if the Plotter class
            first plot its own labels, colour bars, and legends. This makes
            it immediately visible when something hasn't been shared that 
            should have been shared.
        row, col: scalar. Indicies of the first row and columns in the 
            underlying grid that the subplot should occupy
        endRow, endCol: scalar. Only required for subplots that span more than 
            one row/column of the underlying grid. One greater (consistent with 
            normal python indexing) than the index of the final row and columns 
            in the underlying grid that the subplot should occupy
        tags: None | list. List of string or scalar tags. This plot can latter
            be refered to using any of its tags. If None is passed, an empty
            list will be stored for tags.
        """
        if endRow is None:
            endRow = row +1
        if endCol is None:
            endCol = col +1
        if tags is None:
            tags = []
        assert isinstance(tags, list)

        plotSpec = {
            'plotter': plotter,
            'row': row,
            'col': col,
            'endRow': endRow,
            'endCol': endCol,
            'tags': tags
        }
        self.plots.append(plotSpec) 


    def addShare(self, prop: str | list, tags: list, pos: dict = None):
        """ Store the information on one property that should be shared 
        amongst some of the subplots. Call prior to calling the perform method, 
        which performs the requested plotting.

        INPUT
        prop: str | list[str]. The name of the property or properties to share. 
            Options are...
            'title': As default the first plot (as ordered in self.plots) will 
                be the plot that retains its title, unless pass pos
            'xLabel': As default the final plot (as ordered in self.plots) will 
                be the plot that retains its x-labeling, unless pass pos
            'yLabel': As default the first plot (as ordered in self.plots) will 
                be the plot that retains its y-labeling, unless pass pos
            'xAxis': Tick-labels will also be removed, apart from (as default) 
                on the final plot (as ordered in self.plots)
            'yAxis': Tick-labels will also be removed, apart from on (as 
                default) the first plot (as ordered in self.plots)
            'legend': Only Plotters that are also instances of SeriesPlotter 
                can share legends
            'colourBar': Only Plotters that are instances of subclasses of 
                ColourPlotter can share colour bars
        tags: list of string or scalar. Specifies the plots that the "share" 
            should apply to. All plots with any of the tags in the list are
            included.
        pos: None | str | dict. Only provide in the following cases...
            Prop is 'legend' or 'colourBar', then pos a dict with keys 
                'row' and 'col' specifying the position of the legend or colour 
                bar in the usual format (see comments for addPlot)
            Prop is 'title', 'xLabel' or 'yLabel' and want to override the 
                default position of the shared label. In this case provide a 
                dict with keys 'row', 'col', 'endRow', 'endCol' to specify the 
                position of the label in the usual format (see comments for 
                addPlot)
            Prop is 'xLabel' or 'xAxis', want to override the default 
                position of the label / tick labels, and simply want to move
                the label / tick labels to the first plot (instead of being on 
                the final plot as default). Then pos should be the string 
                'first' or 'last' to specify on which plot to have the 
                label / tick labels
        """
        if isinstance(prop, str):
            prop = [prop]
        
        for thisProp in prop:
            self._addSingleShare(thisProp, tags, pos)


    def perform(self):
        """ Perform all the requested plotting. The plot should be completely
        specified using the other methods before calling this method.
        """
        self._checkShares()
        self._prepareColourBars()
        self._producePlots()
        self._implementShared()

        return self.fig
    

    def _addSingleShare(self, prop: str, tags: list, 
                        pos: None | str | dict = None):
        """ Same as addShare but adds a single property at a time. I.e. the 
        prop input must be a string, and cannot be a list.
        """
        assert isinstance(tags, list)
        sharedSpec = {
            'property': prop,
            'tags': tags,
            'pos': pos
        }

        if prop in ['xAxis', 'yAxis']:
            sharedSpec['axis'] = None

        if (prop in ['title', 'yLabel', 'yAxis']) and (pos is None):
            sharedSpec['pos'] = 'first'
        elif (prop in ['xLabel', 'xAxis']) and (pos is None):
            sharedSpec['pos'] = 'last'
        elif (prop in ['legend', 'colourBar']) and (pos is None):
            raise TypeError(f'Must supply a position for the {prop}.')
        assert sharedSpec['pos'] is not None
        
        self.shared.append(sharedSpec)
        self._checkShares(finalCheck=False)


    def _checkShares(self, finalCheck=True):
        """ Check that the requested shares make sense, and are consistent with
        the required format.

        INPUT
        finalCheck: bool. If True, run some additional checks that can only
            be run once we have all share information.
        """
        for thisShare in self.shared:
            assert np.all(np.isin(list(thisShare.keys()), 
                                  ['property', 'tags', 'axis', 'pos']))
            assert thisShare['property'] in ['title', 'xLabel', 'yLabel',
                                             'xAxis', 'yAxis',
                                             'legend', 'colourBar']
            
            entries = self._findShareEntries(thisShare['property'],
                                             thisShare['tags'])
            if len(entries) == 1:
                pass
            elif len(entries) > 1:
                raise ValueError('self.shared contains a duplicate or an'+
                                 'ambiguously overlapping '+
                                 f"shared {thisShare['property']}")
            else:
                raise AssertionError('Bug')

        if finalCheck:
            plotTags = []
            for thisPlot in self.plots:
                plotTags = plotTags + thisPlot['tags']
            shareTags = []
            for thisShare in self.shared:
                shareTags = shareTags + thisShare['tags']
            if not np.array_equal(np.unique(plotTags), np.unique(shareTags)):
                raise ValueError('Tags on plots and tags for shared '+
                                 'properties do not match.')

            if self.checkAllShare:
                for thisPlot in self.plots:
                    legend = self._findShareEntries('legend', 
                                                    thisPlot['tags'])
                    cBar = self._findShareEntries('colourBar', 
                                                  thisPlot['tags'])
                    if (len(legend) == 0) and (len(cBar) == 0):
                        raise ValueError('A plot has no shared legend or '
                                         'colour bar associated with it.')
            

    def _prepareColourBars(self):
        """ Look through all the shared colour bars, determine the range these
        colour bars need to have, and set them.
        """
        for thisShare in self.shared:
            if thisShare['property'] == 'colourBar':
                theseTags = thisShare['tags']
                
                shareColour = []
                for thisPlot in self.plots:
                    if np.any(np.isin(thisPlot['tags'], theseTags)):
                        shareColour.append(thisPlot['plotter'])

                cLims = []
                for thisPlot in shareColour:
                    cLims.append(thisPlot.findColourRange())
                cMin, cMax, cCentre = zip(*cLims)

                if np.all(np.isin(cCentre, [None])):
                    pass
                else:
                    assert len(np.unique(cCentre)) == 1

                cMin = np.min(cMin)
                cMax = np.max(cMax)

                for thisPlot in shareColour:
                    thisPlot.finaliseCBarSpec(cMin, cMax)
            else:
                assert thisShare['property'] in ['title', 'xLabel', 'yLabel', 
                                                    'xAxis', 'yAxis', 'legend']
    

    def _producePlots(self):
        """ Produce all subplots taking care of setting up shared axes if 
        requested.
        """
        self._checkShares()

        for thisPlot in self.plots:
            ax = self._addAxisForPlot(thisPlot)
            thisPlot['plotter'].plot(ax)


    def _addAxisForPlot(self, thisPlot):
        """ Add an axis acording to the specification for a specific plot, 
        including setting up axis sharing if requested.

        INPUT
        thisPlot: dict. One of the entries in self.plots.

        OUTPUT
        ax: The created axis.
        """
        shareSpec = {}
        for thisProp, thisKey in zip(['xAxis', 'yAxis'], ['sharex', 'sharey']):
            shareEntry = self._findShareEntry(thisProp, thisPlot['tags'])
            if shareEntry is None:
                shareSpec[thisKey] = None
            else:
                shareSpec[thisKey] = shareEntry['axis']

        ax = self._addAxis(thisPlot['row'], thisPlot['col'],
                            thisPlot['endRow'], thisPlot['endCol'],
                            **shareSpec)
        
        for thisProp in ['xAxis', 'yAxis']:
            shareEntry = self._findShareEntry(thisProp, thisPlot['tags'])
            if (shareEntry is not None) and (shareEntry['axis'] is None):
                shareEntry['axis'] = ax
            
        return ax

    
    def _addAxis(self, row, col, endRow=None, endCol=None, 
                 sharex=None, sharey=None, invis=False,
                 centreOnly=False):
        """ Add an axis at the specified point on the underlying grid.

        INPUT
        row, col: scalar. The first row and columns in the underlying grid  
            that the subplot should occupy
        endRow, endCol: scalar. The final row and columns in the underlying 
            grid that the subplot should occupy. Only needed if height/width
            is greater than one row/col.
        sharex, sharey: None | axes. If provided, then the newly created
            axies will share x- or y- axis with the axes passed as sharex
            or sharey.
        invis: bool. If True, make the created axis invisible.
        centreOnly: bool. If True, use only a central horizontal band of the
            area specified using row, col, endRow and endCol.

        OUTPUT
        ax: The created axis.
        """
        if endRow is None:
            endRow = row +1
        if endCol is None:
            endCol = col +1

        gridSec = self.grid[row:endRow, col:endCol]
        shareSpec = {}
        if sharex is not None:
            shareSpec['sharex'] = sharex
        if sharey is not None:
            shareSpec['sharey'] = sharey

        if centreOnly:
            fineGrid = gridspec.GridSpecFromSubplotSpec(10, 1, 
                                                        subplot_spec=gridSec)
            gridSec = fineGrid[3:7, :]

        ax = self.fig.add_subplot(gridSec, **shareSpec)

        if invis:
            ax.patch.set_visible(False)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
        applyDefaultAxisProperties(ax, invis)

        return ax
        

    def _findShareEntries(self, prop, tags):
        """ Find the relevant entries in self.shared given a particular 
        property and list of tags.

        INPUT
        prop: str. The property of the entry we are looking for.
        tags: list. We are interested in any entries with any of the
            listed tags.

        OUTPUT
        entries: list[dict]. List that contains all the entries matching
            the conditions.
        """
        entries = []
        for thisShare in self.shared:
            propMatch = thisShare['property'] == prop
            tagMatch = np.any(np.isin(tags, thisShare['tags']))
            
            if propMatch and tagMatch:
                entries.append(thisShare)
        
        return entries
    

    def _findShareEntry(self, prop, tags):
        """ Find the relevant entry in self.shared given a particular property
        and list of tags.

        INPUT
        prop: str. The property of the entry we are looking for.
        tags: list of str. We are interested in any entries with any of the
            listed tags.

        OUTPUT
        shareEntry: None | dict. Either none if there is no matching entry, or
            the matching share entry. If more that one entry matches an error
            will be raised.
        """
        matches = self._findShareEntries(prop, tags)

        if len(matches) == 0:
            shareEntry = None
        elif len(matches) == 1:
            shareEntry = matches[0]
        else: 
            raise ValueError('More than one entry matches the conditions.')
        return shareEntry
    

    def _findPlotWithTags(self, tags):
        """ Find all the plots that have any of the requested tags.

        INPUT
        tags: list[str]. The names of any of the tags that we want to look for.

        OUPPUT
        matches: list[Plotter instances]. A list as long as the number of 
            matching plots. Elements are Plotter instances from self.plots.
        """
        matches = []
        for thisPlot in self.plots:
            if np.any(np.isin(thisPlot['tags'], tags)):
                matches.append(thisPlot['plotter'])
        return matches
    

    def _implementShared(self):
        """ Impliment the sharing of properties: Remove duplicate properties 
        from the individual subplots and add shared properties as appropriate.
        """
        for thisShare in self.shared:
            if thisShare['property'] in ['title', 'xLabel', 'yLabel']:
                self._shareLabels(thisShare['property'], thisShare['tags'],
                                  thisShare['pos'])
            
            elif thisShare['property'] in ['xAxis', 'yAxis']:
                self._shareTicks(thisShare['property'], thisShare['tags'],
                                 thisShare['pos'])

            elif thisShare['property'] == 'legend':
                self._shareLegend(thisShare['tags'], **thisShare['pos'])

            elif thisShare['property'] == 'colourBar':
                self._shareColourbar(thisShare['tags'], **thisShare['pos'])

            else:
                raise ValueError('Unrecognised property')
        
    
    def _shareLabels(self, label, tags, pos):
        """ Remove duplicated title, x- or y-labels from a group of plots. An 
        error is raised if the existing labels do not match.

        INPUT
        label: str. 'title', 'xLabel' or 'yLabel'.
        tags: list[str]. List of tags identifying the plots that we want to 
            remove duplicate labels from.
        pos: str | dict. If str, should be either 'first' or 'last', to 
            determine on which plot (as ordered in self.plots) to leave the 
            tick labels. Alternatively, if a dict, all labels are removed and 
            the dict specifies the location for a new label. In this case pos 
            has the keys row, col, endRow, endCol, which have the usual 
            meanings (see comments for addPlot).
        """
        sharePlots = self._findPlotWithTags(tags)

        if label in ['xLabel', 'yLabel']:
            labelTxt = [thisPlot.axisLabels[label] for thisPlot in sharePlots]
            
            if np.all([thisTxt is None for thisTxt in labelTxt]):
                labelTxt = None
            else:
                labelTxt = np.unique(labelTxt)
                if len(labelTxt) != 1:
                    raise ValueError(f'{label} does not match across plots even '+
                                    'though requested to share.')
                labelTxt = labelTxt[0]

        elif label == 'title':
            errorMsg = 'Requested to share titles that do not match'
            checkSameAttr(sharePlots, 'title', errorMsg)
        else:
            raise ValueError('Unrecognised option for label')

        if isinstance(pos, str):
            if pos == 'last':
                toRemove = sharePlots[:-1]
            elif pos == 'first':
                toRemove = sharePlots[1:]
            else:
                raise ValueError('Unrecognised option for position')
        else:
            toRemove = sharePlots

        for thisPlot in toRemove:
            if label == 'xLabel':
                thisPlot.ax.set_xlabel(None)
            elif label == 'yLabel':
                thisPlot.ax.set_ylabel(None)
            elif label == 'title':
                thisPlot.ax.set_title(None)
            else:
                raise AssertionError('Bug')

        if not isinstance(pos, str):
            assert set(pos.keys()) == set(['row', 'col', 'endRow', 'endCol'])
            ax = self._addAxis(**pos, invis=True)

            if label == 'xLabel':
                ax.set_xlabel(labelTxt)
            elif label == 'yLabel':
                ax.set_ylabel(labelTxt)
            elif label == 'title':
                sharePlots[0].addTitle(ax)
            else:
                raise AssertionError('Bug')


    def _shareTicks(self, axis, tags, pos):
        """ Remove duplicated x- or y-tick labels from a group of plots. An 
        error is raised if the existing labels do not match.

        INPUT
        axis: str. 'xAxis' or 'yAxis'.
        tags: list[str]. List of tags identifying the plots that we want to 
            remove duplicate labels from.
        pos: str. 'first' or 'last'. On which plot (as ordered in self.plots)
            to leave the tick labels.
        """
        sharePlots = self._findPlotWithTags(tags)

        allTickLabels = []
        for thisPlot in sharePlots:
            if axis == 'xAxis':
                allTickLabels.append(thisPlot.ax.get_xticklabels())
            elif axis == 'yAxis':
                allTickLabels.append(thisPlot.ax.get_yticklabels())
            else:
                raise ValueError('Unknown option for axis')
        
        for theseLabels in allTickLabels:
            checkTickLabelsMatch(allTickLabels[0], theseLabels)

        if pos == 'last':
            toRemove = sharePlots[:-1]
        elif pos == 'first':
            toRemove = sharePlots[1:]
        else:
            raise ValueError('pos should be \'first\' or \'last\'')
        
        for thisPlot in toRemove:
            if axis == 'xAxis':
                tickLabels = thisPlot.ax.get_xticklabels()
            elif axis == 'yAxis':
                tickLabels = thisPlot.ax.get_yticklabels()
            else:
                raise AssertionError('Bug')
            plt.setp(tickLabels, visible=False)


    def _shareLegend(self, tags, row, col):
        """ Setup a shared legend for a group of plots, and remove the 
        individual legends.

        INPUT
        tags: list[str]. List of tags identifying the plots that we want to 
            remove duplicate labels from.
        row, col: scalars. Indicies of the first row and column in 
            the underlying grid that the legend should occupy.
        """
        sharePlots = self._findPlotWithTags(tags)

        errorMsg = 'Requested to share legends that do not match'
        checkSameAttr(sharePlots, 'legendSpec', errorMsg)
            
        for thisPlot in sharePlots:
            thisPlot.ax.get_legend().remove()

        ax = self._addAxis(row, col, invis=True)
        sharePlots[0].addLegend(ax)

    
    def _shareColourbar(self, tags, row, col):
        """ Setup a shared colourbar for a group of plots, and remove the 
        individual colourbars.

        INPUT
        tags: list[str]. List of tags identifying the plots that we want to 
            remove duplicate labels from.
        row, col: scalars. Indicies of the first row and column in 
            the underlying grid that the legend should occupy.
        """
        sharePlots = self._findPlotWithTags(tags)

        errorMsg = 'Requested to share colourbars that do not match'
        checkSameAttr(sharePlots, 'cBarSpec', errorMsg)
            
        for thisPlot in sharePlots:
            thisPlot.removeColourBar()

        ax = self._addAxis(row, col, invis=True, centreOnly=True)

        sharePlots[0].addColourBar(ax)


def setIfMissing(thisDict, thisKey, thisValue):
    """ Add a key and value to a dict, but only if the key is currently missing
    from the dict. 

    OUTPUT
    None. Dict is modified in place.
    """
    if not thisKey in thisDict:
        thisDict[thisKey] = thisValue


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
    dataframe with a single column (or a grouped pandas series) and a single
    index level and computes the mean and SEM for each group.

    INPUT
    grouped: grouped pandas dataframe. Must have a single column and a single
        index level.
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
    if len(grouped.mean().index.names) != 1:
        raise ValueError('Index should have only one level')

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
    if np.any(np.isnan(data.to_numpy())):
        raise ValueError('No data may be missing when performing permutation '+
                         'testing on a series.')

    _, _, pVals, _ = mne.stats.permutation_cluster_1samp_test(
                                    data.to_numpy(),
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
            "posErr": (optional) Error shading will be plotted between Y+posErr 
                and Y-posErr
            "sig": (optional) Contains boolean values. Where True a line 
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
            
        if 'posErr' in seriesData.columns:
            ax.fill_between(seriesData['X'].to_numpy(), 
                    seriesData['Y'].to_numpy()-seriesData['posErr'].to_numpy(), 
                    seriesData['Y'].to_numpy()+seriesData['posErr'].to_numpy(), 
                    color=sColours[iSeries], alpha=0.5)
        
        if 'sig' in seriesData.columns:
            plotHasSig = True
            sigRegions = findSigEnds(seriesData['X'], seriesData['sig'])
   
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


def findCenteredScale(lower, upper, centre):
    """ Find a scale that runs from at least lower to upper, but is extended
    (as little as possible) to ensure that it is centered on centre
    """
    assert upper > lower
    maxDiff = max(np.absolute([upper-centre, lower-centre]))

    newMin = centre - maxDiff
    newMax = centre + maxDiff

    assert newMin <= lower
    assert newMax >= upper
    return newMin, newMax


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
            cNorm: matplotlib normaliser instance. E.g. an initalised instance
                of pltColours.Normalize.

    OUTPUT
    Returns the figure produced/modified
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
        assert specs == ['addCBar', 'cMap', 'cNorm']
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
        cbarNorm = cbarSpec['cNorm']

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

        # TODO -- this option delete
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

    return fig


def checkTickLabelsMatch(tickLabels1, tickLabels2):
    """ Check two sets of tick labels are the same. Can find the tick 
    labels on a matplotlib axis using axis.get_xticklabels() or 
    axis.get_yticklabels()
    """
    assert len(tickLabels1) == len(tickLabels2)
    for this1, this2 in zip(tickLabels1, tickLabels2):
        assert this1.get_text() == this2.get_text()
        assert this1.get_position() == this2.get_position()


def checkSameAttr(objs, attr, errorMsg=None):
    """ Check a set of object instances all have an identical attribute.

    INTPUT
    objs: list. Object instances to check.
    attr: str. The attribute to check.
    errorMsg: str. The error message to use if check fails.
    """
    attrVals = [getattr(thisObj, attr) for thisObj in objs]
    for thisVal in attrVals:
        if thisVal != attrVals[0]:
            raise ValueError(errorMsg)
        

def applyDefaultAxisProperties(axis, invis=False):
    """ Apply some defaults to the appearance of the axes

    INPUT
    axis: The axis we want to modify
    invis: bool. If true use defaults appropriate for an invisible axis
    """
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)

    if invis:
        axis.spines['left'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
