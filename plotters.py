""" Tools for plotting, including for plotting several related plots as part 
of larger subplots.

"""
from copy import deepcopy
import numpy as np
import pandas as pd
from . import helpers


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


    def group(self, strategy: str, 
                opts: dict, 
                groupName: str,
                sepFor: None | str | list[str] = None):
        """ Group data and add a new column to the data, givng the group to
        which each case belongs.

        INPUT
        strategy: str. Grouping strategy. Options are...
            'col': The values in a column directly as the groups. 
            'specifiedEdges': The x-values are binned based on specified bin
                edges.
        opts: dict. Depends on the value of strategy. The keys and 
            values for each strategy option are as follows.
                'col': Keys and values...
                    'col': The name of the column to use directly as the groups
                'specifiedEdges': Keys and values...
                    'col': The name of the column upon which the binning is 
                        to be conducted.
                    'edges': Sequence of scalars giving the bin edges to use.
        groupName: str. The name of the column that will be added to the data
            giving the group to which each case belongs.
        sepFor: None | str | list[str]. Name or names of columns in the 
            dataset. Grouping is performed seperately for each unique 
            combination of these variables.

        TODO Add option to group by percentile See groupby(...)[var].rank(pct=true)
        """
        startCases = len(self.data)
        startCols = deepcopy(list(self.data.columns))

        if sepFor is None: 
            self.data = self._groupAll(self.data, strategy, opts, groupName)
        else:
            if isinstance(sepFor, str):
                sepFor = [sepFor]
            assert isinstance(sepFor, list)

            grouped = self.data.groupby(sepFor)
            processed = []
            for _, group in grouped:
                thisProcessed = self._groupAll(group, strategy, opts, 
                                               groupName)
                processed.append(thisProcessed)

            processed = pd.concat(processed, verify_integrity=True)
            assert len(processed) == len(startCases)
            assert list(processed.columns) == startCols + [groupName]
            self.data = processed


    def _groupAll(self, data: pd.DataFrame, strategy: str, opts: dict, 
                  groupName: str):
        """ Perform grouping of all input data. Similar functionality to group
        except operates on the input, not on self.data, and cannot perform 
        grouping seperately for different subsets of the data.

        INPUT
        data: pandas dataframe to perform the grouping on. Data is deepcopied
            to ensure no mutation of the input data.
        strategy, opts, groupName: Same as input to the method .group()

        OUTPUT
        data: pandas dataframe. Copy of the input data with the grouping 
            perormed and the results of the grouping stored in a new column
            with name groupName.
        """
        data = deepcopy(data)
        if groupName in data.columns:
            raise ValueError('Grouping data cannot be assigned to requested '+
                             'columns because it already exists.')

        if strategy == 'col':
            assert set(opts.keys()) == {'col'}
            data[groupName] = data[opts['col']]
        
        elif strategy == 'specifiedEdges':
            assert set(opts.keys()) == {'col', 'edges'}
            relCol = opts['col']
            data[groupName] = pd.cut(data[relCol],
                                        bins=opts['edges'],
                                        labels=False)
            assert not np.any(np.isnan(data[groupName]))
        else:
            raise ValueError('Unrecognised grouping option')
        
        return data
            

    def runSeriesStats(self, xVar: str, yVar: str, group: str, 
                       obsID: None | str = None,
                       checkEqual: bool = True):
        """ Run statistics on series data.

        INPUT
        xVar: str. Name of column giving the values of the independent 
            variable.
        yVar: str. Name of column giving the values of the dependent variable.
        group: str. Name of column giving the variable that will be used
            to bin the data. Averaging, and statistical tests are performed 
            for each bin. For example, this might give a binned version of the 
            xVar, or xVar itself, if xVar takes discrete values. Data will
            be returned in assending order of this grouping variable, therefore
            it must be numeric. Groups which are adjacent acording to numeric
            ordering of this grouping variable are also treated as ajdacent 
            for cluster-based permutation tests.
        obsID: None | str. Only needed if want to perform cluster based 
            permutation tests. In this case it should be the name of a column
            giving the independent observation to which each datapoint belongs. 
            In this case, group takes on the more specific meaning of 
            identifying the different repeated measurements that make up
            each observeration. E.g. obsID could be a unique identifier for 
            each participant.
        checkEqual: bool. If true, check that there are the same number
            of cases in each group. Must be true if performing cluster based
            permutation tests (i.e. if obsID was provided).
        """
        if obsID is not None:
            assert checkEqual

        grouped = self.data.groupby(group)
        if checkEqual:
            checkGroupsEqual(grouped)

        xMean, _, xGroup = groupedToMeanSd(grouped[xVar])
        yMean, ySem, xGroupForY = groupedToMeanSd(grouped[yVar])
        assert np.array_equal(xGroup, xGroupForY)
        assert np.all(np.diff(xGroup) > 0)


    def runStats(col: str | list[str], 
                 sepFor: str | list[str],
                 caseID: None | str = None, 
                 checkEqual: bool = True)
        """
        INPUT
        col: str. The name of the columns we want to perform statistics on.
            Statistics are performed seperately for each column. -- this becomes
            dict TODO with keys and column names and values as the stats that we want
        sepFor: str | list[str]. Name or names of columns in the 
            dataset. Statistics are performed seperately for each unique 
            combination of these variables (i.e. each unique combination 
            corresponds to one comparison), before potentially being combined 
            if e.g FDR correction, or cluster-based permutation tests are 
            requested.
        obsID: None | str. Only needed if performing cluster based permutation
            tests. In which case it should be a column in the dataset giving
            the independent observation to which each datapoint belongs. E.g.
            this could be a unique identifier for each participant.
        checkEqual: bool. If true, check there are an equal number of cases
            in each comparison performed.

        OUTPUT
        statsData: columns for each var in col col_mean col_SEM col_sig

        TODO SR*meg.brainPlotting.thresholdClsts
        TODO oldPlotting.OneSeriesSetPlotter.prepareData
        """
        pass
        # WORKING HERE -- NOT USING ANYMORE -- now writing specific functions for specific 
        # stats


def checkGroupsEqual(grouped):
    """ Check there are an equal number of cases in each group, after 
    performing grouping with pandas.

    INPUT
    grouped: pandas groupby instance.
    """
    counts = grouped.count()
    counts = np.unique(counts.to_numpy())
    if len(counts) != 1:
        raise ValueError('Number of cases being averages is not '+
                            'constant.')
    

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
    mean = removeDimIfNeeded(mean)
    group = meanDf.index.to_numpy()

    semDf = grouped.sem()
    if sort:
        semDf = semDf.sort_index()
    sem = semDf.to_numpy()
    sem = removeDimIfNeeded(sem)
    group2 = semDf.index.to_numpy()

    assert np.array_equal(group, group2)
    assert np.all([np.ndim(thisOut) == 1 
                  for thisOut in [mean, sem, group]])
    return mean, sem, group


def removeDimIfNeeded(thisArray):
    """ Expects either a 1D array, or a 2D array, where the shape of the second
    dimention is 1. This dimention will then be removed.
    """
    if np.ndim(thisArray) == 1:
        pass
    elif np.ndim(thisArray) == 2:
        assert thisArray.shape[1] == 1
        thisArray = np.squeeze(thisArray, axis=1)
    else:
        raise AssertionError('Unexpected data shape')
    
    return thisArray



################### WORKING HERE -- below in progres ########################
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