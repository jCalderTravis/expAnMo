import os
import numpy as np
import pandas as pd
import time

def writeToTopOfLog(logFile, strToAdd, lockFile=True):
    """ Write a string to the first line of a log file (creating the file
    if it does not already exist).

    INPUT
    logFile: str. File path to a text file.
    strToAdd: str. The string to write
    lockFile: bool. If True prevent more than one process writting to the 
        log file at the same time. Should be set to True except when working
        on windows, where this functionality is not implemented.
    """
    if lockFile:
        import fcntl

    if not os.path.isfile(logFile):
        with open(logFile, 'x'):
            pass

    with open(logFile, 'r+') as log:
        # Prevent other processes from writing to the log file at the same time
        if lockFile:
            fcntl.flock(log, fcntl.LOCK_EX) 
        prevEntries = log.read()
        log.seek(0)
        log.write(strToAdd)
        log.write(prevEntries)
        if lockFile:
            fcntl.flock(log, fcntl.LOCK_UN)


def dispProgress(msg, repType="[Progress]"):
    """ Display the time along with the message msg

    repType: string. Will prefex the report text
    """
    timeStr = time.strftime("%H:%M:%S", time.localtime())
    print(repType + ' ' + msg + '  Time: ' + timeStr)


def avOutDfLevel(df, avOut):
    """ Take a dataframe with a multi-index and average out a specific level.

    INPUT
    df: dataframe with multi-index.
    avOut: str. The name of the index level to average out.
    """
    indexLvs = df.index.names
    assert avOut in indexLvs
    keepLvs = [thisLvl for thisLvl in indexLvs if thisLvl != avOut]
    assert avOut not in keepLvs
    assert len(keepLvs) == len(indexLvs) - 1

    df = df.groupby(keepLvs).mean()
    checkDfLevels(df, indexLvs=keepLvs)
    return df


def trimDfIndex(df: pd.DataFrame, idxLevel: str, win: tuple,
                assertTrim: bool = True,
                centreTol = np.inf) -> pd.DataFrame:
    """ Trim the rows of a dataframe based on the value of one level of the
    index.

    INPUT
    df: pandas dataframe.
    idxLevel: str. Name of the index level.
    win: 2-length tuple of scalars. All and only rows will be retained for 
        which the value of the index level idxLevel is greater than or equal
        to the first value of win, and smaller than or equal to the second 
        value.
    assertTrim: bool. If True check that we will be trimming at least some 
        data from both ends, otherwise raise an error.
    centerTol: scalar. Gives a tolerance. If set to a number smaller than 
        infinity the code checks that the average value of the relevant index 
        level is within this tolerance of the center of the trimming window. 

    OUTPUT
    df: pandas dataframe. A copy of the input modified accordingly.
    """
    assert len(win) == 2
    assert win[0] < win[1]
    
    vals = df.index.get_level_values(idxLevel).values

    if assertTrim:
        assert np.min(vals) < win[0]
        assert np.max(vals) > win[0]

    included = np.logical_and(
        vals >= win[0],
        vals <= win[1]
    )
    df = df.loc[included, :]

    winCentre = np.mean(win)
    dataCentre = np.mean(df.index.get_level_values(idxLevel).values)
    assert np.abs(dataCentre - winCentre) < centreTol

    return df


def subtractMatchingDfs(df1: pd.DataFrame, df2: pd.DataFrame, 
                        checkNan: bool = True) -> pd.DataFrame:
    """For two dataframes compute df1 - df2, but only after sorting the indexes
    and columns, and after checking that these columns and indexes are 
    matching.  

    INPUT
    df1
    df2
    checkNan: bool. If True, additionally check that the resulting dataframe
        contains no NaN entries.
    """
    dfs = [df1, df2]
    dfsSorted = []
    for thisDf in dfs:
        thisDf = thisDf.sort_index(axis=0)
        thisDf = thisDf.sort_index(axis=1)
        dfsSorted.append(thisDf)

    assert len(dfsSorted) == 2
    df1 = dfsSorted[0]
    df2 = dfsSorted[1]

    assert df1.index.equals(df2.index)
    assert df1.columns.equals(df2.columns)

    result = df1.sub(df2)
    if checkNan:
        assert not np.any(np.logical_or(np.isnan(result.values), 
                                        pd.isna(result).values))
    return result 


def checkDfLevels(df, indexLvs=None, colLvs=None, ignoreOrder=False):
    """ Check that the the levels of the index, or levels of the columns, of a
    pandas dataframe, match those that are expected.

    INPUT
    df: Dataframe to check. Can also pass a series as long as colLvs is None.
    indexLvs: list. Expected levels of the index. If None, is not checked
    colLvs: list. Expected levels of the columns. If None, is not checked
        ignoreOrder: boolean. If true, ignore the order of the index and column 
        levels
    """
    for thisCheck in [indexLvs, colLvs]:
        if (thisCheck is not None) and isinstance(thisCheck, str):
            raise TypeError('indexLvs or colLvs is not the correct type')

    levelInfo = {
        'exptIdxLvs': indexLvs,
        'realIdxLvs': df.index.names
    }
    if colLvs is not None:
        colLevelInfo = {
            'exptColLvs': colLvs,
            'realColLvs': df.columns.names
        }
        levelInfo.update(colLevelInfo)

    for key, val in levelInfo.items():
        levelInfo[key] = operateIfNotNone(np.asarray, val)
        
        if ignoreOrder:
            levelInfo[key] = operateIfNotNone(np.sort, val)
        
    if indexLvs is not None:
        if not np.array_equal(levelInfo['realIdxLvs'], 
                                levelInfo['exptIdxLvs']):
            raise mkDfLvsException('index', levelInfo['exptIdxLvs'],
                                levelInfo['realIdxLvs'])
    if colLvs is not None:
        if not np.array_equal(levelInfo['realColLvs'],
                                levelInfo['exptColLvs']):
            raise mkDfLvsException('columns', levelInfo['exptColLvs'],
                                levelInfo['realColLvs'])

    if (indexLvs is None) and (colLvs is None):
        raise ValueError("No check was requested")


def mkDfLvsException(axis, expectedLevels, actualLevels):
    """
    INPUT
    axis: str. 'index' or 'columns'
    """
    txt = ('Dataframe check failure: {} levels did not match the expected.'+
            '\nExpected: {}'+
            '\nActual: {}')
    return Exception(txt.format(axis.capitalize(), expectedLevels, 
                                actualLevels))


def operateIfNotNone(operation, thisInput):
    if thisInput is None:
        return thisInput
    else:
        return operation(thisInput)
    

def safeObjToFloat(array):
    """ Convert an array to float type, checking that each element is the same
    before and after the transformation.
    """
    newArray = array.astype(np.float64, copy=True)
    assert np.all(np.equal(array, newArray))
    return newArray


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


def dictCheckAndMerge(dictA, dictB):
    """ Check keys in dictA and dictB are unique and then merge
    """
    allKeys = [list(thisDict.keys()) for thisDict in [dictA, dictB]]
    if len(set(allKeys[0] + allKeys[1])) != \
        (len(allKeys[0]) + len(allKeys[1])):
        raise ValueError('Was requested to merge dictionaries ' +
                        'but keys are not unqiue.')

    dictC = {**dictA, **dictB}
    return dictC