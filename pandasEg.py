"""
Dataframe examples
"""

import numpy as np
import pandas as pd
idx = pd.IndexSlice

def makeDf(withNan=False):
    time = np.arange(0, 0.3, 0.05)
    time = np.tile(time, 3)
    trial = np.arange(3, dtype=float)
    trial = np.repeat(trial, 6)
    cluster = np.array(["A", "B", "C"])
    cluster = np.tile(cluster, 3)
    freq = np.array([5, 10, 90])
    freq = np.repeat(freq, 3)

    if withNan:
        trial[-1]=np.nan

    df = pd.DataFrame(np.random.randn(18, 9), index=[time, trial], 
                                                columns=[cluster, freq])
    df.index.set_names(["time", "trial"], inplace=True)
    df.columns.set_names(["cluster", "freq"], inplace=True)

    # newCol = np.arange(10, 10+len(time))
    # df2 = pd.DataFrame(newCol, index=time, columns=["new_col"])
    # df2.index.set_names("time", inplace=True)

    # df3 = pd.concat([df, df2], axis=1)
    # df.insert(len(df.columns), "new_col", newCol)


    # df2 = pd.DataFrame(np.random.randn(18, 9), index=[time, trial], 
    #                                             columns=cluster)
    # df2.index.set_names(["time", "trial"], inplace=True)
    # df2.columns.set_names(["cluster"], inplace=True)

    return df


def makeDf2(withNan=False):
    time = np.arange(0, 0.3, 0.05)
    time = np.tile(time, 3)
    trial = np.arange(3, dtype=float)
    trial = np.repeat(trial, 6)
    cluster = np.array(["A", "B", "C"])
    cluster = np.tile(cluster, 3)
    freq = np.array([5, 10, 90])
    freq = np.repeat(freq, 3)

    if withNan:
        trial[-1]=np.nan

    df = pd.DataFrame(np.random.randn(18, 9), index=[time, trial], 
                                                columns=[cluster, freq])
    df.index.set_names(["time", "trial"], inplace=True)
    df.columns.set_names(["cluster", "freq"], inplace=True)
    
    return df

def makeDf3():
    time = np.arange(0, 0.3, 0.05)
    time = np.tile(time, 3)
    trial = np.arange(3, dtype=float)
    trial = np.repeat(trial, 6)
    freq = np.array([5, 10, 90])
    freq = np.repeat(freq, 3)

    allDfs = []
    sessNums = np.arange(2)
    for iDf in sessNums:
        df = pd.DataFrame(np.random.randn(18, 9), index=[time, trial], 
                                                    columns=[freq])
        df.index.set_names(["time", "trial"], inplace=True)
        df.columns.set_names(["freq"], inplace=True)
        allDfs.append(df)

    df = pd.concat(allDfs, keys=sessNums, names=['Session'])

    time = np.flip(time)
    vals = np.arange(10, 10+len(time))
    df2 = pd.DataFrame(vals, index=[time, trial], columns=['MergeVal'])
    df2.index.set_names(['time', 'trial'], inplace=True)

    return df, df2
