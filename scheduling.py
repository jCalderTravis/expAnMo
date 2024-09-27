""" Functions for submitting jobs on a computer cluster

KEY FUNCTIONS
submitJobs: Can be used to submit multiple job submission scripts to slurm,
    for different participants and sessions
runJob: Can be used in a job submission script to launch an analysis
"""
import os
import subprocess
import datetime
from copy import deepcopy
from expAnMo import helpers

logInfo = {}

def setLogConfig(dirForTodos, ptpnt, sess, slurmID):
    """ Also updates the 
    'directory' dictionary in this module, with some relevant directories.
    """
    print('Updating information that will be used for logging TODOs')
    logInfo['centralTodoLog'] = os.path.join(dirForTodos, 'z_todos.txt')
    logInfo['ptpnt'] = ptpnt
    logInfo['sess'] = sess
    logInfo['slurmID'] = slurmID


def writeToDo(logStr):
    """ Write a "to-do" for the user, to the centrally maintained log of 
    to-dos.
    
    INPUT
    logStr: str. Added to log message.
    """
    thisTime = datetime.datetime.now()
    thisTime = thisTime.strftime('%d.%m.%Y %H:%M:%S')

    todo = '{}, {}, sess {} (slurm {}) -- {}'.format(thisTime, 
                                                     logInfo['ptpnt'],
                                                     logInfo['sess'],
                                                     logInfo['slurmID'],
                                                     logStr) 
    helpers.writeToTopOfLog(logInfo['centralTodoLog'], todo)


def writeToOverviewLog(thisDir, success, slurmID, ptpnt, sess, command,
                       lockFile=True):
    """ Write a line to the overview log file.

    INPUT
    thisDir: str. The directory of the log file. .
    success: bool. True if the step completed succesfully, False otherwise.
    ptpnt: str.
    sess: str.
    inCommand: str. The command asked to run by the user.
    lockFile: bool. If True prevent more than one process writting to the 
        log file at the same time. Should be set to True except when working
        on windows, where this functionality is not implemented.
    """
    overviewLog = os.path.join(thisDir, 'z_overviewLog.txt') 

    thisTime = datetime.datetime.now()
    thisTime = thisTime.strftime('%d.%m.%Y %H:%M:%S')

    if success:
        succStr = 'COMPLETED'
    else:
        succStr = 'ERROR    '

    strToAdd = '{} -- SlurmID {} -- {} -- {} -- sess {} -- {} \n'.format(
                                                                    succStr,
                                                                    slurmID, 
                                                                    thisTime,
                                                                    ptpnt,
                                                                    sess,
                                                                    command)
    helpers.writeToTopOfLog(overviewLog, strToAdd, lockFile)
    

def submitJobs(mode, step, ptpnt, sess, script, sessMode, skipBehav):
    """ Submit multiple jobs.

    INPUT
    mode: 'slurm' or 'local' for whether to submit to slurm or run locally (or
        'test' for debugging)
    step: Requested processing step
    ptpnt: The identifier used for this participant or group of participants. 
        e.g. 'Pilot01' or 'allMain'. May also pass a comma seperated list with 
        no spaces eg. "Pilot01,Pilot02,Pilot04" to run the step for several 
        participants.
    sess: The number of the session to analyse, or two numbers seperated by a 
        colon with no spaces eg. 1:4. In this case sessions 1 - 4 (inclusive) 
        are processed. Can pass 'all' if session number is not applicable or 
        wish to analyse all data simultanously. In this case the colon notation 
        can not be used.
    script: str. The name of the bash submission script. 
        E.g. ./submitOneJob.sh
    sessMode: 'serial' or 'parallel'. If colon notation is used to specify the 
        sessions to analyse, then this input determines whether each session is 
        analysed one after another as part of a single job (e.g. if on slurm
        will submit only 1 job to the scheuduler), or whether different 
        sessions are analysed as a seperate jobs (e.g. if on slurm will submit 
        a number of jobs equal to the number of sessions).
    skipBehav: 'skipIfPos' or 'alwaysRun'. Determines the behaviour of the 
        analysis, if one of the analyses in analyses.py is called, and if all 
        results files already exist.
    """
    if ',' in ptpnt:
        allPtpnts = ptpnt.split(',')
    else:
        allPtpnts = [ptpnt]

    if sessMode == 'parallel':
        sess = splitSessSpec(sess)
    elif sessMode == 'serial':
        sess = [sess]
    else:
        raise ValueError('Unknown input')

    print('')
    print('=========== Scheduler info ===========')
    print(f'Will submit jobs from {os.getcwd()}')
    print(f'Sumbmitting with the script {script}')
    print('For the following participants:')
    print(allPtpnts)
    print('And the following sessions:')
    print(sess)
    print('(If individual sessions are specified, '+
            f'these will be processed in {sessMode})')
    print('======================================')
    print('')

    for thisPtpnt in allPtpnts:
        for thisSess in sess:
            command = makeCommand(mode, script, skipBehav, step, 
                                  thisPtpnt, thisSess)
            subprocess.run(command, check=True)


def makeCommand(mode, script, skipBehav, thisStep, thisPtpnt, thisSess):
    
    if mode == 'local':
        command = [script]
    elif mode == 'slurm':
        command = ['sbatch', script]
    elif mode == 'test':
        command = ['echo', 'submitOneTestJob']
    else:
        raise ValueError('Requested mode of operation not recognised.')
    
    command = command + [thisStep, thisPtpnt, thisSess, skipBehav]
    return command


def splitSessSpec(sess):
    """ Sessions to analyse are specified as strings in a particiular form.
    Convert to a list of session specifications, with one entry for each job to
    submit.

    INPUT
    sess: str.

    OUTPUT
    allSess: list of str.
    """
    assert type(sess) is str

    if (':' in sess) and (not ('all' in sess)) :
        firstSess, lastSess = sess.split(':')
        allSess = range(int(firstSess), (int(lastSess)+1))
        allSess = [str(thisSess) for thisSess in allSess]
    else:
        allSess = [sess]
    
    return allSess


def runJob(module, keyDirs, inCommand, ptpnt, sess, slurmID, skipBehav,
           lockFile):
    """ Run a single python job.

    INPUT
    module: Python module that contains the analysis to run
    keyDirs: dict. Dictionary of key directories that will be passed down to
        the analysis.
    inCommand: string. Name of the processing/analysis step(s) to run. See 
        the function analysis.py for options. Keyword areguments may be passed in 
        the following format: analysisName-kwargName=kwargVal-kwargName=kwargVal. 
        Multiple steps may be requstested. To do this use a single string 
        containing the names of mulitple steps, separated by the 
        charachters '-THEN-'
    ptpnt: The identifier used for this participant or group of participants. 
        e.g. 'Pilot01' or 'allMain'. 
    sess: str. The number of the session to analyse. Alternatively can pass two 
        numbers seperated by a colon with no spaces e.g. 1:4. In this case 
        sessions 1 - 4 (inclusive) are processed serially.
    slurmID: If applicable, the slurm job ID. Pass "notSlm" if not run via 
        slurm.
    skipIfPos: 'skipIfPos' or 'alwaysRun'. Determines the behaviour of the 
        analysis if all results files already exist.
    lockFile: bool. If True prevent more than one process writting to the 
        log file at the same time. Should be set to True except when working
        on windows, where this functionality is not implemented.
    """

    allSess = splitSessSpec(sess)

    if skipBehav == 'skipIfPos':
        skipIfPos = True
    elif skipBehav == 'alwaysRun':
        skipIfPos = False
    else:
        raise ValueError('Unknown option selectied.')

    print('')
    print('-- JOB REPORT --')
    print('Within this job, the following analyses will be '+
          'conducted serially:')
    for thisSess in allSess:
        print(f'{ptpnt}, sess {thisSess}')
    print('')

    try:
        for thisSess in allSess:
            runSess(module, keyDirs, inCommand, ptpnt, thisSess, skipIfPos, 
                    slurmID)
        writeToOverviewLog(keyDirs['codeMain'], True, slurmID,
                            ptpnt, sess, inCommand, lockFile)
    except:
        writeToOverviewLog(keyDirs['codeMain'], False, slurmID,
                           ptpnt, sess, inCommand, lockFile)
        raise


def runSess(module, keyDirs, inCommand, ptpnt, sess, skipIfPos, slurmID):
    keyDirs = deepcopy(keyDirs)
    
    if sess != 'all':
        formattedSess = int(sess)
    else:
        formattedSess = sess

    setLogConfig(keyDirs['codeMain'], ptpnt, formattedSess, slurmID)
    run = {
        'Ptpnt': ptpnt,
        'Sess': formattedSess
    }
    runStep(module, inCommand, run, keyDirs, skipIfPos)


def runStep(module, inCommand, run, keyDirs, skipIfPos):
    """
    INPUT
    module: Python module that contains the analysis to run
    skipIfPos: bool. If true and all the results files for this analysis
            already exist, then simply return without running anthing.
    """
    # Which processing steps have been requested and with what keyword arguments
    # Format for a single step requests is 
    # analysisName-kwargName=kwargVal-kwargName=kwargVal
    steps = inCommand.split('-THEN-')
    analysisNms = list()
    kwargs = list()
    for thisStep in steps:
        if '-' in thisStep:
            parts = thisStep.split('-')
            analysisNms.append(parts[0])
            
            kwargSpecs = parts[1:]
            theseKwargs = dict()
            for thisSpec in kwargSpecs:
                thisSpecSplt = thisSpec.split('=')
                assert(len(thisSpecSplt) == 2)
                theseKwargs[thisSpecSplt[0]] = thisSpecSplt[1]
            kwargs.append(theseKwargs)
        else:
            analysisNms.append(thisStep)
            kwargs.append(dict())

    assert(len(analysisNms) == len(kwargs))

    for thisName, thisKwargs in zip(analysisNms, kwargs):
        ThisAnalysis = getattr(module, thisName)
        rdyAnalysis = ThisAnalysis(keyDirs, run['Ptpnt'], run['Sess'],
                                    **thisKwargs)
        rdyAnalysis.prepAndRun(automatedCall=False, skipIfPos=skipIfPos)


def findSlurmID():
    if "SLURM_JOB_ID" in os.environ:
        slurmID = os.getenv("SLURM_JOB_ID")
    else:
        slurmID = "notSlm"

    return slurmID