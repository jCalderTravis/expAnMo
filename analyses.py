""" Classes an methods for coordinating multiple analysis steps. 
"""
from pathlib import Path
from copy import deepcopy
import os
import glob
from pprint import pprint
import numpy as np
import matplotlib
import mne
from expAnMo import helpers

class LowVerbosity():
    """ Context manager for temporarily setting Analysis.verbose = False.
    """
    def __init__(self):
        self.oldVerbose = Analysis.verbose

    def __enter__(self):
        Analysis.verbose = False

    def __exit__(self, *args):
        Analysis.verbose = self.oldVerbose
        return False # Reraise any exceptions that have been encountered


class Analysis(object):
    """ Analysis base class. An analysis step can be specified by creating an
    class that inherits from this class. Analyses can be called conveniently 
    using runPythonStep.py
    
    METHODS TO OVERRIDE
    run()
    nameSaveFiles()  
    specifyLoadFiles() 

    CLASS ATTRIBUTES
    requiredSettings: list of str. Used to check all and only permitted settings 
        have been provided in the call to the analysis. Does not need to contain
        any settings that appear in defaultSettings
    defaultSettings: dict (optional). dict of setting-name, setting-value 
        pairs, that specify any default settings values we would like. See the
        instance attribute 'settings' below.
    suppressIfDefault: list (optional). Contains a subset of the keys in the 
        dict defaultSettings. If a key is in this list, and the corresponding 
        setting is indeed the default value, then this setting-value pair will 
        somtimes be suppressed when naming files for saving.
    autoCallsPermitted: bool. If false automated calls to this 
        analysis (made when prerequisite files are missing) will not be 
        permitted and will raise an exception. Default is false.
    permitAllSessns bool. If true this analysis can be run after 
        sess='all' has been passed at initialisation. If false, a specific 
        session number must be provided at initialisation. 
    permitAllPtpnts: serves the same role as permitAllSessns but for participants.
    interactiveMatplotlib: controls whether to run matplotlib in interactive 
        mode (default False). 
    allowPartialNames: bool. If true allow save and load names to only be 
        partially specified, in that they contain the string "{}" (or multiple
        of such strings), that will be customised later. Using this 
        functionality is deprechiated, becuase it undermines the ability of
        the analysis class to evaluate which files belong to which analysis
        and whether all or only some such files already exist.

    INSTANCE ATTRIBUTES 
    settings: A dictionary set up when an analysis class is initialised using
        all keyword arguments as the name and values of the settings. Values of 
        settings should only be strings (this helps ensure these analyses can be 
        called easily from the command line).
    keyDirs: A dictionary containing some key directories, including the following.
        resultsMain: Directory in which to store all results from analysis
        megRawMain: Directory in which all MEG data from the experiment is located
        behavRawMain: Directory in which all behavioural data from the experiment 
            is located 
        thisInteractDir: The directory, specific to a participant, in which to save 
            intermediate results that the user must interact with (e.g. ICA info)
        thisStepsDir: The directory, specific to a participant, in which to save 
            intermediate results
        thisFinalDir: The directory, specific to a participant, in which save 
            final results
    loadFiles: None | dict. Once set up gives a dict that is a converted version
        of the dict returned by the method "specifyLoadFiles". Specifically, 
        "specifyLoadFiles" returns 2-length tuples, or a list of 2-length tuples. 
        The first element in each tuple should be an initialised instance object 
        (A) for one of the other analyses, and the second element should be a 
        string (B). This will be converted into another dictionary with the same 
        keys but values given by A.nameSaveFiles()[B]. 
    saveFiles: None | dict. Once set up gives the dictionary producted by the 
        method "nameSaveFiles". 
    """
    
    verbose = True # Don't override in subclasses, so can control 
    # verbosity of all analyses at once
    permitAllSessns = False
    permitAllPtpnts = False
    autoCallsPermitted = False
    njobs = 4
    interactiveMatplotlib = False
    allowPartialNames = False

    requiredSettings = []
    defaultSettings = dict()
    suppressIfDefault = []

    namingWarningsIssues = dict()
    namingWarningsIssues['Analysis'] = []
    namingWarningsIssues['Setting'] = []
    namingWarningsIssues['AssocVal'] = []


    def __init__(self, keyDirs, ptpnt, sess, **settings):
        """ 
        Creates some directories for saving, and stores lots of info.

        INPUT
        keyDirs: Dictionary of key directories used for saving and loading.
        ptpnt: string. must match the string used by freesurfer. May pass 'all' 
            for some analyses
        sess: int or string. 1 for behavioural session, 2 for first MEG 
            session and so on. Can use 'all' if session is not applicable, or 
            wish to analyse data from all sessions.
        settings: Pass any other relevant settings as keyword arguments and 
            these will be stored in a dictionary at self.settings
        """
        self.analysisName = self.__class__.__name__
        self.dispProgress = lambda msg: helpers.dispProgress(msg, 
                            ("[Scheduler] Analysis: "+self.analysisName+". "))

        if (not self.permitAllSessns) and (sess not in np.arange(1000)):
            self.dispProgress('The following error occoured setting up.')
            raise ValueError('Under the current settings, a specific session' +
                            ' number must be given. This analysis is ' +
                            '{}.'.format(self.analysisName))
        if (not self.permitAllPtpnts) and (ptpnt in ['all', 'All']):
            self.dispProgress('The following error occoured setting up.')
            raise ValueError('Under the current settings, a specific ' +
                            ' participant must be specified. This analysis '+
                            'is {}.'.format(self.analysisName))

        # We will add some participant and session specific entries to keyDirs,
        # so save a record of keyDirs without these specifics for later use
        assert(set(keyDirs.keys()) == set(['fsurferSubjects', 
                                            'codeMain', 'pyMegDir', 
                                            'dataMain', 'jobLibCache',
                                            'resultsMain', 'interactMain',
                                            'megRawMain',
                                            'behavRawMain',
                                            'behavProcessedMain',
                                            'splitRecordings']))
        self.keyGeneralDirs = keyDirs
        self.ptpnt = ptpnt
        self.keyDirs = self.createRequiredDirs(info_only=False)
        assert ((type(sess) in [int, float, np.int64]) or (sess == 'all'))
        self.sess = sess
        self.settings = settings
        self.applyDefaults()
        self.checkSettingsPermitted()
        self.loadFiles = None
        self.saveFiles = None


    def applyDefaults(self):
        """ Applies any default settings to self.settings. Retrieves default 
        settings from self.defaultSettings which should be a 
        dict of setting-name, setting-value pairs 
        """
        for settingName, settingValue in self.defaultSettings.items():
            if settingName in self.settings:
                pass
            else:
                self.settings[settingName] = settingValue
                if self.verbose:
                    self.dispProgress(
                        f'Default setting applied for {settingName}.')


    def checkSettingsPermitted(self):
        allPermitted = self.requiredSettings + list(
                                                self.defaultSettings.keys())
        allPermitted.sort()
        providedSettings = list(self.settings.keys())
        providedSettings.sort()
        
        warningMsg = ('For the analysis, '+self.analysisName+', '+
                    'the following settings were '+
                    ' provided but not in the '+
                    'list of permitted settings...')
        self.checkAllIn(providedSettings, allPermitted, warningMsg)

        warningMsg = 'Following settings are required but were not found...'
        self.checkAllIn(allPermitted, providedSettings, warningMsg)


    def checkAllIn(self, test, comparison, warningMsg):
        matched = np.isin(test, comparison)
        if not np.all(matched):
            print(warningMsg)
            test = np.array(test)
            print(test[np.logical_not(matched)])
            raise ValueError('Incorrect settings. See above for detail.')


    def spec(self, ovrWrtSess=None, ovrWrtPtpnt=None):
        """ Produce a dictionary continaing keyGeneralDirs, ptpnt, and 
        sess of this analysis. Helpful when want to initialise another analysis
        with the same specification.

        ovrWrtSess: If provided the returned dict uses this value for the
            key 'sess', instead of the actual session number
        ovrWrtPtpnt: Same as ovrWrtSess but for 'ptpnt'
        """
        spec = {
            'keyDirs': self.keyGeneralDirs,
            'ptpnt': self.ptpnt,
            'sess': self.sess
        }

        if ovrWrtSess != None:
            spec['sess'] = ovrWrtSess
        if ovrWrtPtpnt != None:
            spec['ptpnt'] = ovrWrtPtpnt

        return spec


    def displaySettings(self):
        print('Current settings:')
        for settingName, settingValue in self.settings.items():
            print('       {}...   {}'.format(settingName, settingValue))


    def createRequiredDirs(self, info_only=False):
        """ Create results directories if they don't exist already. 
        
        INPUT
        run: dict. Contains at least the following fields...
            Ptpnt: string. The identifier used for this participant.
        dirs: dict. Values are strings representing directories.
        info_only: boolean. If true, new directories are not created. 
            Instead information on which directories would have been created 
        if info_only was False is printed.

        OUTPUT
        Returns dirs with extra important directories added. Note that these extra
        directories are only correct for the participant specified in
        the input 'run'.
        """
        dirs = deepcopy(self.keyGeneralDirs)
        resultsDir = dirs['resultsMain']
        jobLibCacheDir = dirs['jobLibCache']

        # Specify directories
        interactDir = os.path.join(dirs['interactMain'], self.ptpnt, '')
        stepsDir = os.path.join(resultsDir, 'Steps', self.ptpnt, '')
        finalDir = os.path.join(resultsDir, 'Final', self.ptpnt, '')

        allDirs = [stepsDir, finalDir, jobLibCacheDir, dirs['splitRecordings'],
                interactDir]

        # Create directories for saving if they don't yet exist.
        for thisDir in allDirs:
            if info_only:
                print('Would create: ' + thisDir)
            else:
                Path(thisDir).mkdir(parents=True, exist_ok=True)

        # Add to dirs
        dirs['thisInteractDir'] = interactDir
        dirs['thisStepsDir'] = stepsDir
        dirs['thisFinalDir'] = finalDir

        if info_only:
            print('\n')
            print('All key directories ...')
            pprint(dirs)

        return dirs


    def findAllDirs(self):
        """ Return a dictionary contianing all the elements in self.loadFiles
        and self.saveFiles
        """
        return helpers.dictCheckAndMerge(self.loadFiles, self.saveFiles)


    def specifyLoadFiles(self):
        """ Return a dictionary where the values are 2-length tuples, or a 
        list of 2-length tuples. The first element in each tuple 
        should be an initialised instance object (A) for one of the other 
        analyses, and the second element should be a string (B). Later this 
        will be converted into another dictionary with the same keys but values 
        given by A.nameSaveFiles()[B]. The dictionary may be empty if there is
        nothing to load.
        """
        raise NotImplementedError('This analysis has not specified load files')


    def nameSaveFiles(self):
        """ Return a dictionary in which the values specify filenames to be 
        used for saving. The dictionary may be empty if there are no files
        to save.
        """
        raise NotImplementedError('This analysis has not specified save files')


    def safeNameSaveFiles(self):
        """ Like nameSaveFiles but also checks that the produced filenames 
        are not too long.
        """
        saveFiles = self.nameSaveFiles()
        for thisFile in saveFiles.values():
            if len(thisFile) > 255:
                raise ValueError(('File names are too long to use on Linux.'+
                                 '\nOffending filename: {}'+
                                 '\nLength {}/255').format(thisFile, 
                                                          len(thisFile))) 
        return saveFiles


    def run(self):
        """ Run the analysis itelf. 
        """
        raise NotImplementedError('This analysis has specified no analysis to '
                                  'run.')


    def prepAndRun(self, automatedCall, skipIfPos=False):
        """ Make the necessary preparations, including running prerequisite 
        analyses that have not yet been run, before calling run()

        INPUT
        automatedCall: bool. Was the call to this function made automatically 
            or specifically requested by the user? 
        skipIfPos: bool. If true and all the results files for this analysis
            already exist, then simply return without running anthing.
        """
        self.dispProgress('File checks, and analysis run requested.')
        if automatedCall and (not self.autoCallsPermitted):
            raise ValueError('Automated calls to this analysis are not '+
                            'permitted')
       
        if self.interactiveMatplotlib:
            self.dispProgress('Using MNE interactive plotting backend')
            # By default an interactive matplotlib backend is active
            mne.viz.set_browser_backend('qt') 
        else:
            self.dispProgress('Using matplotlib with non-interactive backend')
            mne.viz.set_browser_backend('matplotlib') 
            matplotlib.use(backend='AGG')

        # Check whether the output already exists
        with LowVerbosity():
            saveFiles = self.safeNameSaveFiles()
        if saveFiles is None:
            raise AssertionError('Requesting save names only returns None')
        globSaveFiles = self.makeGlobStringDict(saveFiles)
        status, _ = self.checkForFiles(globSaveFiles)

        if status == 'all':
            text = 'All files for this analysis have already been produced.'
            if automatedCall:
                raise AssertionError(text + 'Hence, an automated call '+
                                     'should not have been made.')
            elif skipIfPos:
                self.dispProgress('*** Skipping analysis *** even though ' +
                                  'directly called for the folling '+
                                  'reason: {}'.format(text))
                return
            else: 
                self.dispProgress(text + ' Running again anyway as requested.')
        elif status == 'some':
            self.dispProgress('Some files for this analysis have already been '+
                                'produced, but not all. Running the analysis.')

        # Check whether the required input already exists. If files are missing
        # run a required ancestor analysis, and then repeat the whole 
        # procedure again. Note it is important to do a fresh file check as 
        # one ancestor analyis may produce several of the missing files.
        filesMissing = True
        with LowVerbosity():
            loadFiles, ancestors = self.convertLoadSpecToNames()
        globLoadFiles = self.makeGlobStringDict(loadFiles)

        while filesMissing:
            status, missInfo = self.checkForFiles(globLoadFiles)
            
            if (status == 'some') or (status == 'none'):
                self.dispProgress('Files required for this analysis have not yet' +
                                    ' been produced. Calling ancestor analysis.')
                firstMissInfo = missInfo[0]
                missingAnalysis = ancestors[firstMissInfo[0]][firstMissInfo[1]]
                missingAnalysis.prepAndRun(automatedCall=True)
            else:
                filesMissing = False
          
        line = '*******************************' 
        print('\n'+line+line+line)
        self.dispProgress('Begining analysis.')
        print('Current participant: {}'.format(self.ptpnt))
        print('Current session: {}'.format(self.sess))
        self.displaySettings()
        print('\nDirectories in use:')
        pprint(self.keyDirs)
        print('\nPreparing load and save names...')
        self.loadFiles, _ = self.convertLoadSpecToNames()
        self.saveFiles = self.nameSaveFiles()
        print('\nFiles in use for loading:')
        pprint(self.loadFiles)
        print('\nFiles in use for saving:')
        pprint(self.saveFiles)
        print(line+line+line+'\n')

        self.run()

        # Check the analysis saved everything it claimed it would to avoid 
        # infinite loops of calls to this analysis
        status, _ = self.checkForFiles(globSaveFiles)
        if status not in  ['all', 'noneSpecified']:
            raise AssertionError('Analysis ran, but not all files were '
                                 'produced.')


    def convertLoadSpecToNames(self):
        """ Calls specifyLoadFiles(). This produces a specification of where to 
        find the filenames for loading. This function follows the instructions
        to find those files.
        
        OUTPUT
        loadFiles: a dict containing file names to load, as strings or lists of
        strings.
        ancestors: a dict with the same keys as loadFiles but containing the 
        analyses to run if a file to load is missing. Values are all lists of 
        analysis instance objects (with each list being as long as the 
        corresponding list in loadFiles).
        """
        loadSpec = self.specifyLoadFiles()
        if loadSpec is None:
            raise AssertionError('Requesting load names only returns None')
        loadFiles = dict()
        ancestors = dict()
        for itemName, theseLoadSpecs in loadSpec.items():
            # theseLoadSpecs may be a single tuple or a list of tuples
            if type(theseLoadSpecs) != list:
                theseLoadSpecs = [theseLoadSpecs]
                convertFromList = True
            else:
                convertFromList = False

            theseLoadFiles = []
            theseAncestors = []
            for thisLoadSpec in theseLoadSpecs:
                thisAncestor = thisLoadSpec[0]
                try:
                    thisLoadFile = thisAncestor.nameSaveFiles()[
                                                            thisLoadSpec[1]]
                except:
                    self.dispProgress('Error loading from ' + 
                                        thisAncestor.analysisName)
                    print('Ancestor analysis has the following keys for '+
                            ' save files...')
                    pprint(thisAncestor.nameSaveFiles().keys())

                    print('\nThe requested key was... ')
                    pprint(thisLoadSpec[1])
                    raise

                theseLoadFiles.append(thisLoadFile)
                theseAncestors.append(thisAncestor)

            if convertFromList:
                assert(len(theseLoadFiles) == 1)
                theseLoadFiles = theseLoadFiles[0]
            loadFiles[itemName] = theseLoadFiles
            ancestors[itemName] = theseAncestors

        return loadFiles, ancestors

    def makeGlobStringDict(self, dirs):
        """ We often specify files to load or save as the values of a 
        dictionary. We often use {} in file names to allow low-level functions
        to further customise file names. Here we remove all {} entries and 
        insert a * so can use these file names as glob strings.

        INPUT
        dirs: The dict specifying files to load as strings or lists of strings
        """
        globStrings = dict()
        for itemName, theseFiles in dirs.items():
            # theseFiles may be a string or a list of strings
            if not isinstance(theseFiles, list):
                theseFiles = [theseFiles]
            
            theseGlobStrings = []
            for thisFile in theseFiles:
                assert isinstance(thisFile, str)
                if ('{}' in thisFile) and (not self.allowPartialNames):
                    raise ValueError('Under the current settings save and '
                                     'load names are not permitted to contain '
                                     'the charachters "{}".')
                
                theseGlobStrings.append(thisFile.replace('{}', '*'))

            globStrings[itemName] = theseGlobStrings

        return globStrings

    def checkForFiles(self, dirs):
        """ Check for the existance of a list of files, and return info on
        whether all, some, or none exist, and the specific missing files.

        INPUT
        dirs: dict. Values are glob strings or lists of glob strings to check

        OUPUT
        status: str. 'all', 'some' or 'none' exist. If dirs specifies no files
            to check then result='noneSpecified'
        missInfo: list of 2-length tuples. Each tuple contains info on a 
            missing file. Specifically, the first element specifies the 
            corresponding key in dirs (thisKey), and the second element 
            specifies the index of the file name in the list given by 
            dirs[thisKey]
        """
        someFound = False
        someMissng = False
        missInfo = []
        for itemName, theseFiles in dirs.items():
            # theseFiles may be a string or a list of strings
            if type(theseFiles) != list:
                theseFiles = [theseFiles]

            for iFile, thisFile in enumerate(theseFiles):
                matches = glob.glob(thisFile)

                if len(matches) == 0:
                    someMissng = True
                    missInfo.append((itemName, iFile))
                else:
                    someFound = True
        
        if (someFound == True) and (someMissng == False):
            status = 'all'
        elif (someFound == False) and (someMissng == True):
            status = 'none'
        elif (someFound == True) and (someMissng == True):
            status = 'some'
        if (someFound == False) and (someMissng == False):
            status = 'noneSpecified'
            assert(len(missInfo) == 0)

        return status, missInfo


    def checkSuppressIfDefault(self):
        for thisSetting in self.suppressIfDefault:
            if thisSetting not in self.defaultSettings:
                raise AssertionError('All entries in self.suppressIfDefault '+
                                     'must be keys in self.defaultSettings '+
                                     'but the follwing setting is missing: '+
                                     '{} from {}.'.format(thisSetting,
                                                          self.analysisName))
    

    def autoName(self, thisDir, addText='', ext=''):
        """ Generate a name for saving files based on the current analysis 
        settings. Uses the name of the analysis, participant num, session num,
        and all the options in self.settings. Can be further customised with
        a string provided as addText

        INPUT
        thisDir: str. Directory in which to save the file
        addText: str. Appended to the end of the save name
        ext: str. Whether to use a specific extension. See code for details.
        """
        self.checkSuppressIfDefault()

        if self.sess == 'all':
            session = 'All'
        else:
            session = self.sess
        
        saveName = '{}_s{}_{}'
        saveName = saveName.format(self.ptpnt, session, self.analysisName)

        # When naming always want to do it in the same order so sort first
        for setting in sorted(self.settings):
            assocVal = self.settings[setting]
            assert(type(assocVal) == str)

            suppressIfDefault = setting in self.suppressIfDefault
            if setting in self.defaultSettings:
                isDefault = assocVal == self.defaultSettings[setting]
            else:
                isDefault = False

            if suppressIfDefault and isDefault:
                # Have we already issued this warning?
                analysisMatch = [self.analysisName == prev for prev in
                                    self.namingWarningsIssues['Analysis']]
                settingMatch = [setting == prev for prev in 
                                self.namingWarningsIssues['Setting']]
                valMatch = [assocVal == prev for prev in 
                            self.namingWarningsIssues['AssocVal']]
                matchesPrev = np.logical_and(analysisMatch,
                                        np.logical_and(settingMatch, valMatch))
                assert np.sum(matchesPrev) <= 1

                if np.sum(matchesPrev) == 0:
                    self.dispProgress('Suppressing the following '+
                        'setting-value pair from the save name '+
                        'because was requested to, and the setting matches '+
                        'the default value: {}... {}'.format(
                                                            setting, assocVal))
                    self.namingWarningsIssues['Analysis'].append(
                                                            self.analysisName)
                    self.namingWarningsIssues['Setting'].append(setting)
                    self.namingWarningsIssues['AssocVal'].append(assocVal)
            else:
                saveName += '_' + setting + '-' + assocVal

        if addText != '':
            saveName += '_' + addText

        if ext != '':
            if ext == 'numpy':
                extNameSeg = '.npy'
            elif ext == 'transMat':
                extNameSeg = '-trans.fif'
            elif ext == 'ica':
                extNameSeg = '-ica.fif'
            elif ext == 'raw':
                extNameSeg = '-raw.fif'
            elif ext == 'epochs':
                extNameSeg = '-epo.fif'
            elif ext == 'hdf':
                extNameSeg = '.hdf'
            elif ext == 'csv':
                extNameSeg = '.csv'
            elif ext == 'pdf':
                extNameSeg = '.pdf'
            elif ext == 'png':
                extNameSeg = '.png'
            else:
                raise Exception('Unrecognised option')
            
            saveName += extNameSeg

        saveName = os.path.join(thisDir, saveName)
        return saveName
    

class Test(Analysis):
    requiredSettings = []
    defaultSettings = dict()

    def run(self):
        print('Test only')


class Example(Analysis):
    """ Example analysis subclass

    REQURIED SETTINGS
    """
    autoCallsPermitted = True
    permitAllSessns = False
    permitAllPtpnts = False
    requiredSettings = []
    defaultSettings = dict()
    suppressIfDefault = []

    def specifyLoadFiles(self):
        pass

    def nameSaveFiles(self):
        pass

    def run(self):
        pass