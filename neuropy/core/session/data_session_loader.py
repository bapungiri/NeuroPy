from pathlib import Path
from neuropy.core.session.Formats.Specific.BapunDataSessionFormat import BapunDataSessionFormatRegisteredClass
from neuropy.core.session.Formats.Specific.KDibaOldDataSessionFormat import KDibaOldDataSessionFormatRegisteredClass
from neuropy.core.session.Formats.Specific.RachelDataSessionFormat import RachelDataSessionFormat
    

class DataSessionLoader:
    """ An extensible class that performs session data loading operations. 
        Data might be loaded into a Session object from many different source formats depending on lab, experimenter, and age of the data.
        Often this data needs to be reverse engineered and translated into the correct format, which is a tedious and time-consuming process.
        This class allows clearly defining and documenting the requirements of a given format once it's been reverse-engineered.
        
        Primary usage methods:
            DataSessionLoader.bapun_data_session(basedir)
            DataSessionLoader.kdiba_old_format_session(basedir)
    """    
    # pix2cm = 287.7698 # constant conversion factor for spikeII and IIdata (KDiba) formats
    
    #######################################################
    ## Public Methods:
    #######################################################
    
    # KDiba Old Format:
    @staticmethod
    def bapun_data_session(basedir=r'R:\data\Bapun\Day5TwoNovel', override_parameters_flat_keypaths_dict=None):
        _test_session = BapunDataSessionFormatRegisteredClass.build_session(Path(basedir), override_parameters_flat_keypaths_dict=override_parameters_flat_keypaths_dict)
        _test_session, loaded_file_record_list = BapunDataSessionFormatRegisteredClass.load_session(_test_session)
        return _test_session
        
    # KDiba Old Format:
    @staticmethod
    def kdiba_old_format_session(basedir=r'R:\data\KDIBA\gor01\one\2006-6-07_11-26-53', override_parameters_flat_keypaths_dict=None):
        _test_session = KDibaOldDataSessionFormatRegisteredClass.build_session(Path(basedir), override_parameters_flat_keypaths_dict=override_parameters_flat_keypaths_dict)
        _test_session, loaded_file_record_list = KDibaOldDataSessionFormatRegisteredClass.load_session(_test_session)
        return _test_session
        
    
    
    # RachelFormat:
    @staticmethod
    def rachel_format_session(basedir=r'R:\data\Rachel\merged_M1_20211123_raw_phy', override_parameters_flat_keypaths_dict=None):
        _test_session = RachelDataSessionFormat.build_session(Path(basedir), override_parameters_flat_keypaths_dict=override_parameters_flat_keypaths_dict)
        _test_session, loaded_file_record_list = RachelDataSessionFormat.load_session(_test_session)
        return _test_session
       

    

        