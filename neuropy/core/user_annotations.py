import sys
from copy import deepcopy
from typing import List, Dict, Any, Tuple, Optional, Callable
from attrs import define, field, Factory, asdict
from nptyping import NDArray
from pathlib import Path
import numpy as np
import pandas as pd
import tables as tb
from datetime import datetime
from neuropy.utils.misc import numpyify_array
from neuropy.utils.result_context import IdentifyingContext
from neuropy.utils.result_context import IdentifyingContext as Ctx

from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, serialized_field, serialized_attribute_field, non_serialized_field, custom_define
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

# ==================================================================================================================== #
# 2023-06-21 User Annotations                                                                      #
# ==================================================================================================================== #

""" 
Migrated from 
from pyphoplacecellanalysis.General.Model.user_annotations import UserAnnotationsManager
to
from neuropy.core.user_annotations import UserAnnotationsManager

"""


# @custom_define(slots=False)
# class UserAnnotationRecord(HDFMixin):
#     """ CONCEPTUAL, NEVER USED ANYWHERE - an annotation made by the user at a specific date/time covering a specific context """
#     created: datetime
#     modified: datetime
#     context: IdentifyingContext
#     content: dict

# class UserAnnotationTable(tb.IsDescription):
#     """ conceptual pytables table"""
#     created = tb.Time64Col()
#     modified = tb.Time64Col()
#     context = tb.StringCol(itemsize=320)

@custom_define(slots=False)
class SessionCellExclusivityRecord:
    """ 2023-10-04 - Holds hardcoded specifiers indicating whether a cell is LxC/SxC/etc """
    LxC: np.ndarray = serialized_field(default=Factory(list))
    LpC: np.ndarray = serialized_field(default=Factory(list))
    Others: np.ndarray = serialized_field(default=Factory(list))
    SpC: np.ndarray = serialized_field(default=Factory(list))
    SxC: np.ndarray = serialized_field(default=Factory(list))
    


@custom_define(slots=False)
class UserAnnotationsManager(HDFMixin, AttrsBasedClassHelperMixin):
    """ class for holding User Annotations of the data. Performed interactive by the user, and then saved to disk for later use. An example are the selected replays to be used as examples. 
    
    Usage:
        from neuropy.core.user_annotations import UserAnnotationsManager
        
    """
    annotations: Dict[IdentifyingContext, Any] = serialized_field(default=Factory(dict))


    def __attrs_post_init__(self):
        """ builds complete self.annotations from all the separate hardcoded functions. """

        for a_ctx, a_val in self.get_hardcoded_specific_session_override_dict().items():
            self.annotations[a_ctx] = a_val

        for a_ctx, a_val in self.get_user_annotations().items():
            self.annotations[a_ctx] = a_val

        for a_ctx, a_val in self.get_hardcoded_specific_session_cell_exclusivity_annotations_dict().items():
            # Not ideal. Adds a key 'session_cell_exclusivity' to the extant session context instead of being indexable by an entirely new context
            self.annotations[a_ctx] = self.annotations.get(a_ctx, {}) | dict(session_cell_exclusivity=a_val)
            # annotation_man.annotations[a_ctx.overwriting_context(user_annotation='session_cell_exclusivity')] = a_val


    @function_attributes(short_name=None, tags=['XxC','LxC', 'SxC'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-10-05 16:18', related_items=[])
    def add_neuron_exclusivity_column(self, neuron_indexed_df, included_session_contexts, neuron_uid_column_name='neuron_uid'):
        """ adds 'XxC_status' column to the `neuron_indexed_df`: the user-labeled cell exclusivity (LxC/SxC/Shared) status {'LxC', 'SxC', 'Shared'}
        
            annotation_man = UserAnnotationsManager()
            long_short_fr_indicies_analysis_table = annotation_man.add_neuron_exclusivity_column(long_short_fr_indicies_analysis_table, included_session_contexts, aclu_column_name='neuron_id')
            long_short_fr_indicies_analysis_table
    
        """
        LxC_uids = []
        SxC_uids = []

        for a_ctxt in included_session_contexts:
            session_uid = a_ctxt.get_description(separator="|", include_property_names=False)
            session_cell_exclusivity: SessionCellExclusivityRecord = self.annotations[a_ctxt].get('session_cell_exclusivity', None)
            LxC_uids.extend([f"{session_uid}|{aclu}" for aclu in session_cell_exclusivity.LxC])
            SxC_uids.extend([f"{session_uid}|{aclu}" for aclu in session_cell_exclusivity.SxC])
            
        neuron_indexed_df['XxC_status'] = 'Shared'
        neuron_indexed_df.loc[np.isin(neuron_indexed_df[neuron_uid_column_name], LxC_uids), 'XxC_status'] = 'LxC'
        neuron_indexed_df.loc[np.isin(neuron_indexed_df[neuron_uid_column_name], SxC_uids), 'XxC_status'] = 'SxC'

        return neuron_indexed_df




    
    @staticmethod
    def get_user_annotations():
        """ hardcoded user annotations
        

        New Entries can be generated like:
            from pyphoplacecellanalysis.GUI.Qt.Mixins.PaginationMixins import SelectionsObject
            from neuropy.core.user_annotations import UserAnnotationsManager
            from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import DecodedEpochSlicesPaginatedFigureController

            ## Stacked Epoch Plot
            example_stacked_epoch_graphics = curr_active_pipeline.display('_display_long_and_short_stacked_epoch_slices', defer_render=False, save_figure=True)
            pagination_controller_L, pagination_controller_S = example_stacked_epoch_graphics.plot_data['controllers']
            ax_L, ax_S = example_stacked_epoch_graphics.axes
            final_figure_context_L, final_context_S = example_stacked_epoch_graphics.context

            user_annotations = UserAnnotationsManager.get_user_annotations()

            ## Capture current user selection
            saved_selection_L: SelectionsObject = pagination_controller_L.save_selection()
            saved_selection_S: SelectionsObject = pagination_controller_S.save_selection()
            final_L_context = saved_selection_L.figure_ctx.adding_context_if_missing(user_annotation='selections')
            final_S_context = saved_selection_S.figure_ctx.adding_context_if_missing(user_annotation='selections')
            user_annotations[final_L_context] = saved_selection_L.flat_all_data_indicies[saved_selection_L.is_selected]
            user_annotations[final_S_context] = saved_selection_S.flat_all_data_indicies[saved_selection_S.is_selected]
            # Updates the context. Needs to generate the code.

            ## Generate code to insert int user_annotations:
            print('Add the following code to UserAnnotationsManager.get_user_annotations() function body:')
            print(f"user_annotations[{final_L_context.get_initialization_code_string()}] = np.array({list(saved_selection_L.flat_all_data_indicies[saved_selection_L.is_selected])})")
            print(f"user_annotations[{final_S_context.get_initialization_code_string()}] = np.array({list(saved_selection_S.flat_all_data_indicies[saved_selection_S.is_selected])})")


        Usage:
            user_anootations = get_user_annotations()
            user_anootations

        """
        # from neuropy.utils.result_context import IdentifyingContext as Ctx
        # from numpy import array

        user_annotations = {}

        ## IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15')
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = [13,  14,  15,  25,  27,  28,  31,  37,  42,  45,  48,  57,  61,  62,  63,  76,  79,  82,  89,  90, 111, 112, 113, 115]
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = [  9,  11,  13,  14,  15,  20,  22,  25,  37,  40,  45,  48,  61, 62,  76,  79,  84,  89,  90,  93,  94, 111, 112, 113, 115, 121]

        # 2023-07-19 - New Annotations, lots more than before.
        with Ctx(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15',display_fn_name='DecodedEpochSlices',user_annotation='selections') as ctx:
            with (ctx + IdentifyingContext(epochs='replays')) as ctx:
                user_annotations[ctx + Ctx(decoder='long_results_obj')] = [1, 3, 11, 13, 14, 15, 17, 21, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 36, 37, 39, 42, 43, 44, 45, 46, 48, 51, 52, 53, 55, 57, 58, 60, 61, 62, 68, 69, 70, 72, 74, 76, 81, 84, 85, 86, 87, 88, 89, 90, 91, 92, 95, 96, 97, 98, 100, 101, 105, 106, 109, 112, 113, 114, 115, 118, 119, 120, 121, 123, 124, 125, 126, 127, 128, 130, 131, 132]
                user_annotations[ctx + Ctx(decoder='short_results_obj')] = [2, 3, 4, 8, 9, 10, 11, 13, 14, 15, 16, 17, 20, 21, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 36, 38, 39, 40, 41, 42, 43, 44, 46, 48, 49, 51, 53, 55, 63, 64, 66, 67, 69, 70, 72, 75, 77, 78, 80, 81, 83, 84, 85, 86, 87, 88, 91, 92, 93, 95, 96, 97, 98, 99, 100, 101, 102, 104, 105, 106, 107, 110, 111, 112, 113, 114, 115, 116, 118, 119, 120, 121, 122, 123, 124, 126, 127, 131, 132]
            with (ctx + IdentifyingContext(epochs='ripple')) as ctx:
                user_annotations[ctx + Ctx(decoder='long_LR')] = [[181.692, 181.900], [188.797, 189.046], [193.648, 193.893], [210.712, 211.049], [218.107, 218.507], [241.692, 241.846], [282.873, 283.142], [323.968, 324.431], [380.259, 380.597], [419.651, 419.876], [571.061, 571.225], [749.247, 749.420], [799.927, 800.140], [878.106, 878.315], [1158.821, 1159.079], [1191.678, 1191.890], [1285.369, 1285.508], [1306.519, 1306.778], [1492.934, 1493.018], [1493.483, 1493.693], [1818.494, 1818.659], [1848.992, 1849.224]]
                user_annotations[ctx + Ctx(decoder='long_RL')] = [[64.877, 65.123], [240.488, 240.772], [398.601, 399.047], [911.215, 911.588], [1152.558, 1152.761], [1368.482, 1368.847], [1504.300, 1504.475], [1523.409, 1523.857], [1811.138, 1811.395]]
                user_annotations[ctx + Ctx(decoder='short_LR')] = [[61.397, 61.662], [72.607, 72.954], [77.735, 78.048], [91.578, 91.857], [241.692, 241.846], [338.425, 338.644], [367.853, 368.127], [380.259, 380.597], [485.046, 485.367], [630.858, 631.206], [689.063, 689.285], [743.166, 743.293], [799.927, 800.140], [815.267, 815.468], [867.655, 867.836], [906.068, 906.269], [1136.947, 1137.501], [1296.292, 1296.619], [1325.176, 1325.339], [1378.883, 1379.019], [1410.607, 1410.783], [1453.569, 1453.790], [1456.699, 1457.127], [1485.888, 1486.146], [1492.934, 1493.018], [1493.483, 1493.693], [1530.547, 1530.794], [1540.999, 1541.243], [1658.499, 1658.923], [1807.340, 1807.478], [1818.494, 1818.659], [1832.060, 1832.191], [1835.039, 1835.215], [1848.992, 1849.224], [1866.809, 1867.072], [1892.861, 1893.089], [1998.453, 1998.566]]
                user_annotations[ctx + Ctx(decoder='short_RL')] = [[41.012, 41.359], [146.972, 147.245], [204.162, 204.322], [267.606, 267.774], [303.683, 303.898], [341.709, 342.028], [398.601, 399.047], [543.325, 543.487], [799.366, 799.545], [1318.305, 1318.500], [1424.499, 1424.714], [1472.388, 1472.686], [1513.619, 1513.767], [1519.637, 1519.792], [1633.030, 1633.265], [1697.468, 1697.711], [1780.357, 1780.492], [1789.761, 1790.010], [1840.475, 1840.606], [1855.422, 1855.711], [1892.268, 1892.522], [1925.050, 1925.228], [1966.703, 1967.053], [2045.095, 2045.393], [2051.142, 2051.269]]
                            

        # Pho + Kamran Selected and Agreed on these events together on 2024-07-25 ____________________________________________ #
        with Ctx(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15',display_fn_name='DecodedEpochSlices',epochs='ripple',user_annotation='selections') as ctx:
            user_annotations[ctx + Ctx(decoder='long_LR')] = [[282.873, 283.142], [323.968, 324.431], [380.259, 380.597], [419.651, 419.876], [434.102, 434.316], [571.061, 571.225], [878.106, 878.315], [1158.821, 1159.079], [1493.483, 1493.693]]
            user_annotations[ctx + Ctx(decoder='long_RL')] = [[911.215, 911.588], [1152.558, 1152.761], [1855.422, 1855.711]]
            user_annotations[ctx + Ctx(decoder='short_LR')] = [[193.648, 193.893], [241.692, 241.846], [267.606, 267.774], [338.425, 338.644], [485.046, 485.367], [545.744, 546.074], [689.063, 689.285], [749.247, 749.420], [799.927, 800.140], [815.267, 815.468], [886.561, 886.679], [906.068, 906.269], [1285.369, 1285.508], [1296.292, 1296.619], [1325.176, 1325.339], [1378.883, 1379.019], [1410.607, 1410.783], [1453.569, 1453.790], [1456.699, 1457.127], [1504.300, 1504.475], [1523.409, 1523.857], [1540.999, 1541.243], [1818.494, 1818.659], [1835.039, 1835.215], [1848.992, 1849.224], [1866.809, 1867.072], [1885.974, 1886.315], [2053.277, 2053.445]]
            user_annotations[ctx + Ctx(decoder='short_RL')] = [[303.683, 303.898], [398.601, 399.047], [663.843, 664.227], [799.366, 799.545], [1296.292, 1296.619], [1318.305, 1318.500], [1368.482, 1368.847], [1424.499, 1424.714], [1472.388, 1472.686], [1513.619, 1513.767], [1633.030, 1633.265], [1780.357, 1780.492], [1840.475, 1840.606], [1859.245, 1859.388], [1892.268, 1892.522], [1915.874, 1916.032], [1925.050, 1925.228], [2051.142, 2051.269]]



        ## IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19')
        with IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19',display_fn_name='DecodedEpochSlices',user_annotation='selections') as ctx:
            with (ctx + IdentifyingContext(epochs='replays')) as ctx:
                user_annotations[ctx + Ctx(decoder='long_results_obj')] = [5,  13,  15,  17,  20,  21,  24,  31,  33,  43,  44,  49,  63, 64,  66,  68,  70,  71,  74,  76,  77,  78,  84,  90,  94,  95, 104, 105, 122, 123]
                user_annotations[ctx + Ctx(decoder='short_results_obj')] = [ 12,  13,  15,  17,  20,  24,  30,  31,  32,  33,  41,  43,  49, 54,  55,  68,  70,  71,  73,  76,  77,  78,  84,  89,  94, 100, 104, 105, 111, 114, 115, 117, 118, 122, 123, 131]
            with (ctx + IdentifyingContext(epochs='ripple')) as ctx:
                user_annotations[ctx + Ctx(decoder='long_LR')] = [[292.624, 292.808], [304.440, 304.656], [380.746, 380.904], [785.350, 785.851], [873.001, 873.269], [953.942, 954.258], [1006.840, 1007.132], [1024.832, 1025.103], [1193.988, 1194.152], [2212.470, 2212.538], [2214.238, 2214.441], [2214.655, 2214.684], [2219.725, 2219.871], [2248.613, 2248.810], [2422.595, 2422.819], [2451.063, 2451.225], [2452.072, 2452.222], [2453.381, 2453.553], [2470.819, 2470.974], [2472.998, 2473.150]]
                user_annotations[ctx + Ctx(decoder='long_RL')] = [[461.141, 461.290], [487.205, 487.451], [489.425, 489.656], [518.520, 518.992], [524.868, 525.022], [528.483, 528.686], [572.248, 572.442], [802.912, 803.114], [803.592, 803.901], [804.192, 804.338], [831.621, 831.910], [893.989, 894.103], [914.058, 914.391], [977.822, 978.050], [982.605, 982.909], [1034.816, 1034.863], [1035.124, 1035.314], [1096.396, 1096.547], [1184.870, 1185.008], [1200.697, 1200.903], [1273.352, 1273.544], [1274.121, 1274.443], [1292.484, 1292.619], [1380.751, 1380.890], [1448.171, 1448.337], [1746.254, 1746.426], [1871.004, 1871.223], [2050.894, 2050.995], [2051.248, 2051.677]]
                user_annotations[ctx + Ctx(decoder='short_LR')] = [[488.296, 488.484], [677.993, 678.341], [721.893, 722.227], [876.270, 876.452], [888.227, 888.465], [890.870, 891.037], [922.816, 923.107], [950.183, 950.448], [953.942, 954.258], [1006.840, 1007.132], [1044.949, 1045.453], [1096.396, 1096.547], [1129.646, 1129.839], [1259.291, 1259.445], [1259.725, 1259.882], [1284.424, 1284.590], [1329.846, 1330.068], [1511.195, 1511.428], [1511.974, 1512.058], [1549.236, 1549.366], [1558.468, 1558.679], [1560.657, 1560.753], [1561.312, 1561.414], [1561.818, 1561.886], [1655.991, 1656.213], [1730.888, 1731.068], [1734.810, 1734.952], [1861.411, 1861.532], [1909.781, 1910.039], [1967.738, 1968.092], [2036.973, 2037.326], [2038.027, 2038.271], [2038.531, 2038.732], [2042.389, 2042.639], [2064.367, 2064.545], [2070.819, 2071.031], [2119.100, 2119.256], [2125.548, 2125.738], [2132.661, 2132.983], [2153.029, 2153.144], [2170.259, 2170.624], [2191.256, 2191.387], [2192.123, 2192.362], [2193.779, 2193.987], [2194.559, 2194.763], [2200.654, 2200.799], [2201.847, 2202.033], [2219.725, 2219.871], [2229.350, 2229.845], [2248.613, 2248.810], [2249.704, 2249.915], [2313.885, 2314.058], [2372.291, 2372.609], [2422.595, 2422.819], [2462.666, 2462.743], [2482.132, 2482.613], [2484.411, 2484.484], [2530.725, 2530.921], [2531.219, 2531.302], [2556.112, 2556.383], [2556.597, 2556.918]]
                user_annotations[ctx + Ctx(decoder='short_RL')] = [[66.662, 66.779], [470.140, 470.318], [518.520, 518.992], [788.911, 789.495], [831.621, 831.910], [870.931, 871.079], [888.227, 888.465], [890.870, 891.037], [910.571, 911.048], [953.942, 954.258], [982.605, 982.909], [996.526, 997.074], [1001.634, 1001.967], [1014.096, 1014.277], [1033.052, 1033.226], [1096.396, 1096.547], [1136.294, 1136.682], [1200.697, 1200.903], [1211.214, 1211.332], [1214.611, 1214.835], [1266.591, 1266.719], [1317.706, 1318.221], [1333.493, 1333.689], [1380.751, 1380.890], [1381.958, 1382.319], [1448.171, 1448.337], [1499.587, 1499.711], [1620.637, 1620.793], [1744.335, 1744.587], [1798.635, 1798.766], [1970.811, 1970.953], [1994.072, 1994.253], [2050.894, 2050.995], [2051.248, 2051.677], [2077.674, 2077.962], [2132.661, 2132.983], [2203.726, 2203.819], [2204.539, 2204.658], [2317.030, 2317.124], [2330.010, 2330.161], [2331.841, 2331.958], [2333.100, 2333.317], [2335.649, 2335.817], [2349.621, 2349.833], [2368.841, 2369.000], [2379.690, 2379.862], [2403.112, 2403.409], [2456.244, 2456.327], [2456.467, 2456.573], [2457.493, 2458.009], [2505.335, 2505.672]]

        with Ctx(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19',display_fn_name='DecodedEpochSlices',epochs='ripple', source='diba_evt_file', user_annotation='selections') as ctx:
            user_annotations[ctx + Ctx(decoder='long_LR')] = [[240.289, 240.593], [257.224, 257.379], [400.450, 400.635], [556.017, 556.236], [922.935, 923.138], [1193.992, 1194.155], [1201.830, 1202.019]]
            user_annotations[ctx + Ctx(decoder='long_RL')] = [[240.289, 240.593], [380.729, 380.971], [422.649, 422.887], [452.370, 452.589], [461.147, 461.271], [524.869, 525.018], [528.485, 528.687], [784.232, 784.383], [838.186, 838.448], [888.232, 888.400], [908.192, 908.403], [910.828, 911.051], [967.634, 967.792], [1035.121, 1035.307], [1272.890, 1273.020], [1746.259, 1746.427], [1970.832, 1970.955], [2080.650, 2080.778], [2087.270, 2087.419], [2192.079, 2192.268]]
            user_annotations[ctx + Ctx(decoder='short_LR')] = [[224.721, 225.028], [292.625, 292.852], [295.973, 296.110], [304.521, 304.657], [487.266, 487.460], [512.776, 512.873], [873.107, 873.271], [890.921, 891.040], [934.106, 934.419], [950.225, 950.387], [977.841, 978.060], [1024.842, 1025.106], [1111.265, 1111.420], [1211.218, 1211.317], [1256.835, 1256.975], [1266.599, 1266.700], [1301.772, 1301.972], [1359.901, 1360.075], [1500.969, 1501.076], [1530.095, 1530.241], [1569.808, 1569.904], [1683.428, 1683.516], [1743.419, 1743.537], [1751.104, 1751.226], [1804.385, 1804.521], [1838.754, 1838.848], [1851.774, 1851.870], [1905.212, 1905.317], [1949.899, 1950.214], [2111.682, 2111.803], [2125.632, 2125.742], [2234.936, 2235.059], [2248.665, 2248.811], [2307.667, 2307.764], [2313.892, 2314.001], [2330.034, 2330.168], [2333.119, 2333.322], [2376.105, 2376.246], [2401.432, 2401.567], [2530.734, 2530.928], [2556.120, 2556.388]]
            user_annotations[ctx + Ctx(decoder='short_RL')] = [[257.224, 257.379], [478.725, 478.830], [502.732, 502.870], [647.955, 648.097], [676.013, 676.334], [833.416, 833.582], [907.500, 907.610], [954.047, 954.202], [996.896, 997.077], [1033.094, 1033.220], [1045.288, 1045.398], [1096.399, 1096.598], [1164.642, 1164.875], [1184.873, 1185.032], [1214.615, 1214.826], [1233.264, 1233.444], [1326.652, 1326.756], [1371.850, 1371.976], [1427.235, 1427.394], [1453.582, 1453.741], [1797.982, 1798.092], [1843.176, 1843.283], [1870.983, 1871.187], [1943.534, 1943.632], [1967.960, 1968.088], [2038.594, 2038.738], [2042.447, 2042.645], [2051.554, 2051.683], [2077.861, 2077.959], [2214.245, 2214.395], [2255.984, 2256.074], [2275.941, 2276.092], [2328.668, 2328.822], [2335.671, 2335.817], [2368.900, 2368.982], [2408.837, 2409.006], [2536.522, 2536.617]]


        with Ctx(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19',display_fn_name='DecodedEpochSlices',epochs='ripple',user_annotation='selections') as ctx:
            user_annotations[ctx + Ctx(decoder='long_LR')] = []
            user_annotations[ctx + Ctx(decoder='long_RL')] = [[304.440, 304.656], [380.746, 380.904], [487.205, 487.451], [489.425, 489.656], [802.912, 803.114]]
            user_annotations[ctx + Ctx(decoder='short_LR')] = [[292.624, 292.808], [304.440, 304.656], [873.001, 873.269], [1259.725, 1259.882], [1326.545, 1326.765], [1329.846, 1330.068], [1620.637, 1620.793], [1917.094, 1917.270], [1967.738, 1968.092], [1970.811, 1970.953], [2038.531, 2038.732], [2077.674, 2077.962], [2183.966, 2184.102], [2248.613, 2248.810], [2333.100, 2333.317], [2372.291, 2372.609], [2482.132, 2482.613]]
            user_annotations[ctx + Ctx(decoder='short_RL')] = [[66.662, 66.779], [461.141, 461.290], [470.140, 470.318], [487.205, 487.451], [528.483, 528.686], [572.248, 572.442], [888.227, 888.465], [898.303, 898.433], [950.183, 950.448], [953.942, 954.258], [1001.634, 1001.967], [1033.052, 1033.226], [1044.949, 1045.453], [1096.396, 1096.547], [1184.870, 1185.008], [1193.988, 1194.152], [1200.697, 1200.903], [1211.214, 1211.332], [1214.611, 1214.835], [1284.424, 1284.590], [1333.493, 1333.689], [1380.751, 1380.890], [1453.578, 1453.742], [1732.093, 1732.291], [1742.701, 1742.809], [1751.031, 1751.221], [1776.568, 1776.789], [1871.004, 1871.223], [2050.285, 2050.667], [2051.248, 2051.677], [2064.367, 2064.545], [2119.100, 2119.256], [2376.093, 2376.239], [2505.335, 2505.672]]


            
        # ==================================================================================================================== #
        #MARK '2006-6-12_15-55-31'                                                                                                
        # ==================================================================================================================== #
        with Ctx(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-12_15-55-31',display_fn_name='DecodedEpochSlices',epochs='ripple',user_annotation='selections') as ctx:
            user_annotations[ctx + Ctx(decoder='long_LR')] = [[735.218, 735.375], [1006.693, 1006.829]]
            user_annotations[ctx + Ctx(decoder='long_RL')] = [[535.654, 535.812], [540.199, 540.442], [685.390, 685.632], [706.614, 706.937], [712.075, 712.343], [1021.844, 1021.978]]
            user_annotations[ctx + Ctx(decoder='short_LR')] = [[615.091, 615.238], [735.218, 735.375]]
            user_annotations[ctx + Ctx(decoder='short_RL')] = [[127.873, 128.242], [153.506, 153.851], [165.036, 165.260], [535.654, 535.812], [540.199, 540.442], [712.075, 712.343], [761.180, 761.553], [892.791, 893.041], [1021.844, 1021.978], [1028.172, 1028.618]]
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-12_15-55-31',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = [10, 11, 12, 17, 18, 22]
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-12_15-55-31',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = [10, 11, 12, 16, 18, 19, 23]

        # ==================================================================================================================== #
        #MARK '11-02_17-46-44'                                                                                                #
        # ==================================================================================================================== #
        # IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_17-46-44')
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_17-46-44',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = [13, 23, 41, 46]
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_17-46-44',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = [4, 7, 10, 15, 21, 23, 41]
        with Ctx(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_17-46-44',display_fn_name='DecodedEpochSlices',epochs='ripple',user_annotation='selections') as ctx:
            user_annotations[ctx + Ctx(decoder='long_LR')] = [[1128.687, 1129.087], [1434.460, 1434.643]]
            user_annotations[ctx + Ctx(decoder='long_RL')] = [[1798.653, 1798.915]]
            user_annotations[ctx + Ctx(decoder='short_LR')] = [[419.116, 419.338], [538.868, 539.061], [973.037, 973.303], [1100.022, 1100.352], [1182.640, 1182.866]]
            user_annotations[ctx + Ctx(decoder='short_RL')] = [[403.834, 404.235], [449.090, 449.284], [466.119, 466.660], [702.706, 702.952], [1413.042, 1413.420]]


        # ==================================================================================================================== #
        #MARK '11-02_19-28-0'                                                                                                
        # ==================================================================================================================== #
        with Ctx(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_19-28-0',display_fn_name='DecodedEpochSlices',user_annotation='selections') as ctx:
            with (ctx + IdentifyingContext(epochs='replays')) as ctx:
                user_annotations[ctx + Ctx(decoder='long_results_obj')] = [2, 6]
                user_annotations[ctx + Ctx(decoder='short_results_obj')] = [2, 5, 9, 10]
            with (ctx + IdentifyingContext(epochs='ripple')) as ctx:
                user_annotations[ctx + Ctx(decoder='long_LR')] = [[208.356, 208.523], [954.574, 954.679]]
                user_annotations[ctx + Ctx(decoder='long_RL')] = [[224.037, 224.312]]
                user_annotations[ctx + Ctx(decoder='short_LR')] = [[145.776, 146.022], [198.220, 198.582], [208.356, 208.523], [220.041, 220.259], [511.570, 511.874], [865.238, 865.373]]
                user_annotations[ctx + Ctx(decoder='short_RL')] = [[191.817, 192.100], [323.147, 323.297]]
                
        # ==================================================================================================================== #
        #MARK '11-03_12-3-25'                                                                                                
        # ==================================================================================================================== #
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_12-3-25',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = []
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_12-3-25',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = [3, 4, 5]
        with Ctx(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_12-3-25',display_fn_name='DecodedEpochSlices',epochs='ripple',user_annotation='selections') as ctx:
            user_annotations[ctx + Ctx(decoder='long_LR')] = []
            user_annotations[ctx + Ctx(decoder='long_RL')] = []
            user_annotations[ctx + Ctx(decoder='short_LR')] = [[650.569, 650.759]]
            user_annotations[ctx + Ctx(decoder='short_RL')] = []

        ## ==================================================================================================================== #
        #MARK fet11-01_12-58-54                                                                                                
        # ==================================================================================================================== #
        with IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-01_12-58-54',display_fn_name='DecodedEpochSlices',user_annotation='selections') as ctx:
            with (ctx + IdentifyingContext(epochs='replays')) as ctx:
                user_annotations[ctx + Ctx(decoder='long_results_obj')] = [19, 23, 26, 29, 44, 57, 64, 83, 90, 92, 110, 123, 125, 126, 131]
                user_annotations[ctx + Ctx(decoder='short_results_obj')] = [5, 10, 19, 23, 24, 26, 31, 35, 36, 39, 44, 48, 57, 61, 64, 65, 71, 73, 77, 83, 89, 92, 93, 94, 96, 97, 98, 100, 102, 108, 111, 113, 116, 117, 118, 123, 124, 125, 126, 131]
            with (ctx + IdentifyingContext(epochs='ripple')) as ctx:
                user_annotations[ctx + Ctx(decoder='long_LR')] = [[135.957, 136.408], [403.414, 403.818], [765.576, 765.899], [767.898, 768.227], [961.169, 961.534], [1032.318, 1032.708], [1678.770, 1679.169], [1698.742, 1699.104], [1774.204, 1774.536], [1973.078, 1973.381], [2192.360, 2192.729], [2202.593, 2202.739], [2299.280, 2299.576], [2767.283, 2767.443], [2984.596, 2984.844]]
                user_annotations[ctx + Ctx(decoder='long_RL')] = [[127.407, 127.814], [583.209, 583.421], [710.132, 710.272], [756.700, 756.870], [776.909, 777.358], [793.618, 793.836], [840.625, 840.887], [873.844, 874.247], [906.872, 907.088], [913.670, 913.997], [960.367, 960.876], [974.823, 974.975], [985.189, 985.421], [996.211, 996.545], [1001.143, 1001.491], [1010.452, 1010.973], [1015.451, 1015.784], [1019.475, 1019.786], [1092.514, 1092.913], [1194.214, 1194.639], [1207.971, 1208.522], [1231.077, 1231.299], [1410.845, 1411.402], [1416.479, 1416.751], [1666.657, 1667.047], [1845.977, 1846.337], [1973.078, 1973.381], [2096.822, 2097.041], [2188.747, 2189.144], [2390.000, 2390.216], [2882.524, 2882.883]]
                user_annotations[ctx + Ctx(decoder='short_LR')] = [[72.773, 72.959], [248.042, 248.455], [459.258, 459.493], [515.191, 515.382], [583.209, 583.421], [641.164, 641.521], [974.823, 974.975], [1010.452, 1010.973], [1032.318, 1032.708], [1034.151, 1034.318], [1045.985, 1046.297], [1085.149, 1085.547], [1173.110, 1173.518], [1701.600, 1701.786], [1801.789, 1801.895], [1898.371, 1898.735], [1923.863, 1924.232], [2007.016, 2007.254], [2112.551, 2112.728], [2169.387, 2169.592], [2188.747, 2189.144], [2193.303, 2193.589], [2585.775, 2585.850], [2804.318, 2804.726], [2808.438, 2808.732]]
                user_annotations[ctx + Ctx(decoder='short_RL')] = [[514.232, 514.534], [583.209, 583.421], [721.530, 721.862], [776.909, 777.358], [961.169, 961.534], [996.211, 996.545], [1139.799, 1140.052], [1319.801, 1319.934], [1666.657, 1667.047], [1671.685, 1671.904], [1715.347, 1715.616], [1732.210, 1732.584], [1755.430, 1755.648], [1761.609, 1761.856], [1836.915, 1837.300], [1845.977, 1846.337], [1886.594, 1886.825], [1979.255, 1979.408], [2043.567, 2043.787], [2096.822, 2097.041], [2155.092, 2155.453], [2249.904, 2250.456], [2269.116, 2269.193], [2281.042, 2281.254], [2299.280, 2299.576], [2301.648, 2302.190], [2393.337, 2393.676], [2767.283, 2767.443], [2907.136, 2907.351], [2922.757, 2922.924], [2947.266, 2947.608], [2953.522, 2953.791]]         
        # IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43')
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = [3, 13, 16, 18, 19, 20, 23, 24, 27, 28, 36, 38, 40, 43, 44, 47, 48, 52, 55, 64, 65]
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = [3, 10, 13, 16, 18, 19, 24, 27, 28, 36, 40, 43, 44, 47, 48, 50, 55, 60, 64, 65]
        # 2023-07-19 Annotations:
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = [3, 5, 12, 18, 23, 26, 28, 30, 32, 33, 35, 37, 44, 59, 61, 64, 66, 70, 71, 74, 76, 79, 84, 85, 97, 99]
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = [5, 6, 8, 18, 23, 28, 29, 30, 32, 40, 44, 59, 61, 64, 66, 70, 71, 73, 74, 79, 80, 81, 84, 85, 88, 97, 99]
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',epochs='ripple',decoder='long_LR',user_annotation='selections')] = [[132.511, 132.791], [149.959, 150.254], [1186.9, 1187], [1284.18, 1284.29], [1302.65, 1302.8], [1316.06, 1316.27], [1693.34, 1693.48], [1725.28, 1725.6]]
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',epochs='ripple',decoder='long_RL',user_annotation='selections')] = [[149.959, 150.254], [307.08, 307.194], [1332.28, 1332.39]]
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',epochs='ripple',decoder='short_LR',user_annotation='selections')] = [[132.511, 132.791], [571.304, 571.385], [1284.18, 1284.29], [1302.65, 1302.8], [1316.06, 1316.27], [1699.23, 1699.36]]
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',epochs='ripple',decoder='short_RL',user_annotation='selections')] = [[105.4, 105.563], [1302.65, 1302.8], [1332.28, 1332.39], [1450.89, 1451.02]]

        with IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',user_annotation='selections') as ctx:
            with (ctx + IdentifyingContext(epochs='replays')) as ctx:
                user_annotations[ctx + Ctx(decoder='long_results_obj')] = [3, 5, 12, 18, 23, 26, 28, 30, 32, 33, 35, 37, 44, 59, 61, 64, 66, 70, 71, 74, 76, 79, 84, 85, 97, 99]
                user_annotations[ctx + Ctx(decoder='short_results_obj')] = [5, 6, 8, 18, 23, 28, 29, 30, 32, 40, 44, 59, 61, 64, 66, 70, 71, 73, 74, 79, 80, 81, 84, 85, 88, 97, 99]

            with (ctx + IdentifyingContext(epochs='ripple')) as ctx: 
                user_annotations[ctx + Ctx(decoder='long_LR')] = [[132.511, 132.791], [149.959, 150.254], [191.609, 191.949], [251.417, 251.812], [624.226, 624.499], [637.785, 638.182], [745.811, 746.097], [783.916, 784.032], [826.546, 826.823], [1085.080, 1085.184], [1136.192, 1136.453], [1186.899, 1186.997], [1244.038, 1244.176], [1284.181, 1284.287], [1302.651, 1302.801], [1316.056, 1316.270], [1693.342, 1693.482], [1725.279, 1725.595]]
                user_annotations[ctx + Ctx(decoder='long_RL')] = [[149.959, 150.254], [307.080, 307.194], [1332.283, 1332.395], [1705.053, 1705.141]]
                user_annotations[ctx + Ctx(decoder='short_LR')] = [[132.511, 132.791], [438.267, 438.448], [571.304, 571.385], [743.517, 743.653], [808.137, 808.222], [1117.650, 1118.019], [1252.562, 1252.739], [1262.523, 1262.926], [1284.181, 1284.287], [1302.651, 1302.801], [1316.056, 1316.270], [1317.977, 1318.181], [1348.890, 1349.264], [1351.622, 1351.704], [1440.852, 1441.328], [1699.225, 1699.357], [1707.712, 1707.919], [1716.016, 1716.168], [1731.111, 1731.288]]
                user_annotations[ctx + Ctx(decoder='short_RL')] = [[105.400, 105.563], [534.584, 534.939], [564.149, 564.440], [637.785, 638.182], [670.216, 670.418], [675.972, 676.153], [689.092, 689.241], [760.981, 761.121], [993.868, 994.185], [1131.640, 1131.887], [1161.001, 1161.274], [1188.466, 1188.592], [1302.651, 1302.801], [1332.283, 1332.395], [1450.894, 1451.024], [1673.437, 1673.920], [1705.053, 1705.141]]   


        # with Ctx(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',epochs='ripple',user_annotation='selections') as ctx:
        #     user_annotations[ctx + Ctx(decoder='long_LR')] = [[149.959, 150.254], [808.799, 808.948], [993.868, 994.185]]
        #     user_annotations[ctx + Ctx(decoder='long_RL')] = [[251.417, 251.812], [624.226, 624.499], [637.785, 638.182], [783.916, 784.032], [1085.080, 1085.184], [1136.192, 1136.453], [1693.342, 1693.482], [1725.279, 1725.595]]
        #     user_annotations[ctx + Ctx(decoder='short_LR')] = [[105.400, 105.563], [132.511, 132.791], [564.149, 564.440], [670.216, 670.418], [675.972, 676.153], [722.678, 722.933], [760.981, 761.121], [1131.640, 1131.887], [1161.001, 1161.274], [1191.563, 1191.864], [1302.651, 1302.801], [1332.283, 1332.395], [1450.894, 1451.024], [1705.053, 1705.141], [1707.712, 1707.919]]
        #     user_annotations[ctx + Ctx(decoder='short_RL')] = [[154.499, 154.853], [191.609, 191.949], [599.710, 599.905], [745.811, 746.097], [1117.650, 1118.019], [1233.968, 1234.181], [1244.038, 1244.176], [1252.562, 1252.739], [1262.523, 1262.926], [1267.918, 1268.235], [1316.056, 1316.270], [1317.977, 1318.181], [1348.890, 1349.264], [1440.852, 1441.328], [1731.111, 1731.288]]
                        

        # with Ctx(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',epochs='ripple',user_annotation='selections') as ctx:
        #     user_annotations[ctx + Ctx(decoder='long_LR')] += [[993.868, 994.185]]
        #     user_annotations[ctx + Ctx(decoder='long_RL')] += [[473.423, 473.747], [624.226, 624.499], [637.785, 638.182], [1136.192, 1136.453], [1348.890, 1349.264], [1673.437, 1673.920], [1693.342, 1693.482]]
        #     user_annotations[ctx + Ctx(decoder='short_LR')] += [[534.584, 534.939], [564.149, 564.440], [760.981, 761.121], [1131.640, 1131.887], [1161.001, 1161.274], [1332.283, 1332.395], [1707.712, 1707.919]]
        #     user_annotations[ctx + Ctx(decoder='short_RL')] += [[438.267, 438.448], [637.785, 638.182], [1085.596, 1086.046], [1117.650, 1118.019], [1252.562, 1252.739], [1262.523, 1262.926], [1316.056, 1316.270], [1348.890, 1349.264], [1440.852, 1441.328], [1729.689, 1730.228], [1731.111, 1731.288]]
            
        with Ctx(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',epochs='ripple',user_annotation='selections') as ctx:
            user_annotations[ctx + Ctx(decoder='long_LR')] = [[105.400, 105.563], [132.511, 132.791], [149.959, 150.254], [154.499, 154.853], [191.609, 191.949], [251.417, 251.812], [438.267, 438.448], [473.423, 473.747], [534.584, 534.939], [564.149, 564.440], [599.710, 599.905], [624.226, 624.499], [637.785, 638.182], [670.216, 670.418], [675.972, 676.153], [722.678, 722.933], [745.811, 746.097], [760.981, 761.121], [783.916, 784.032], [808.799, 808.948], [826.546, 826.823], [993.868, 994.185], [1085.080, 1085.184], [1085.596, 1086.046], [1117.650, 1118.019], [1131.640, 1131.887], [1136.192, 1136.453], [1161.001, 1161.274], [1191.563, 1191.864], [1233.968, 1234.181], [1244.038, 1244.176], [1252.562, 1252.739], [1262.523, 1262.926], [1267.918, 1268.235], [1302.651, 1302.801], [1316.056, 1316.270], [1317.977, 1318.181], [1332.283, 1332.395], [1348.890, 1349.264], [1440.852, 1441.328], [1450.894, 1451.024], [1673.437, 1673.920], [1693.342, 1693.482], [1705.053, 1705.141], [1707.712, 1707.919], [1725.279, 1725.595], [1729.689, 1730.228], [1731.111, 1731.288]]
            user_annotations[ctx + Ctx(decoder='long_RL')] = [[105.400, 105.563], [132.511, 132.791], [149.959, 150.254], [154.499, 154.853], [191.609, 191.949], [251.417, 251.812], [438.267, 438.448], [473.423, 473.747], [534.584, 534.939], [564.149, 564.440], [599.710, 599.905], [624.226, 624.499], [637.785, 638.182], [670.216, 670.418], [675.972, 676.153], [722.678, 722.933], [745.811, 746.097], [760.981, 761.121], [783.916, 784.032], [808.799, 808.948], [993.868, 994.185], [1085.080, 1085.184], [1085.596, 1086.046], [1117.650, 1118.019], [1131.640, 1131.887], [1136.192, 1136.453], [1161.001, 1161.274], [1191.563, 1191.864], [1233.968, 1234.181], [1244.038, 1244.176], [1252.562, 1252.739], [1262.523, 1262.926], [1267.918, 1268.235], [1302.651, 1302.801], [1316.056, 1316.270], [1317.977, 1318.181], [1332.283, 1332.395], [1348.890, 1349.264], [1440.852, 1441.328], [1450.894, 1451.024], [1673.437, 1673.920], [1693.342, 1693.482], [1705.053, 1705.141], [1707.712, 1707.919], [1725.279, 1725.595], [1729.689, 1730.228], [1731.111, 1731.288]]
            user_annotations[ctx + Ctx(decoder='short_LR')] = [[105.400, 105.563], [132.511, 132.791], [149.959, 150.254], [154.499, 154.853], [191.609, 191.949], [251.417, 251.812], [438.267, 438.448], [473.423, 473.747], [534.584, 534.939], [564.149, 564.440], [599.710, 599.905], [624.226, 624.499], [637.785, 638.182], [670.216, 670.418], [675.972, 676.153], [722.678, 722.933], [745.811, 746.097], [760.981, 761.121], [783.916, 784.032], [808.799, 808.948], [933.699, 934.094], [993.868, 994.185], [1085.080, 1085.184], [1085.596, 1086.046], [1117.650, 1118.019], [1131.640, 1131.887], [1136.192, 1136.453], [1161.001, 1161.274], [1191.563, 1191.864], [1233.968, 1234.181], [1244.038, 1244.176], [1252.562, 1252.739], [1262.523, 1262.926], [1267.918, 1268.235], [1302.651, 1302.801], [1316.056, 1316.270], [1317.977, 1318.181], [1332.283, 1332.395], [1348.890, 1349.264], [1440.852, 1441.328], [1450.894, 1451.024], [1673.437, 1673.920], [1693.342, 1693.482], [1705.053, 1705.141], [1707.712, 1707.919], [1725.279, 1725.595], [1729.689, 1730.228], [1731.111, 1731.288]]
            user_annotations[ctx + Ctx(decoder='short_RL')] = [[105.400, 105.563], [132.511, 132.791], [149.959, 150.254], [154.499, 154.853], [191.609, 191.949], [251.417, 251.812], [438.267, 438.448], [473.423, 473.747], [534.584, 534.939], [564.149, 564.440], [599.710, 599.905], [624.226, 624.499], [637.785, 638.182], [670.216, 670.418], [675.972, 676.153], [722.678, 722.933], [745.811, 746.097], [760.981, 761.121], [783.916, 784.032], [808.799, 808.948], [993.868, 994.185], [1085.080, 1085.184], [1085.596, 1086.046], [1117.650, 1118.019], [1131.640, 1131.887], [1136.192, 1136.453], [1161.001, 1161.274], [1191.563, 1191.864], [1233.968, 1234.181], [1244.038, 1244.176], [1252.562, 1252.739], [1262.523, 1262.926], [1267.918, 1268.235], [1302.651, 1302.801], [1316.056, 1316.270], [1317.977, 1318.181], [1332.283, 1332.395], [1348.890, 1349.264], [1375.574, 1375.809], [1440.852, 1441.328], [1450.894, 1451.024], [1673.437, 1673.920], [1693.342, 1693.482], [1705.053, 1705.141], [1707.712, 1707.919], [1725.279, 1725.595], [1729.689, 1730.228], [1731.111, 1731.288]]

                
                        
        # IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_21-16-25')
        with Ctx(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_21-16-25',display_fn_name='DecodedEpochSlices',user_annotation='selections') as ctx:
            with (ctx + IdentifyingContext(epochs='replays')) as ctx:
                user_annotations[ctx + Ctx(decoder='long_results_obj')] = [2, 13, 18, 23, 25, 27, 32]
                user_annotations[ctx + Ctx(decoder='short_results_obj')] = [2, 8, 9, 13, 16, 18, 25, 27, 28, 32, 33]
            with (ctx + IdentifyingContext(epochs='ripple')) as ctx:
                user_annotations[ctx + Ctx(decoder='long_LR')] = [[785.738, 785.923]]
                user_annotations[ctx + Ctx(decoder='long_RL')] = [[427.461, 427.557]]
                user_annotations[ctx + Ctx(decoder='short_LR')] = [[833.339, 833.451]]
                user_annotations[ctx + Ctx(decoder='short_RL')] = [[491.798, 492.178], [940.016, 940.219]]


        # IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-09_22-24-40')
        with IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-09_22-24-40') as session_ctx:
            with (session_ctx + IdentifyingContext(display_fn_name='DecodedEpochSlices', user_annotation='selections')) as ctx:
                with (ctx + IdentifyingContext(epochs='replays')) as ctx:
                    user_annotations[ctx + Ctx(decoder='long_results_obj')] = [4, 22, 24, 28, 30, 38, 42, 50, 55, 60, 67, 70, 76, 83, 85, 100, 103, 107, 108, 113, 118, 121, 122, 131, 140, 142, 149, 153, 170, 171]
                    user_annotations[ctx + Ctx(decoder='short_results_obj')] = [2, 7, 11, 17, 20, 22, 30, 34, 38, 39, 41, 43, 47, 49, 55, 59, 60, 69, 70, 75, 77, 80, 83, 85, 86, 100, 107, 110, 113, 114, 115, 118, 120, 121, 122, 126, 130, 131, 138, 140, 142, 149, 157, 160, 168, 170]

                with (ctx + IdentifyingContext(epochs='ripple')) as ctx:
                    user_annotations[ctx + Ctx(decoder='long_LR')] = [[380.739, 380.865], [550.845, 551.034], [600.244, 600.768], [1431.7, 1431.87], [2121.38, 2121.72]]
                    user_annotations[ctx + Ctx(decoder='long_RL')] = [[1202.96, 1203.26], [1433.42, 1433.58], [1600.77, 1601.16], [1679.18, 1679.68]]
                    user_annotations[ctx + Ctx(decoder='short_LR')] = [[551.872, 552.328], [565.161, 565.417], [616.348, 616.665], [919.581, 919.692], [1149.57, 1149.8], [1167.82, 1168.17], [1384.71, 1385.01], [1424.02, 1424.22], [1446.52, 1446.65], [1538.1, 1538.48], [1690.72, 1690.82], [1820.96, 1821.29], [1979.72, 1979.86], [1995.48, 1995.95], [2121.38, 2121.72], [2267.05, 2267.41]]
                    user_annotations[ctx + Ctx(decoder='short_RL')] = [[373.508, 373.754], [391.895, 392.163], [600.244, 600.768], [1015.26, 1015.5], [1079.9, 1080.08], [1310.59, 1310.92], [1433.42, 1433.58], [1494.95, 1495.4], [1558.22, 1558.42], [1616.92, 1617.09], [1774.48, 1774.61], [1956.96, 1957.2], [2011.36, 2011.54], [2059.35, 2059.56], [2074.35, 2074.62], [2156.53, 2156.79], [2233.53, 2233.95], [2260.49, 2260.61], [2521.1, 2521.31]]


        # IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-12_16-53-46')
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-12_16-53-46',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = [3, 6, 15, 16, 18]
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-12_16-53-46',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = [3, 4, 6, 9, 13, 15, 16, 18]

        # IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-09_17-29-30')
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-09_17-29-30',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = [5, 6, 7, 14, 17, 19, 20, 24]
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-09_17-29-30',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = [5, 6, 19, 20]


        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-10_12-25-50',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = [0, 3, 4, 5, 9, 12]
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-10_12-25-50',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = [0, 3, 4, 6, 9, 12, 13]

        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-09_16-40-54',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = [3]
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-09_16-40-54',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = [3]

        # 2023-07-21T19-17-02
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-10_12-58-3',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = [3, 19]
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-10_12-58-3',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = [ 5, 16, 17, 19, 20]


        # 2006-4-17_12-52-15 _________________________________________________________________________________________________ #
        with Ctx(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-17_12-52-15',display_fn_name='DecodedEpochSlices',epochs='ripple',user_annotation='selections') as ctx:
            user_annotations[ctx + Ctx(decoder='long_LR')] = []
            user_annotations[ctx + Ctx(decoder='long_RL')] = [[545.997, 546.377]]
            user_annotations[ctx + Ctx(decoder='short_LR')] = []
            user_annotations[ctx + Ctx(decoder='short_RL')] = [[161.848, 162.029], [702.907, 703.008]]
        
        # 2006-4-16_18-47-52 _________________________________________________________________________________________________ #
        with Ctx(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-16_18-47-52',display_fn_name='DecodedEpochSlices',epochs='ripple',user_annotation='selections') as ctx:
            user_annotations[ctx + Ctx(decoder='long_LR')] = [[715.966, 716.140]]
            user_annotations[ctx + Ctx(decoder='long_RL')] = [[464.763, 464.957]]
            user_annotations[ctx + Ctx(decoder='short_LR')] = [[72.222, 72.320]]
            user_annotations[ctx + Ctx(decoder='short_RL')] = [[694.698, 695.182]]
            # user_annotations[ctx + Ctx(decoder='short_RL')] = []

        # 2006-4-25_13-20-55 _________________________________________________________________________________________________ #
        with Ctx(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-25_13-20-55',display_fn_name='DecodedEpochSlices',epochs='ripple',user_annotation='selections') as ctx:
            user_annotations[ctx + Ctx(decoder='long_LR')] = [[247.539, 247.734], [458.548, 458.639]]
            user_annotations[ctx + Ctx(decoder='long_RL')] = [[51.287, 51.457], [63.019, 63.232], [541.021, 541.161]]
            user_annotations[ctx + Ctx(decoder='short_LR')] = []
            user_annotations[ctx + Ctx(decoder='short_RL')] = []
            
        # 2006-4-28_12-38-13 _________________________________________________________________________________________________ #
        with Ctx(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-28_12-38-13',display_fn_name='DecodedEpochSlices',epochs='ripple',user_annotation='selections') as ctx:
            user_annotations[ctx + Ctx(decoder='long_LR')] = [[304.098, 304.301]]
            user_annotations[ctx + Ctx(decoder='long_RL')] = [[98.551, 98.723]]
            user_annotations[ctx + Ctx(decoder='short_LR')] = []
            user_annotations[ctx + Ctx(decoder='short_RL')] = []


        # Process raw annotations with the helper function
        for context, sequences in user_annotations.items():
            user_annotations[context] = numpyify_array(sequences)


        return user_annotations
    
    @classmethod
    def has_user_annotation(cls, test_context):
        user_anootations = cls.get_user_annotations()
        was_annotation_found: bool = False
        # try to find a matching user_annotation for the final_context_L
        for a_ctx, selections_array in user_anootations.items():
            an_item_diff = a_ctx.diff(test_context)
            if an_item_diff == {('user_annotation', 'selections')}:
                was_annotation_found = True
                break # done looking
        return was_annotation_found


    # def add_user_annotation(self, context: IdentifyingContext, value):
    @classmethod
    def get_hardcoded_specific_session_cell_exclusivity_annotations_dict(cls) -> dict:
        """ hand-labeled by pho on 2023-10-04 """
        session_cell_exclusivity_annotations: Dict[IdentifyingContext, SessionCellExclusivityRecord] = {
        IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15'):
            SessionCellExclusivityRecord(LxC=[109],
                LpC=[],
                SpC=[67, 52],
                SxC=[23,4,58]),
        IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43'):
            SessionCellExclusivityRecord(LxC=[3, 29, 103],
                LpC=[],
                SpC=[33, 35, 58],
                SxC=[55]),
        IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-12_15-55-31'):
            SessionCellExclusivityRecord(LxC=[],
                LpC=[2, 3, 34],
                SpC=[31, 33, 53],
                SxC=[30]),
        IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19'):
            SessionCellExclusivityRecord(LxC=[],
                LpC=[],
                SpC=[18, 65],
                SxC=[3, 19]),
        IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_21-16-25'):
            SessionCellExclusivityRecord(LxC=[90],
                LpC=[23, 73],
                SpC=[4, 16, 82],
                SxC=[8]),
        IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-09_22-24-40'):
            SessionCellExclusivityRecord(LxC=[91, 95],
                LpC=[15, 16, 32],
                SpC=[11],
                SxC=[]),
        IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-12_16-53-46'):
            SessionCellExclusivityRecord(LxC=[38, 59],
                LpC=[51, 60],
                SpC=[7],
                SxC=[8]),
        ## Break
        IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-09_17-29-30'):
            SessionCellExclusivityRecord(LxC=[],
                LpC=[4, 6, 17, 28, 12],
                SpC=[21, 31],
                SxC=[41]),
        IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-10_12-25-50'):
            SessionCellExclusivityRecord(LxC=[23],
                LpC=[19],
                SpC=[36],
                SxC=[29]),
        IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-09_16-40-54'):
            SessionCellExclusivityRecord(LxC=[25],
                LpC=[12, 14, 17],
                SpC=[30],
                SxC=[]),
        IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-10_12-58-3'):
            SessionCellExclusivityRecord(LxC=[14, 30, 32],
                LpC=[40],
                SpC=[42],
                SxC=[]),
        IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_17-46-44'):
            SessionCellExclusivityRecord(LxC=[8, 27],
                LpC=[10],
                SpC=[18,20,40],
                SxC=[17]),
        IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_19-28-0'):
            SessionCellExclusivityRecord(LxC=[27],
                LpC=[8, 13],
                SpC=[],
                SxC=[]),
        IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_12-3-25'):
            SessionCellExclusivityRecord(LxC=[],
                LpC=[],
                SpC=[13,22,28],
                SxC=[]),
        IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-01_12-58-54'):
            SessionCellExclusivityRecord(LxC=[],
                LpC=[6, 10, 16, 19],
                SpC=[24],
                SxC=[]),

        }
        return session_cell_exclusivity_annotations


            
    @classmethod
    def get_hardcoded_specific_session_override_dict(cls) -> dict:
        """ ## Create a dictionary of overrides that have been specified manually for a given session:
            # Used in `build_lap_only_short_long_bin_aligned_computation_configs`
            History: Extracted from `neuropy.core.session.Formats.Specific.KDibaOldDataSessionFormat` 
        
        Change Log:
            2025-01-14 - added ['unit_grid_bin_bounds', 'cm_grid_bin_bounds', 'real_unit_x_grid_bin_bounds', 'real_cm_x_grid_bin_bounds'] annotations to `UserAnnotationsManager.get_hardcoded_specific_session_override_dict(...)`, removed ambiguous 'grid_bin_bounds' to purposefully break dependencies
        
        Usage:
            ## Get specific grid_bin_bounds overrides from the `UserAnnotationsManager._specific_session_override_dict`
            override_dict = UserAnnotationsManager.get_hardcoded_specific_session_override_dict().get(sess.get_context(), {})
            if override_dict.get('grid_bin_bounds', None) is not None:
                grid_bin_bounds = override_dict['grid_bin_bounds']
            else:
                # no overrides present
                pos_df = sess.position.to_dataframe().copy()
                if not 'lap' in pos_df.columns:
                    pos_df = sess.compute_laps_position_df() # compute the lap column as needed.
                laps_pos_df = pos_df[pos_df.lap.notnull()] # get only the positions that belong to a lap
                laps_only_grid_bin_bounds = PlacefieldComputationParameters.compute_grid_bin_bounds(laps_pos_df.x.to_numpy(), laps_pos_df.y.to_numpy()) # compute the grid_bin_bounds for these positions only during the laps. This means any positions outside of this will be excluded!
                print(f'\tlaps_only_grid_bin_bounds: {laps_only_grid_bin_bounds}')
                grid_bin_bounds = laps_only_grid_bin_bounds
                # ## Determine the grid_bin_bounds from the long session:
                # grid_bin_bounds = PlacefieldComputationParameters.compute_grid_bin_bounds(sess.position.x, sess.position.y) # ((22.736279243974774, 261.696733348342), (125.5644705153173, 151.21507349463707))
                # # refined_grid_bin_bounds = ((24.12, 259.80), (130.00, 150.09))
                # DO INTERACTIVE MODE:
                # grid_bin_bounds = interactive_select_grid_bin_bounds_2D(curr_active_pipeline, epoch_name='maze', should_block_for_input=True)
                # print(f'grid_bin_bounds: {grid_bin_bounds}')
                # print(f"Add this to `specific_session_override_dict`:\n\n{curr_active_pipeline.get_session_context().get_initialization_code_string()}:dict(grid_bin_bounds=({(grid_bin_bounds[0], grid_bin_bounds[1]), (grid_bin_bounds[2], grid_bin_bounds[3])})),\n")

        """
        user_annotations = {}            
        # user_annotations.update({ 
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15'):{'grid_bin_bounds':((29.16, 261.70), (130.23, 150.99))},
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19'):{'grid_bin_bounds':((22.397021260868584, 245.6584673739576), (133.66465594522782, 155.97244934208123))},
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_21-16-25'):dict(grid_bin_bounds=(((17.01858788173554, 250.2171441367766), (135.66814125966783, 154.75073313142283)))),
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-09_22-24-40'):{'grid_bin_bounds':(((29.088604852961407, 251.70402561515647), (138.496638485457, 154.30675703402517)))},
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-09_17-29-30'):{'grid_bin_bounds':(((29.16, 261.7), (133.87292045454544, 150.19888636363635)))},
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-10_12-25-50'):{'grid_bin_bounds':((25.5637332724328, 257.964172947664), (89.1844223602494, 131.92462510535915))},
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-09_16-40-54'):{'grid_bin_bounds':(((19.639345624112345, 248.63934562411234), (134.21607306829767, 154.57926689187622)))},
        #     # IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43'):{'grid_bin_bounds':((28.54313873072426, 255.54313873072425), (80.0, 151.0))}, # (-56.2405385510412, -12.237798967230454)
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43'):dict(grid_bin_bounds=(((36.58620390950715, 248.91627658974846), (132.81136363636367, 149.2840909090909)))),
        #     # IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15'):{'grid_bin_bounds':((25.5637332724328, 257.964172947664), (89.1844223602494, 131.92462510535915))},
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_3-23-37'):{'grid_bin_bounds':(((29.64642522460817, 257.8732552112081), (106.68603845428224, 146.71219371189815)))},
        #     # IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-12_15-55-31'):{'grid_bin_bounds':(((36.47611374385336, 246.658598426423), (134.75608863422366, 149.10512838805013)))},
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-13_14-42-6'):{'grid_bin_bounds':(((34.889907585004366, 250.88049171752402), (131.38802948402946, 148.80548955773958)))},
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_15-46-47'):{'grid_bin_bounds':(((37.58127153781621, 248.7032779553949), (133.5550653393467, 147.88514770982718)))},
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-17_12-33-47'):{'grid_bin_bounds':(((26.23480758754316, 249.30607830191923), (130.58181353748455, 153.36300919999059)))},
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-18_13-6-1'):{'grid_bin_bounds':(((31.470464455344967, 252.05028043482017), (128.05945067500747, 150.3229156741395)))},
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-19_13-34-40'):{'grid_bin_bounds':(((29.637787747400818, 244.6377877474008), (138.47834488369824, 155.0993015545914)))},
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-27_14-43-12'):{'grid_bin_bounds':(((27.16098236570231, 249.70986567911666), (106.81005068995495, 118.74413456592755)))},
        #     # IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-10_12-58  -3'):{'grid_bin_bounds':(((28.84138997640293, 259.56043988873074), (101.90256273413083, 118.33845994931318)))},
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-11_12-48-38'):{'grid_bin_bounds':(((21.01014932647431, 250.0101493264743), (92.34934413366932, 128.1552287735411)))},
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-11_16-2-46'):{'grid_bin_bounds':(((17.270839996578303, 259.97986762679335), (94.26725170377283, 131.3621243061284)))},
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-12_15-25-59'):{'grid_bin_bounds':(((30.511181558838498, 247.5111815588389), (106.97411662767412, 146.12444016982818)))},
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-16_14-49-24'):{'grid_bin_bounds':(((30.473731136762368, 250.59478046470133), (105.10585244511995, 149.36442051808177)))},
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-16_18-47-52'):{'grid_bin_bounds':(((27.439671363238585, 252.43967136323857), (106.37372678405141, 149.37372678405143)))},
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-17_12-52-15'):{'grid_bin_bounds':(((25.118453388111003, 253.3770388211908), (106.67602982073078, 145.67602982073078)))},
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-19_13-50-7'):{'grid_bin_bounds':(((22.47237613669028, 247.4723761366903), (109.8597911774777, 148.96242871522395)))},
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-19_16-37-40'):{'grid_bin_bounds':(((27.10059856429566, 249.16997904433555), (104.99819196992492, 148.0743732909197)))},
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-21_11-19-2'):{'grid_bin_bounds':(((19.0172498755827, 255.42277198494864), (110.04725120825609, 146.9523233129975)))},
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-25_13-20-55'):{'grid_bin_bounds':(((12.844282158261015, 249.81408485606906), (107.18107171696062, 147.5733884981106)))},
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-26_13-51-50'):{'grid_bin_bounds':(((29.04362374788327, 248.04362374788326), (104.87398380095135, 145.87398380095135)))},
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-28_12-38-13'):{'grid_bin_bounds':(((14.219834349211556, 256.8892365192059), (104.62582591329034, 144.76901436952045)))},
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_17-46-44'):dict(grid_bin_bounds=(((26.927879930920472, 253.7869451377655), (129.2279041328145, 152.59317191760715)))),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_19-28-0'):dict(grid_bin_bounds=(((20.551685242617875, 249.52142297024744), (136.6282885482392, 154.9308054334688)))),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_12-3-25'):dict(grid_bin_bounds=(((22.2851382680749, 246.39985985110218), (133.85711719213543, 152.81579979839964)))),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-01_12-58-54'):dict(grid_bin_bounds=(((24.27436551166163, 254.60064907635376), (136.60434348821698, 150.5038133052293)))),
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-12_15-55-31'):dict(grid_bin_bounds=(((28.300282316379977, 259.7976187852487), (128.30369397123394, 154.72988093974095)))),
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-12_16-53-46'):dict(grid_bin_bounds=(((24.481516142738176, 255.4815161427382), (132.49260896751392, 155.30747604466447)))),
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-07_11-26-53'):dict(grid_bin_bounds=(((38.73738238974907, 252.7760510005677), (132.90403388034895, 148.5092041989809)))),
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-13_15-22-3'):dict(grid_bin_bounds=(((30.65617684737977, 242.10210639375669), (135.26433229328896, 154.7431209356581)))),
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-18_15-23-32'):dict(grid_bin_bounds=(((22.21801384517548, 249.21801384517548), (96.86784196156162, 123.98778194291367)))),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_21-26-8'):dict(grid_bin_bounds=(((20.72906822314644, 251.8461197472293), (135.47600437022294, 153.41070963308343)))),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-05_19-26-43'):dict(grid_bin_bounds=(((22.368808752432248, 251.37564395184629), (137.03177602287607, 158.05294971766054)))),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_11-43-50'):dict(grid_bin_bounds=(((24.69593655030406, 253.64858157619915), (136.41037216273168, 154.5621182921704)))),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_12-15-3'):dict(grid_bin_bounds=(((23.604005859617253, 246.60400585961725), (140.24525850461768, 156.87513105457236)))),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_21-17-16'):dict(grid_bin_bounds=(((26.531441449778306, 250.85301772871418), (138.8068762398286, 154.96868817726707)))),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_22-4-5'):dict(grid_bin_bounds=(((23.443356646345123, 251.83482088135696), (134.31041516724372, 155.28217042078484)))),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-03_20-28-3'):dict(grid_bin_bounds=(((22.32225341225395, 251.16261749302362), (134.7141935930816, 152.90682184511172)))),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-04_21-20-3'):dict(grid_bin_bounds=(((23.682015380447947, 250.68201538044795), (132.83596766730062, 151.01445456436113)))),
        #     ## 2023-07-20 New, will replace existing:
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-07_11-26-53'):dict(grid_bin_bounds=(((39.69907686853041, 254.50216929581626), (132.04374884996278, 148.989363285708)))),
        #     # IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15'):dict(grid_bin_bounds=(((25.5637332724328, 257.964172947664), (89.1844223602494, 131.92462510535915)))),
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43'):dict(grid_bin_bounds=(((36.58620390950715, 248.91627658974846), (132.81136363636367, 149.2840909090909)))),
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_3-23-37'):dict(grid_bin_bounds=(((31.189597143056417, 248.34843718632368), (129.87370603299934, 152.79894671566146)))),
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-12_15-55-31'):dict(grid_bin_bounds=(((28.300282316379977, 259.30028231638), (128.30369397123394, 154.72988093974095)))),
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-13_14-42-6'):dict(grid_bin_bounds=(((34.889907585004366, 249.88990758500438), (131.38802948402946, 148.38802948402946)))),
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19'):dict(grid_bin_bounds=(((22.397021260868584, 245.3970212608686), (133.66465594522782, 155.97244934208123)))),
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_15-46-47'):dict(grid_bin_bounds=(((24.87832507963516, 243.19318180072503), (138.17144977828153, 154.70022850038026)))),
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_21-16-25'):dict(grid_bin_bounds=(((24.71824744583462, 248.6393456241123), (136.77104473778593, 152.85274652666337)))),
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-09_22-24-40'):dict(grid_bin_bounds=(((29.088604852961407, 251.70402561515647), (138.496638485457, 153.496638485457)))),
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-12_16-53-46'):dict(grid_bin_bounds=(((24.481516142738176, 255.4815161427382), (132.49260896751392, 155.30747604466447)))),
        #     IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-13_15-22-3'):dict(grid_bin_bounds=(((30.65617684737977, 241.65617684737975), (135.26433229328896, 154.26433229328896)))),
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-09_17-29-30'):dict(grid_bin_bounds=(((28.54313873072426, 255.54313873072425), (-55.2405385510412, -12.237798967230454)))),
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-10_12-25-50'):dict(grid_bin_bounds=(((25.5637332724328, 257.964172947664), (89.1844223602494, 131.92462510535915)))),
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-17_12-33-47'):dict(grid_bin_bounds=(((28.818747438744666, 252.93642303882393), (105.90899758151346, 119.13817286828353)))),
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-18_13-6-1'):dict(grid_bin_bounds=(((29.717076500273222, 259.56043988873074), (95.7227315733896, 133.44723972697744)))),
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-18_15-23-32'):dict(grid_bin_bounds=(((26.23603700440421, 249.21801384517548), (92.8248259903438, 133.94007118631126)))),
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-19_13-34-40'):dict(grid_bin_bounds=(((24.30309074447524, 252.62022717482705), (88.84795233953739, 129.8246799924719)))),
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-27_14-43-12'):dict(grid_bin_bounds=(((18.92582426040771, 259.0914978964906), (92.73825146448141, 127.71789984751534)))),
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-09_16-40-54'):dict(grid_bin_bounds=(((29.64642522460817, 257.8732552112081), (106.68603845428224, 146.71219371189815)))),
        #     # IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-10_12-58-3'):dict(grid_bin_bounds=(((28.84138997640293, 259.56043988873074), (106.30424414303856, 118.90256273413083)))),
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-11_12-48-38'):dict(grid_bin_bounds=(((31.074573205168065, 250.46915764230738), (104.49699700791726, 143.3410531370672)))),
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-11_16-2-46'):dict(grid_bin_bounds=(((24.297550846655597, 254.08440958433795), (107.3563450759142, 150.37117694134784)))),
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-12_15-25-59'):dict(grid_bin_bounds=(((30.511181558838498, 247.5111815588389), (106.97411662767412, 145.9741166276741)))),
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-16_14-49-24'):dict(grid_bin_bounds=(((24.981275039670876, 249.50933410068018), (107.26469870056884, 148.83960860385804)))),
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-16_18-47-52'):dict(grid_bin_bounds=(((27.905254233199088, 250.7946946337514), (105.45717926004222, 146.30022614559135)))),
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-17_12-52-15'):dict(grid_bin_bounds=(((18.204825719140114, 251.30717868551005), (102.22431800161995, 148.39669517071601)))),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_17-46-44'):dict(grid_bin_bounds=(((26.927879930920472, 253.7869451377655), (129.2279041328145, 152.2279041328145)))),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_19-28-0'):dict(grid_bin_bounds=(((20.551685242617875, 249.52142297024744), (136.6282885482392, 154.9308054334688)))),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_12-3-25'):dict(grid_bin_bounds=(((22.2851382680749, 246.39985985110218), (133.85711719213543, 152.81579979839964)))),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_21-26-8'):dict(grid_bin_bounds=(((25.037045888933548, 251.8461197472293), (134.08933682327196, 153.70077784443535)))),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-05_19-26-43'):dict(grid_bin_bounds=(((25.586078838864836, 251.7891334064322), (135.5158259968098, 161.196019438371)))),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_11-43-50'):dict(grid_bin_bounds=(((24.69593655030406, 252.54415392264917), (135.0071025581097, 155.43479839454722)))),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_12-15-3'):dict(grid_bin_bounds=(((22.348488031027813, 246.60400585961725), (139.61986158820912, 157.67819754950602)))),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_21-17-16'):dict(grid_bin_bounds=(((27.11122010574816, 251.94824027772256), (141.2080597276766, 157.07149975242388)))),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_22-4-5'):dict(grid_bin_bounds=(((21.369069920503218, 250.59459989762465), (135.63605454614645, 156.15891604175224)))),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-01_12-58-54'):dict(grid_bin_bounds=(((22.403791476255435, 255.28121598502332), (135.43617904962073, 153.6679723832235)))),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-03_20-28-3'):dict(grid_bin_bounds=(((22.32225341225395, 251.16261749302362), (129.90833607359593, 156.190369383283)))),
        #     IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-04_21-20-3'):dict(grid_bin_bounds=(((23.682015380447947, 250.68201538044795), (132.9587491923089, 150.81663988518108)))),
        #     IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-10_12-58-3'):dict(grid_bin_bounds=(((30.511181558838498, 247.5111815588389), (106.97411662767412, 147.52430924258078)))),
        # })
    
        
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',user_annotation='grid_bin_bounds')] = ((37.0773897438341, 250.69004399129707), (138.16397564990257, 146.1197529956474))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (138.16397564990257, 146.1197529956474))))
        # # ==================================================================================================================== #
        # # Generated by `EXTERNAL\TESTING\testing_notebooks\PhoMatFileBrowsingTesting.ipynb` on 2024-04-10                      #
        # # ==================================================================================================================== #
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-07_11-26-53')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (135.23924311831908, 144.1984518410047))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (137.925447118083, 145.16448776601297))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (138.16397564990257, 146.1197529956474))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_3-23-37')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (139.34507862499134, 147.58755064986212))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-12_15-55-31')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (137.97626338793503, 146.00371440346137))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-13_14-42-6')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (138.2791990765373, 145.4465660546858))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (138.39266723911976, 146.9470603477007))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_15-46-47')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (142.39397650284766, 151.15367504603452))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_21-16-25')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (139.89578722770338, 148.51861548115295))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-09_22-24-40')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (139.3053644252862, 147.99662782360443))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-12_16-53-46')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (139.91706986262813, 148.7452035506836))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-13_15-22-3')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (141.07167669307665, 149.9682455260008))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_17-46-44')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (136.10095530133273, 144.59252270314897))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_19-28-0')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (141.44749773951332, 147.73754766678462))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_12-3-25')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (138.9314777686048, 146.63678892951214))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_21-26-8')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (138.0941872269056, 144.73231230966147))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-05_19-26-43')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (142.68438010327935, 149.1812684821468))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_11-43-50')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (141.83695987994884, 148.19261155492785))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_12-15-3')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (143.81427373438675, 151.01734563269633))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_21-17-16')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (143.1080902149446, 149.67559694575627))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_22-4-5')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (141.6251048241162, 147.8395197952068))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-01_12-58-54')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (140.14211943328775, 147.2745729796531))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-03_20-28-3')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (137.74109546718452, 144.66169395771726))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-04_21-20-3')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (138.05172078553375, 143.70812931964014))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-09_17-29-30')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (108.13933162527657, 115.13913718623321))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-10_12-25-50')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (107.8177789584226, 113.7570079192343))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-10_21-2-40')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (108.18519657066037, 113.91231021144307))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-11_15-16-59')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (108.52716865032582, 113.61793633102158))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-12_14-39-31')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (109.19374317979644, 114.35521596716852))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-12_17-53-55')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (107.82954763216065, 113.6273663796197))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-16_15-12-23')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (108.31665318697634, 113.76094640105372))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-17_12-33-47')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (109.64675309404181, 115.02034120144286))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-18_13-6-1')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (108.9445597322445, 114.6716733730272))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-18_15-23-32')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (109.67092311452707, 114.54957547519382))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-19_13-34-40')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (106.80073123839011, 112.7399601992018))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-19_16-48-9')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (108.55947898953707, 114.56941305702512))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-21_10-24-35')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (108.22908621020602, 114.94607134445735))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-25_14-28-51')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (109.0566791531139, 115.63225407401256))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-25_17-17-6')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (109.29954776242832, 115.16807161656368))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-26_13-22-13')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (107.60152468408094, 114.81344556506657))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-27_14-43-12')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (107.22404579212474, 114.08244113972872))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-28_16-48-29')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (108.2501053175898, 113.9772189583725))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-09_16-40-54')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (122.96634289724359, 130.46108420493456))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-10_12-58-3')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (122.84017431563773, 129.69856966324173))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-10_19-11-57')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (122.9536718458597, 129.52924676675838))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-11_12-48-38')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (125.03673561741104, 131.04666968489906))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-11_16-2-46')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (124.05348636842058, 129.56848468917428))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-12_15-25-59')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (126.17725557343053, 131.1266130407736))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-16_14-49-24')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (123.90636700893516, 129.5627755430415))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-16_18-47-52')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (123.63448954400148, 128.79596233137354))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-17_12-52-15')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (124.14624025653195, 129.44912325725667))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-18_13-28-57')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (122.92990621750084, 128.7984300716362))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-18_15-38-2')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (124.22934368316666, 129.46152157721505))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-19_13-50-7')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (124.50144550147462, 129.4508029688177))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-19_16-37-40')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (124.27871634925401, 129.58159934997875))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-21_11-19-2')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (124.01748457963947, 129.67389311374583))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-25_13-20-55')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (124.59625706838818, 130.18196049581826))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-25_17-33-28')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (124.54109241108247, 129.91468051848352))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-26_13-51-50')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (123.37674715861888, 129.74020675948856))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-28_12-38-13')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (124.79328080534583, 130.59109955280485))))
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-28_17-6-14')] = dict(grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (123.16323999738597, 128.89035363816865))))
                                
        # ==================================================================================================================================================================================================================================================================================== #
        # 2025-01-14 17:20 Generated by `_out_user_annotations_add_code_lines, loaded_configs = UserAnnotationsManager.batch_build_user_annotation_grid_bin_bounds_from_exported_position_info_mat_files(search_parent_path=Path('Data/KDIBA'))`                                               #
        # ==================================================================================================================================================================================================================================================================================== #
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-07_11-26-53')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.46995636983615874, 0.5010896201474913))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (135.23924311831908, 144.1984518410047))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4792909287353384, 0.504446594986895))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (137.925447118083, 145.16448776601297))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.48011981538341136, 0.5077661416598747))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (138.16397564990257, 146.1197529956474))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_3-23-37')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4842241482218449, 0.5128667385082708))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (139.34507862499134, 147.58755064986212))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-12_15-55-31')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4794675152730742, 0.5073629075520282))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (137.97626338793503, 146.00371440346137))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-13_14-42-6')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4805202167909671, 0.5054268170400331))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (138.2791990765373, 145.4465660546858))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4809145186559411, 0.5106410347082598))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (138.39266723911976, 146.9470603477007))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_15-46-47')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4948190683473956, 0.5252590207849699))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (142.39397650284766, 151.15367504603452))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_21-16-25')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4861378606162692, 0.5161021887970064))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (139.89578722770338, 148.51861548115295))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-09_22-24-40')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4840861413778695, 0.5142882816870253))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (139.3053644252862, 147.99662782360443))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-12_16-53-46')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.48621181777263267, 0.5168895823386255))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (139.91706986262813, 148.7452035506836))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-13_15-22-3')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4902240765084413, 0.5211396532028527))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (141.07167669307665, 149.9682455260008))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_17-46-44')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4729508196721312, 0.5024590163934426))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (136.10095530133273, 144.59252270314897))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_19-28-0')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.49153005464480876, 0.5133879781420765))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (141.44749773951332, 147.73754766678462))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_12-3-25')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.48278688524590163, 0.5095628415300546))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (138.9314777686048, 146.63678892951214))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_21-26-8')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4798773006134969, 0.5029447852760736))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (138.0941872269056, 144.73231230966147))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-05_19-26-43')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4958282208588957, 0.5184049079754601))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (142.68438010327935, 149.1812684821468))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_11-43-50')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.49288343558282216, 0.5149693251533742))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (141.83695987994884, 148.19261155492785))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_12-15-3')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4997546012269939, 0.5247852760736197))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (143.81427373438675, 151.01734563269633))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_21-17-16')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.49730061349693244, 0.520122699386503))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (143.1080902149446, 149.67559694575627))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_22-4-5')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4921472392638037, 0.5137423312883436))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (141.6251048241162, 147.8395197952068))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-01_12-58-54')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4869938650306749, 0.5117791411042945))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (140.14211943328775, 147.2745729796531))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-03_20-28-3')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.47865030674846615, 0.5026993865030674))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (137.74109546718452, 144.66169395771726))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-04_21-20-3')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4797297297297297, 0.4993857493857494))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (138.05172078553375, 143.70812931964014))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-09_17-29-30')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.37578417739783604, 0.4001085017221604))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (108.13933162527657, 115.13913718623321))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-10_12-25-50')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.3746667818805185, 0.39530560251933916))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (107.8177789584226, 113.7570079192343))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-10_21-2-40')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.37594355808304475, 0.39584527798476465))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (108.18519657066037, 113.91231021144307))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-11_15-16-59')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.3771319110598822, 0.39482232875029993))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (108.52716865032582, 113.61793633102158))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-12_14-39-31')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.3794482575497926, 0.39738437548591055))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (109.19374317979644, 114.35521596716852))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-12_17-53-55')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.37470767802175825, 0.3948550981691784))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (107.82954763216065, 113.6273663796197))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-16_15-12-23')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.37640036982474273, 0.3953192887436616))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (108.31665318697634, 113.76094640105372))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-17_12-33-47')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.38102246700179526, 0.3996956856750139))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (109.64675309404181, 115.02034120144286))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-18_13-6-1')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.3785823450695496, 0.3984840649712695))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (108.9445597322445, 114.6716733730272))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-18_15-23-32')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.3811064578229815, 0.3980597747762985))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (109.67092311452707, 114.54957547519382))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-19_13-34-40')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.3711325410534056, 0.3917713616922262))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (106.80073123839011, 112.7399601992018))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-19_16-48-9')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.3772441894886413, 0.3981287103731622))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (108.55947898953707, 114.56941305702512))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-21_10-24-35')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.3760960745804659, 0.39943759792198924))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (108.22908621020602, 114.94607134445735))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-25_14-28-51')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.3789719600570708, 0.4018220829071936))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (109.0566791531139, 115.63225407401256))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-25_17-17-6')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.37981592847443835, 0.40020904886755876))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (109.29954776242832, 115.16807161656368))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-26_13-22-13')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.37391529827718123, 0.3989767233386063))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (107.60152468408094, 114.81344556506657))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-27_14-43-12')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.37260355912763343, 0.3964364829605572))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (107.22404579212474, 114.08244113972872))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-28_16-48-29')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.3761691159786245, 0.3960708358803444))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (108.2501053175898, 113.9772189583725))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-09_16-40-54')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.42730804156792146, 0.4533522676121476))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (122.96634289724359, 130.46108420493456))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-10_12-58-3')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4268696057468411, 0.450702529579765))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (122.84017431563773, 129.69856966324173))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-10_19-11-57')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4272640096643624, 0.4501141325144853))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (122.9536718458597, 129.52924676675838))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-11_12-48-38')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.43450265627050333, 0.4553871771550242))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (125.03673561741104, 131.04666968489906))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-11_16-2-46')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4310858651302615, 0.45025048429488057))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (124.05348636842058, 129.56848468917428))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-12_15-25-59')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.43846596311767105, 0.45566498031668823))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (126.17725557343053, 131.1266130407736))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-16_14-49-24')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4305746253560496, 0.4502306450120692))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (123.90636700893516, 129.5627755430415))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-16_18-47-52')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4296298511654051, 0.447565969101523))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (123.63448954400148, 128.79596233137354))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-17_12-52-15')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4314081848914485, 0.4498357033189669))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (124.14624025653195, 129.44912325725667))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-18_13-28-57')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4271814241058154, 0.44757454449893574))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (122.92990621750084, 128.7984300716362))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-18_15-38-2')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4316969692990041, 0.4498787874808223))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (124.22934368316666, 129.46152157721505))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-19_13-50-7')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.43264252311762424, 0.4498415403166415))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (124.50144550147462, 129.4508029688177))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-19_16-37-40')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.43186853931365765, 0.4502960577411761))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (124.27871634925401, 129.58159934997875))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-21_11-19-2')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4309607589142471, 0.4506167785702667))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (124.01748457963947, 129.67389311374583))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-25_13-20-55')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4329719933126489, 0.4523823127229684))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (124.59625706838818, 130.18196049581826))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-25_17-33-28')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.43278029612851154, 0.4514535148017302))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (124.54109241108247, 129.91468051848352))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-26_13-51-50')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4287341963762006, 0.4508472184892227))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (123.37674715861888, 129.74020675948856))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-28_12-38-13')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.43365665079857674, 0.4538040709459968))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (124.79328080534583, 130.59109955280485))))
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-28_17-6-14')] = dict(unit_grid_bin_bounds=(((0.12884392935982347, 0.8711479028697573), (0.4279922589909162, 0.447893978892636))), cm_grid_bin_bounds=(((37.0773897438341, 250.69004399129707), (123.16323999738597, 128.89035363816865))))

        # ================================================================================================================================================================================ #
        # Override with the absolute outermost grid_bin_bounds [0.0, 1.0]                                                                                                                  #
        # ================================================================================================================================================================================ #
        pix2cm: float = 287.7697841726619
        real_unit_x_grid_bin_bounds = np.array([0.0, 1.0])
        real_cm_x_grid_bin_bounds = (real_unit_x_grid_bin_bounds * pix2cm)
        

        real_unit_y_grid_bin_bounds = np.array([0.4, 0.6])
        real_cm_y_grid_bin_bounds = (real_unit_y_grid_bin_bounds * pix2cm)
        

        # real_cm_x_grid_bin_bounds # array([0, 287.77])

        for a_ctxt, a_dict in user_annotations.items():
            # a_dict.update(pix2cm=pix2cm, real_unit_x_grid_bin_bounds=tuple(deepcopy(real_unit_x_grid_bin_bounds)),  real_cm_x_grid_bin_bounds=tuple(deepcopy(real_cm_x_grid_bin_bounds)))
            a_dict.update(pix2cm=pix2cm, real_unit_grid_bin_bounds=(tuple(deepcopy(real_unit_x_grid_bin_bounds)), tuple(deepcopy(real_unit_y_grid_bin_bounds))),  real_cm_grid_bin_bounds=(tuple(deepcopy(real_cm_x_grid_bin_bounds)), tuple(deepcopy(real_cm_y_grid_bin_bounds))))
            a_dict.update(grid_bin_bounds=(tuple(deepcopy(real_cm_x_grid_bin_bounds)), tuple(deepcopy(real_cm_y_grid_bin_bounds))))
        
        ## override with the "real" bounds

        # ==================================================================================================================== #
        # 2024-11-05 16:05 Produced programmatically from exported matlab csv                                                 #
        # ==================================================================================================================== #
        # """ 
        # ```python
        #     # global_session.epochs
        #     ## INPUTS: global_data_root_parent_path
        #     matlab_exported_csv = Path('/home/halechr/repos/matlab-to-neuropy-exporter/output/2024-11-05_good_sessions_table.csv').resolve()
        #     Assert.path_exists(matlab_exported_csv)
        #     session_info_matlab_df: pd.DataFrame = pd.read_csv(matlab_exported_csv)
        #     session_info_matlab_df['session_context'] = session_info_matlab_df['session_export_path'].map(lambda v: IdentifyingContext.try_init_from_session_key('_'.join(Path(v).relative_to(global_data_root_parent_path).parts), separator='_'))
        #     user_annotation_code_strs = []
        #     for a_tuple in session_info_matlab_df[['session_context', 'first_valid_pos_time', 'last_valid_pos_time']].itertuples(index=False):
        #         _code_str: str = f"user_annotations[{a_tuple.session_context.get_initialization_code_string()}].update(track_start_t={a_tuple.first_valid_pos_time}, track_end_t={a_tuple.last_valid_pos_time})"
        #         user_annotation_code_strs.append(_code_str)

        #     ## OUTPUTS: user_annotation_code_strs
        # ```
        # """
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19')].update(track_start_t=10.98) ## Manual
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-07_11-26-53')].update(track_start_t=10.1221450000303, track_end_t=1978.00547199999)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15')].update(track_start_t=9.03898199996911, track_end_t=2078.12502899999)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43')].update(track_start_t=38.9233210000675, track_end_t=1737.07045)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_3-23-37')].update(track_start_t=29.1566759999841, track_end_t=520.185412999941)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-12_15-55-31')].update(track_start_t=22.0592309999047, track_end_t=1119.23407399992)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-13_14-42-6')].update(track_start_t=12.8251579999924, track_end_t=863.916483999928)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19')].update(track_start_t=15.549006999936, track_end_t=2578.53313699993)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_15-46-47')].update(track_start_t=9.20195200003218, track_end_t=2619.60006099998)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_21-16-25')].update(track_start_t=12.8726610000012, track_end_t=1235.43789400009)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-09_22-24-40')].update(track_start_t=6.99150600004941, track_end_t=2562.566185)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-12_16-53-46')].update(track_start_t=12.7854130000342, track_end_t=811.422589999973)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-13_15-22-3')].update(track_start_t=10.7205819999799, track_end_t=878.426836000057)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-09_17-29-30')].update(track_start_t=np.nan, track_end_t=np.nan)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-10_12-25-50')].update(track_start_t=6.02368800001568, track_end_t=1426.15366000001)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-17_12-33-47')].update(track_start_t=18.9487729999964, track_end_t=869.104464999997)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-18_13-6-1')].update(track_start_t=20.6751609999919, track_end_t=810.47034)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-18_15-23-32')].update(track_start_t=12.6970189999847, track_end_t=508.128582999983)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-19_13-34-40')].update(track_start_t=15.89814200002, track_end_t=666.954281000013)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-27_14-43-12')].update(track_start_t=16.7988520000363, track_end_t=913.63453400007)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-09_16-40-54')].update(track_start_t=26.0018009999912, track_end_t=1743.198221)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-10_12-58-3')].update(track_start_t=27.4544520000054, track_end_t=1471.14405999999)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-11_12-48-38')].update(track_start_t=1.93768199999977, track_end_t=1279.892742)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-11_16-2-46')].update(track_start_t=23.3863830000009, track_end_t=893.397008)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-12_15-25-59')].update(track_start_t=19.5205369999894, track_end_t=851.158837999988)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-16_14-49-24')].update(track_start_t=18.695103, track_end_t=1034.350836)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-16_18-47-52')].update(track_start_t=23.261978999999, track_end_t=760.703594999999)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-17_12-52-15')].update(track_start_t=21.5540140000085, track_end_t=807.348131000006)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-18_13-28-57')].update(track_start_t=0.633843999996316, track_end_t=622.394167999999)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-19_13-50-7')].update(track_start_t=13.964290999982, track_end_t=507.294616999978)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-19_16-37-40')].update(track_start_t=17.5178040000028, track_end_t=482.08531200001)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-21_11-19-2')].update(track_start_t=11.430015000049, track_end_t=583.70736)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-25_13-20-55')].update(track_start_t=25.2757259999635, track_end_t=569.291590000037)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-26_13-51-50')].update(track_start_t=18.7161159999669, track_end_t=4277.00783000002)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-28_12-38-13')].update(track_start_t=9.6452860001009, track_end_t=524.096874000039)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_17-46-44')].update(track_start_t=-75.3297229999998, track_end_t=1941.410477)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_19-28-0')].update(track_start_t=29.6559119999984, track_end_t=1177.572948)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_12-3-25')].update(track_start_t=19.1183459999993, track_end_t=1003.345493)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_21-26-8')].update(track_start_t=34.375172, track_end_t=860.640345)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-05_19-26-43')].update(track_start_t=4.25055900000007, track_end_t=1241.666767)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_11-43-50')].update(track_start_t=7.62950799999999, track_end_t=727.620426)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_12-15-3')].update(track_start_t=14.9508269999997, track_end_t=764.041609)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_21-17-16')].update(track_start_t=14.403914, track_end_t=719.78661)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-09_22-4-5')].update(track_start_t=20.9049300000001, track_end_t=745.123835)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-01_12-58-54')].update(track_start_t=13.4352230000004, track_end_t=3031.404034)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-03_20-28-3')].update(track_start_t=16.0465110000005, track_end_t=1085.451218)
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-04_21-20-3')].update(track_start_t=8.9677370000004, track_end_t=1139.771152)

        return user_annotations


    @classmethod
    def get_hardcoded_good_sessions(cls) -> List[IdentifyingContext]:
        """Hardcoded included_session_contexts:
                
        Usage:
            from neuropy.core.user_annotations import UserAnnotationsManager
        
            included_session_contexts: List[IdentifyingContext] = UserAnnotationsManager.get_hardcoded_good_sessions()
            
            
        """
        bad_sessions = cls.get_hardcoded_bad_sessions()
        good_sessions = [
            # IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-07_11-26-53'), # ONLY ONE SHORT LAP
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15'), # prev completed
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43'), # prev completed
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-12_15-55-31'), # prev completed
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19'), # prev completed
            # IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_21-16-25'), # 2024-10-04 - has long placefields outside of the possible bounds on the short track for some reason?!?!
            # IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-09_22-24-40'), # 2024-10-04 - has long placefields outside of the possible bounds on the short track for some reason?!?!
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-12_16-53-46'),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-09_17-29-30'),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-10_12-25-50'),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-09_16-40-54'),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-10_12-58-3'),
            # IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_19-28-0'), # 2024-10-04 - has long placefields outside of the possible bounds on the short track for some reason?!?!
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_12-3-25'),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-01_12-58-54'), # 2024-10-04 - very bad position tracking data (jumping around everywhere at high frequency)
        ]
        return [v for v in good_sessions if (v not in bad_sessions)]


    @classmethod
    def get_hardcoded_bad_sessions(cls) -> List[IdentifyingContext]:
        """ Hardcoded excluded_session_contexts:
        
        Usage:
            from neuropy.core.user_annotations import UserAnnotationsManager
        
            excluded_session_contexts: List[IdentifyingContext] = UserAnnotationsManager.get_hardcoded_bad_sessions()
            bad_session_df: pd.DataFrame = pd.DataFrame.from_records([v.to_dict() for v in excluded_session_contexts], columns=['format_name', 'animal', 'exper_name', 'session_name'])
            bad_session_df

        Built Via:
            from neuropy.core.session.Formats.Specific.KDibaOldDataSessionFormat import KDibaOldDataSessionFormatRegisteredClass
            bad_sessions_csv_path = Path(r'~/repos/matlab-to-neuropy-exporter/output/2024-09-23_bad_sessions_table.csv').resolve() ## exported from `IIDataMat_Export_ToPython_2022_08_01.m`
            bad_session_df, bad_session_contexts = KDibaOldDataSessionFormatRegisteredClass.load_bad_sessions_csv(bad_sessions_csv_path=bad_sessions_csv_path)        
            
        
        """
        bad_session_contexts: List[IdentifyingContext] = [
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-07_11-26-53'), # ONLY ONE SHORT LAP
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_21-16-25'), # 2024-10-04 - has long placefields outside of the possible bounds on the short track for some reason?!?!
            IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-09_22-24-40'), # 2024-10-04 - has long placefields outside of the possible bounds on the short track for some reason?!?!
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-10_21-2-40'),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-11_15-16-59'),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-12_14-39-31'),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-12_17-53-55'),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-16_15-12-23'),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-16_18-47-52'),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-19_16-48-9'),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-21_10-24-35'),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-25_14-28-51'),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-25_17-17-6'),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-26_13-22-13'),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-28_12-17-27'),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-28_16-48-29'),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-10_19-11-57'),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-12_14-59-23'),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-18_15-38-2'),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-25_17-33-28'),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-27_18-21-57'),
            IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-28_17-6-14'),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_17-46-44'), # 2024-10-01 - bad laps, Bad Laps "3"/"4" - don't see lap 1/2
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_19-28-0'), # 2024-10-04 - has long placefields outside of the possible bounds on the short track for some reason?!?!
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-19_12-35-59'),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-19_13-2-0'),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-19_13-55-7'),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-03_11-0-53'),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='redundant'),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='showclus'),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='sleep'),
            IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='tmaze')
        ]

        return bad_session_contexts


    @classmethod
    def get_all_known_sessions(cls) -> List[IdentifyingContext]:
        """Hardcoded included_session_contexts:
                
        Usage:
            from neuropy.core.user_annotations import UserAnnotationsManager
        
            included_session_contexts: List[IdentifyingContext] = UserAnnotationsManager.get_hardcoded_good_sessions()
            
        """
        bad_sessions = cls.get_hardcoded_bad_sessions()
        good_sessions = cls.get_hardcoded_good_sessions()
        return (bad_sessions + good_sessions)
        

    @classmethod
    def get_hardcoded_bimodal_aclus(cls) -> Dict[IdentifyingContext, List]:
        """
        Cells/aclus that are bimodal
        """
        bad_aclu_mapping = {
            IdentifyingContext(format_name= 'kdiba', animal= 'vvp01', exper_name= 'one', session_name= '2006-4-09_17-29-30'):
            dict(
                bad_looking_aclus = [4, 25],
                minorly_bad_looking_aclus = [7, 28, 22],
            )
        }

        ## Bad/Icky Bimodal Cells:
        return {IdentifyingContext(format_name= 'kdiba', animal= 'vvp01', exper_name= 'one', session_name= '2006-4-10_12-25-50'): [7, 36, 31, 4, 32, 27, 13, ],

        }



    @classmethod
    def get_hardcoded_laps_override_dict(cls) -> Dict[IdentifyingContext, pd.DataFrame]:
        """
        Return custom-tweaked laps_df objects for each session
        
        """
        laps_override_dict = {IdentifyingContext(format_name= 'kdiba', animal= 'gor01', exper_name= 'one', session_name= '2006-6-09_1-22-43'):np.array([[45.196, 50.87, 1],
            [65.785, 72.223, 0],
            [84.3034, 91.0442, 1],
            [106.426, 112.965, 0],
            [244.13, 250.671, 1],
            [260.414, 281.269, 0],
            [281.302, 290.778, 1],
            [308.896, 313.935, 0],
            [357.845, 365.118, 1],
            [370.624, 375.996, 0],
            [399.654, 406.36, 1],
            [415.704, 422.177, 0],
            [428.285, 435.323, 1],
            [441.164, 445.835, 0],
            [462.884, 470.192, 1],
            [480.803, 489.746, 0],
            [491.313, 496.985, 1],
            [503.559, 509.798, 0],
            [514.037, 520.176, 1],
            [528.45, 533.022, 0],
            [536.058, 541.798, 1],
            [546.403, 551.775, 0],
            [555.412, 563.486, 1],
            [565.554, 570.76, 0],
            [574.063, 580.235, 1],
            [584.542, 591.581, 0],
            [644.3, 652.009, 1],
            [682.545, 688.613, 0],
            [841.1, 847.706, 1],
            [855.014, 858.417, 0],
            [873.596, 878.203, 1],
            [890.015, 895.488, 0],
            [905.498, 910.802, 1],
            [918.243, 922.282, 0],
            [925.44, 930.589, 1],
            [936.163, 940.765, 0],
            [944.403, 949.51, 1],
            [957.918, 962.622, 0],
            [965.523, 972.998, 1],
            [982.841, 992.619, 0],
            [995.821, 1001.56, 1],
            [1014.04, 1018.24, 0],
            [1039.93, 1047.01, 1],
            [1049.14, 1054.25, 0],
            [1064.03, 1068.2, 1],
            [1070.1, 1074.94, 0],
            [1087.55, 1091.32, 1],
            [1095.52, 1100.69, 0],
            [1108.04, 1115.64, 1],
            [1120.98, 1124.42, 0],
            [1151.44, 1157.08, 1],
            [1172.9, 1177.44, 0],
            [1182.98, 1186.65, 1],
            [1201.46, 1204.86, 0],
            [1334.76, 1339.94, 1],
            [1354.25, 1358.02, 0],
            [1380.84, 1385.25, 1],
            [1391.02, 1393.99, 0],
            [1395.92, 1400.33, 1],
            [1402.37, 1410.74, 0],
            [1472.74, 1476.84, 1],
            [1478.24, 1482.68, 0],
            [1487.89, 1492.96, 1],
            [1493.02, 1498.76, 0],
            [1500.5, 1504.47, 1],
            [1509.34, 1512.48, 0],
            [1513.54, 1520.75, 1],
            [1522.62, 1526.56, 0],
            [1528.96, 1532.8, 1],
            [1535.26, 1539.5, 0],
            [1540.5, 1546.71, 1],
            [1557.66, 1561.23, 0],
            [1561.26, 1567.33, 1],
            [1579.44, 1583.55, 0],
            [1585.2, 1592, 1],
            [1594.49, 1598.46, 0],
            [1604.67, 1607.8, 1],
            [1608.84, 1614.15, 0],
            [1620.12, 1625.62, 1],
            [1628.79, 1632.6, 0],
            [1648.42, 1652.35, 1],
            [1654.45, 1658.09, 0]]),
        }

        ## Bad/Icky Bimodal Cells:
        return {k:pd.DataFrame(v, columns=['start', 'stop', 'lap_dir']) for k, v in laps_override_dict.items()}



    # ==================================================================================================================== #
    # Helper/Utility Functions                                                                                             #
    # ==================================================================================================================== #
        
    @function_attributes(short_name=None, tags=['utility', 'importer', 'batch', 'matlab_mat_file', 'multi-session', 'grid_bin_bounds'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-10 07:41', related_items=[])
    @classmethod
    def batch_build_user_annotation_grid_bin_bounds_from_exported_position_info_mat_files(cls, search_parent_path: Path, platform_side_length: float = 22.0, debug_print=False):
        """ finds all *.position_info.mat files recurrsively in the search_parent_path, then try to load them and parse their parent directory as a session to build an IdentifyingContext that can be used as a key in UserAnnotations.
        
        Builds a list of grid_bin_bounds user annotations that can be pasted into `specific_session_override_dict`

        Usage:
            from neuropy.core.user_annotations import UserAnnotationsManager

            _out_user_annotations_add_code_lines, loaded_configs = UserAnnotationsManager.batch_build_user_annotation_grid_bin_bounds_from_exported_position_info_mat_files(search_parent_path=Path(r'W:\\Data\\Kdiba'))
            loaded_configs

        """
        # from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder, DataSessionFormatBaseRegisteredClass
        from neuropy.core.session.Formats.Specific.KDibaOldDataSessionFormat import KDibaOldDataSessionFormatRegisteredClass
        from pyphocorehelpers.geometry_helpers import BoundsRect
        from pyphocorehelpers.Filesystem.path_helpers import discover_data_files

        found_any_position_info_mat_files: list[Path] = discover_data_files(search_parent_path, glob_pattern='*.position_info.mat', recursive=True)
        found_any_position_info_mat_files
        ## INPUTS: found_any_position_info_mat_files
        
        user_annotations = cls.get_user_annotations()
        _out_user_annotations_add_code_lines: List[str] = []

        loaded_configs = {}
        for a_path in found_any_position_info_mat_files:
            if a_path.exists():
                file_prefix: str = a_path.name.split('.position_info.mat', maxsplit=1)[0]
                if debug_print:
                    print(f'file_prefix: {file_prefix}')
                session_path = a_path.parent.resolve()
                session_name = KDibaOldDataSessionFormatRegisteredClass.get_session_name(session_path)
                if debug_print:
                    print(f'session_name: {session_name}')
                if session_name == file_prefix:
                    try:
                        _updated_config_dict = {}
                        _test_session = KDibaOldDataSessionFormatRegisteredClass.build_session(session_path) # Path(r'R:\data\KDIBA\gor01\one\2006-6-07_11-26-53')
                        _test_session_context = _test_session.get_context()
                        if debug_print:
                            print(f'_test_session_context: {_test_session_context}')
                        
                        _updated_config_dict = KDibaOldDataSessionFormatRegisteredClass.perform_load_position_info_mat(session_position_mat_file_path=a_path, config_dict=_updated_config_dict, debug_print=debug_print)
                        _updated_config_dict['session_context'] = _test_session_context

                        assert 'pix2cm' in _updated_config_dict, f"must have pix2cm to convert from cm to unit frames. 2025-01-14 17:10."
                        pix2cm: float = float(_updated_config_dict['pix2cm'])

                        loaded_track_limits = _updated_config_dict.get('loaded_track_limits', None) # {'long_xlim': array([59.0774, 228.69]), 'short_xlim': array([94.0156, 193.757]), 'long_ylim': array([138.164, 146.12]), 'short_ylim': array([138.021, 146.263])}
                        if loaded_track_limits is not None:
                            long_xlim = loaded_track_limits['long_xlim']
                            long_ylim = loaded_track_limits['long_ylim']
                            # short_xlim = loaded_track_limits['short_xlim']
                            # short_ylim = loaded_track_limits['short_ylim']

                            long_unit_xlim = loaded_track_limits['long_unit_xlim']
                            long_unit_ylim = loaded_track_limits['long_unit_ylim']
                            
                            platform_side_unit_length: float = (float(platform_side_length) / pix2cm)
                            from_mat_unit_lims_grid_bin_bounds = BoundsRect(xmin=(long_unit_xlim[0]-platform_side_unit_length), xmax=(long_unit_xlim[1]+platform_side_unit_length), ymin=long_unit_ylim[0], ymax=long_unit_ylim[1])
                            
                            ## Build full grid_bin_bounds:
                            from_mat_cm_lims_grid_bin_bounds = BoundsRect(xmin=(long_xlim[0]-platform_side_length), xmax=(long_xlim[1]+platform_side_length), ymin=long_ylim[0], ymax=long_ylim[1])
                            # print(f'from_mat_lims_grid_bin_bounds: {from_mat_lims_grid_bin_bounds}')
                            # from_mat_lims_grid_bin_bounds # BoundsRect(xmin=37.0773897438341, xmax=250.69004399129707, ymin=116.16397564990257, ymax=168.1197529956474)
                            
                            ## pre 2025-01-14 16:58 [cm] grid-bin-bounds:
                            # from_mat_lims_grid_bin_bounds: BoundsRect = deepcopy(from_mat_cm_lims_grid_bin_bounds)
                            # _updated_config_dict['new_grid_bin_bounds'] = from_mat_lims_grid_bin_bounds.extents ## how it used to be pre 2025-01-14 16:58 
                            

                            _updated_config_dict['new_cm_grid_bin_bounds'] = from_mat_cm_lims_grid_bin_bounds.extents
                            _updated_config_dict['new_unit_grid_bin_bounds'] = from_mat_unit_lims_grid_bin_bounds.extents
                            


                            ## 2025-01-14 16:58 Use Unit-length Grid bin bounds:
                            from_mat_lims_grid_bin_bounds: BoundsRect = deepcopy(from_mat_unit_lims_grid_bin_bounds)
                            _updated_config_dict['new_grid_bin_bounds'] = _updated_config_dict['new_unit_grid_bin_bounds'] # 2025-01-14 16:58 - switch to unit scale bounds
                             
                            ## Build Update:


                            # curr_active_pipeline.sess.config.computation_config['pf_params'].grid_bin_bounds = new_grid_bin_bounds

                            active_context = _test_session_context # curr_active_pipeline.get_session_context()
                            final_context = active_context.adding_context_if_missing(user_annotation='grid_bin_bounds')
                            user_annotations[final_context] = from_mat_lims_grid_bin_bounds.extents
                            



                            # Updates the context. Needs to generate the code.

                            ## Generate code to insert int user_annotations:
                            
                            ## new style:
                            # print(f"user_annotations[{final_context.get_initialization_code_string()}] = {(from_mat_lims_grid_bin_bounds.xmin, from_mat_lims_grid_bin_bounds.xmax), (from_mat_lims_grid_bin_bounds.ymin, from_mat_lims_grid_bin_bounds.ymax)}")
                            # _a_code_line: str = f"user_annotations[{active_context.get_initialization_code_string()}] = dict(grid_bin_bounds=({(from_mat_lims_grid_bin_bounds.xmin, from_mat_lims_grid_bin_bounds.xmax), (from_mat_lims_grid_bin_bounds.ymin, from_mat_lims_grid_bin_bounds.ymax)}))"                            
                            _a_code_line: str = f"""user_annotations[{active_context.get_initialization_code_string()}] = dict(unit_grid_bin_bounds=({(from_mat_unit_lims_grid_bin_bounds.xmin, from_mat_unit_lims_grid_bin_bounds.xmax), (from_mat_unit_lims_grid_bin_bounds.ymin, from_mat_unit_lims_grid_bin_bounds.ymax)}), cm_grid_bin_bounds=({(from_mat_cm_lims_grid_bin_bounds.xmin, from_mat_cm_lims_grid_bin_bounds.xmax), (from_mat_cm_lims_grid_bin_bounds.ymin, from_mat_cm_lims_grid_bin_bounds.ymax)}))"""
                            # _a_code_line: str = f"user_annotations[{active_context.get_initialization_code_string()}] = dict(unit_grid_bin_bounds=({(from_mat_unit_lims_grid_bin_bounds.xmin, from_mat_unit_lims_grid_bin_bounds.xmax), (from_mat_unit_lims_grid_bin_bounds.ymin, from_mat_unit_lims_grid_bin_bounds.ymax)}),\n\t"
                            # _a_code_line += f"cm_grid_bin_bounds=({(from_mat_cm_lims_grid_bin_bounds.xmin, from_mat_cm_lims_grid_bin_bounds.xmax), (from_mat_cm_lims_grid_bin_bounds.ymin, from_mat_cm_lims_grid_bin_bounds.ymax)}),\n"
                            # _a_code_line += f")"

                            # print(_a_code_line)
                            _out_user_annotations_add_code_lines.append(_a_code_line)

                        loaded_configs[a_path] = _updated_config_dict
                    except (AssertionError, Exception) as e:
                        print(f'e: {e}. Skipping.')
                        pass
                        
        # loaded_configs

        print('Add the following code to UserAnnotationsManager.get_user_annotations() function body:')
        print('\n'.join(_out_user_annotations_add_code_lines))
        return _out_user_annotations_add_code_lines, loaded_configs

    # @classmethod
    # def _build_new_grid_bin_bounds_from_mat_exported_xlims(cls, curr_active_pipeline, platform_side_length: float = 22.0, build_user_annotation_string:bool=False):
    #     """ 2024-04-10 -
        
    #     from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _build_new_grid_bin_bounds_from_mat_exported_xlims

    #     """
    #     from neuropy.core.user_annotations import UserAnnotationsManager
    #     from pyphocorehelpers.geometry_helpers import BoundsRect

    #     ## INPUTS: curr_active_pipeline
    #     loaded_track_limits = deepcopy(curr_active_pipeline.sess.config.loaded_track_limits) # {'long_xlim': array([59.0774, 228.69]), 'short_xlim': array([94.0156, 193.757]), 'long_ylim': array([138.164, 146.12]), 'short_ylim': array([138.021, 146.263])}
    #     x_midpoint: float = curr_active_pipeline.sess.config.x_midpoint
    #     pix2cm: float = curr_active_pipeline.sess.config.pix2cm

    #     ## INPUTS: loaded_track_limits
    #     print(f'loaded_track_limits: {loaded_track_limits}')

    #     long_xlim = loaded_track_limits['long_xlim']
    #     long_ylim = loaded_track_limits['long_ylim']
    #     short_xlim = loaded_track_limits['short_xlim']
    #     short_ylim = loaded_track_limits['short_ylim']


    #     ## Build full grid_bin_bounds:
    #     from_mat_lims_grid_bin_bounds = BoundsRect(xmin=(long_xlim[0]-platform_side_length), xmax=(long_xlim[1]+platform_side_length), ymin=long_ylim[0], ymax=long_ylim[1])
    #     # print(f'from_mat_lims_grid_bin_bounds: {from_mat_lims_grid_bin_bounds}')
    #     # from_mat_lims_grid_bin_bounds # BoundsRect(xmin=37.0773897438341, xmax=250.69004399129707, ymin=116.16397564990257, ymax=168.1197529956474)

    #     if build_user_annotation_string:
    #         ## Build Update:
    #         user_annotations = UserAnnotationsManager.get_user_annotations()

    #         # curr_active_pipeline.sess.config.computation_config['pf_params'].grid_bin_bounds = new_grid_bin_bounds

    #         active_context = curr_active_pipeline.get_session_context()
    #         final_context = active_context.adding_context_if_missing(user_annotation='grid_bin_bounds')
    #         user_annotations[final_context] = from_mat_lims_grid_bin_bounds.extents
    #         # Updates the context. Needs to generate the code.

    #         ## Generate code to insert int user_annotations:
    #         print('Add the following code to UserAnnotationsManager.get_user_annotations() function body:')
    #         ## new style:
    #         # print(f"user_annotations[{final_context.get_initialization_code_string()}] = {(from_mat_lims_grid_bin_bounds.xmin, from_mat_lims_grid_bin_bounds.xmax), (from_mat_lims_grid_bin_bounds.ymin, from_mat_lims_grid_bin_bounds.ymax)}")
    #         print(f"user_annotations[{active_context.get_initialization_code_string()}] = dict(grid_bin_bounds=({(from_mat_lims_grid_bin_bounds.xmin, from_mat_lims_grid_bin_bounds.xmax), (from_mat_lims_grid_bin_bounds.ymin, from_mat_lims_grid_bin_bounds.ymax)}))")

    #     return from_mat_lims_grid_bin_bounds


