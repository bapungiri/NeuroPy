from copy import deepcopy
from attrs import define, field, filters, asdict, astuple, Factory
import h5py
import pandas as pd
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, gaussian_filter1d, interpolation
from neuropy.analyses.placefields import PfND, PlacefieldComputationParameters
from neuropy.core.epoch import Epoch
from neuropy.core.position import Position
from neuropy.core.ratemap import Ratemap
from neuropy.analyses.placefields import _normalized_occupancy
from neuropy.utils.dynamic_container import DynamicContainer
from neuropy.utils.misc import safe_pandas_get_group, copy_if_not_none
## Need to apply position binning to the spikes_df's position columns to find the bins they fall in:
from neuropy.utils.mixins.binning_helpers import build_df_discretized_binned_position_columns # for perform_time_range_computation only
from neuropy.utils.mixins.unit_slicing import NeuronUnitSlicableObjectProtocol # allows placefields to be sliced by neuron ids

from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, serialized_field, serialized_attribute_field, non_serialized_field, custom_define
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin

from typing import Dict, OrderedDict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from typing import NewType
import neuropy.utils.type_aliases as types

SnapshotTimestamp = NewType('SnapshotTimestamp', float)

## On saving:
# 2023-04-20 - AttributeError: 'PfND_TimeDependent' object has no attribute '_included_thresh_neurons_indx'
# I think this relates to `__attrs_post_init__` not being called upon unpickling.



@custom_define(slots=False, repr=False)
class PlacefieldSnapshot(HDFMixin, AttrsBasedClassHelperMixin):
    """Holds a snapshot in time for `PfND_TimeDependent`

    raw_occupancy_map: num_position_samples_occupancy
    raw_smoothed_occupancy_map: num_position_samples_smoothed_occupancy
    """
    num_position_samples_occupancy: Optional[np.ndarray] = serialized_field(metadata={'shape': ('n_xbins')})
    num_position_samples_smoothed_occupancy: Optional[np.ndarray] = serialized_field(metadata={'shape': ('n_xbins')})
    seconds_occupancy: np.ndarray = serialized_field(metadata={'shape': ('n_xbins')})
    normalized_occupancy: Optional[np.ndarray] = serialized_field(metadata={'shape': ('n_xbins')})
    spikes_maps_matrix: np.ndarray = serialized_field(metadata={'shape': ('n_neurons', 'n_xbins')})
    smoothed_spikes_maps_matrix: Optional[np.ndarray] = serialized_field(metadata={'shape': ('n_neurons', 'n_xbins')})
    occupancy_weighted_tuning_maps_matrix: np.ndarray = serialized_field(metadata={'shape': ('n_neurons', 'n_xbins')})
    
    # def __init__(self, num_position_samples_occupancy, num_position_samples_smoothed_occupancy, seconds_occupancy, normalized_occupancy, spikes_maps_matrix, smoothed_spikes_maps_matrix, occupancy_weighted_tuning_maps_matrix):
    #     super(PlacefieldSnapshot, self).__init__()
    #     self.num_position_samples_occupancy = num_position_samples_occupancy
    #     self.num_position_samples_smoothed_occupancy = num_position_samples_smoothed_occupancy
    #     self.seconds_occupancy = seconds_occupancy
    #     self.normalized_occupancy = normalized_occupancy

    #     self.spikes_maps_matrix = spikes_maps_matrix
    #     self.smoothed_spikes_maps_matrix = smoothed_spikes_maps_matrix
    #     self.occupancy_weighted_tuning_maps_matrix = occupancy_weighted_tuning_maps_matrix


    def to_dict(self):
        # print(f'WHOAAA PlacefieldSnapshot.to_dict(): {self.__dict__}')
        return {
            'num_position_samples_occupancy':self.num_position_samples_occupancy.copy(),
            'num_position_samples_smoothed_occupancy': copy_if_not_none(self.num_position_samples_smoothed_occupancy),
            'seconds_occupancy':self.seconds_occupancy.copy(),
            'normalized_occupancy':self.normalized_occupancy.copy(),
            'spikes_maps_matrix':self.spikes_maps_matrix.copy(),
            'smoothed_spikes_maps_matrix': copy_if_not_none(self.smoothed_spikes_maps_matrix),
            'occupancy_weighted_tuning_maps_matrix':self.occupancy_weighted_tuning_maps_matrix.copy()
        }

    @classmethod
    def from_dict(cls, d):
        # builds from a dict with keys: ['spikes_maps_matrix', 'smoothed_spikes_maps_matrix', 'num_position_samples_occupancy', 'num_position_samples_smoothed_occupancy', 'seconds_occupancy', 'normalized_occupancy', 'occupancy_weighted_tuning_maps_matrix']
        return cls(
            num_position_samples_occupancy=d["num_position_samples_occupancy"],
            num_position_samples_smoothed_occupancy=d["num_position_samples_smoothed_occupancy"],
            seconds_occupancy=d["seconds_occupancy"],
            normalized_occupancy=d["normalized_occupancy"],
            spikes_maps_matrix=d["spikes_maps_matrix"],
            smoothed_spikes_maps_matrix=d["smoothed_spikes_maps_matrix"],
            occupancy_weighted_tuning_maps_matrix=d["occupancy_weighted_tuning_maps_matrix"],
        )


    ## For serialization/pickling:
    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, state):
        """ assumes state is a dict generated by calling self.__getstate__() previously"""
        if 'num_position_samples_occupancy' not in state:
            assert 'raw_occupancy_map' in state
            state['num_position_samples_occupancy'] = state.pop('raw_occupancy_map', None)
        if 'num_position_samples_smoothed_occupancy' not in state:
            assert 'raw_smoothed_occupancy_map' in state
            state['num_position_samples_smoothed_occupancy'] = state.pop('raw_smoothed_occupancy_map', None)

        self.__dict__ = state # set the dict

    def get_by_IDX(self, included_indicies):
        """Returns a copy of the snapshot with only the neurons specified by the included_indicies        
        NOTE: cannot provide the usual get_by_ids() because this class doesn't keep track of or have access to the neuron_ids.
        """
        copy_snapshot = deepcopy(self)
        copy_snapshot.spikes_maps_matrix = copy_snapshot.spikes_maps_matrix[included_indicies]
        if copy_snapshot.smoothed_spikes_maps_matrix is not None:
            copy_snapshot.smoothed_spikes_maps_matrix = copy_snapshot.smoothed_spikes_maps_matrix[included_indicies]
            
        copy_snapshot.occupancy_weighted_tuning_maps_matrix = copy_snapshot.occupancy_weighted_tuning_maps_matrix[included_indicies]
        return copy_snapshot


def prepare_snapshots_for_export_as_xarray(historical_snapshots: dict, ndim: int=1) -> xr.DataArray:
    """ exports all snapshots as an xarray """
    snapshot_t_times = np.array(list(historical_snapshots.keys()), dtype=np.float64)
    if ndim < 2:
        pos_dim_labels = ("xbin", )
    else:
        pos_dim_labels = ("xbin", "ybin")

    # occupancy_weighted_tuning_maps_over_time_arr = np.stack([placefield_snapshot.occupancy_weighted_tuning_maps_matrix for timestamp_t, placefield_snapshot in self.historical_snapshots.items()]) # Concatenates so that the time is the first dimension
    occupancy_weighted_tuning_maps_over_time_xr = xr.DataArray(np.stack([placefield_snapshot.occupancy_weighted_tuning_maps_matrix for placefield_snapshot in historical_snapshots.values()]), dims=("snapshot_t", "neuron_idx", *pos_dim_labels), name="tuning_maps", coords={'snapshot_t': snapshot_t_times}) # , coords={"x": [10, 20]}
    occupancy_weighted_tuning_maps_over_time_xr.attrs["long_name"] = "occupancy_weighted_tuning_maps"
    occupancy_weighted_tuning_maps_over_time_xr.attrs["units"] = "spikes/sec"
    occupancy_weighted_tuning_maps_over_time_xr.attrs["description"] = "The tuning maps for each cell normalized for their occupancy."
    occupancy_weighted_tuning_maps_over_time_xr.attrs["ndim"] = ndim
    return occupancy_weighted_tuning_maps_over_time_xr


@define(slots=False, repr=False)
class PfND_TimeDependent(PfND):
    """ Time Dependent N-dimensional Placefields
        A version PfND that can return the current state of placefields considering only up to a certain period of time.
    
        Represents a collection of placefields at a given time over binned, N-dimensional space. 
        
        
        from copy import deepcopy
        from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent
        from neuropy.plotting.placemaps import plot_all_placefields

        included_epochs = None
        computation_config = active_session_computation_configs[0]
        # PfND version:
        t_list = []
        ratemaps_list = []
        active_time_dependent_placefields2D = PfND_TimeDependent(deepcopy(sess.spikes_df.copy()), deepcopy(sess.position), epochs=included_epochs,
                                        speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
                                        grid_bin=computation_config.grid_bin, smooth=computation_config.smooth)
        print('\t done.')
        # np.shape(active_time_dependent_placefields2D.curr_spikes_maps_matrix) # (64, 64, 29)


    To REMOVE from parent class:
        filtered_pos_df, filtered_spikes_df, ratemap
    """
    last_t: float = np.finfo('float').max # set to maximum value (so all times are included) just for setup.
    historical_snapshots: Dict[SnapshotTimestamp, PlacefieldSnapshot] = Factory(dict)
    fragile_linear_neuron_IDXs: np.array = None
    n_fragile_linear_neuron_IDXs: int = None
    # _included_thresh_neurons_indx: np.array
    # _filtered_spikes_df: np.ndarray = field(init=False)
    # _included_thresh_neurons_indx: np.ndarray = field(init=False)
    # _peak_frate_filter_function: list = field(init=False)


    # is_additive_mode = True # Default, cannot backtrack to earlier times.
    is_additive_mode = False # allows selecting any times of time range, but recomputes on each update (is not additive). This means later times will take longer to calculate the earlier ones. 
    
    @property
    def smooth(self):
        """The smooth property."""
        return self.config.smooth

    ########## Overrides for temporal dependence:
    """ 
        Define all_time_* versions of the self.filtered_pos_df and self.filtered_spikes_df properties to allow access to the underlying dataframes for setup and other purposes.
    """
    @property
    def all_time_filtered_spikes_df(self):
        """The filtered_spikes_df property."""
        return self._filtered_spikes_df
        
    @property
    def all_time_filtered_pos_df(self):
        """The filtered_pos_df property."""
        return self._filtered_pos_df
        
    @property
    def earliest_spike_time(self):
        """The earliest spike time."""
        return self.all_time_filtered_spikes_df[self.all_time_filtered_spikes_df.spikes.time_variable_name].values[0]
    
    @property
    def earliest_position_time(self):
        """The earliest position sample time."""
        return self.all_time_filtered_pos_df['t'].values[0]
    
    @property
    def earliest_valid_time(self):
        """The earliest time that we have both position and spikes (the later of the two individual earliest times)"""
        return max(self.earliest_position_time, self.earliest_spike_time)
        
        
    """ 
        Override the filtered_spikes_df and filtered_pos_df properties such that they only return the dataframes up to the last time (self.last_t).
        This allows access via self.t, self.x, self.y, self.speed, etc as defined in the parent class to work as expected since they access the self.filtered_pos_df and self.filtered_spikes_df
        
        Note: these would be called curr_filtered_spikes_df and curr_filtered_pos_df in the nomenclature of this class, but they're defined without the curr_* prefix for compatibility and to override the parent implementation.
    """
    @property
    def filtered_spikes_df(self):
        """The filtered_spikes_df property."""
        return self._filtered_spikes_df.spikes.time_sliced(0, self.last_t)
        
    @property
    def filtered_pos_df(self):
        """The filtered_pos_df property."""
        return self._filtered_pos_df.position.time_sliced(0, self.last_t)
        
        
    @property
    def ratemap_spiketrains(self):
        """ a list of spike times for each cell. for compatibility with old plotting functions."""        
        ## Get only the relevant columns and the 'aclu' column before grouping on aclu for efficiency:
        return self.curr_ratemap_spiketrains(self.last_t)
        
    @property
    def ratemap_spiketrains_pos(self):
        """ a list of spike positions for each cell. for compatibility with old plotting functions."""
        return self.curr_ratemap_spiketrains_pos(self.last_t)
    
    
    
    def curr_ratemap_spiketrains_pos(self, t):
        """ gets the ratemap_spiketrains_pos variable at the time t """
        if (self.ndim > 1):
            return [safe_pandas_get_group(self.all_time_filtered_spikes_df.spikes.time_sliced(0, t)[['aclu', self.all_time_filtered_spikes_df.spikes.time_variable_name, 'x', 'y']].groupby('aclu')['x', 'y'], neuron_id).to_numpy().T for neuron_id in self.included_neuron_IDs] # dataframes split for each ID
        else:
            return [safe_pandas_get_group(self.all_time_filtered_spikes_df.spikes.time_sliced(0, t)[['aclu', self.all_time_filtered_spikes_df.spikes.time_variable_name, 'x']].groupby('aclu')['x'], neuron_id).to_numpy().T for neuron_id in self.included_neuron_IDs] # dataframes split for each ID
        
    
    def curr_ratemap_spiketrains(self, t):
        """ gets the ratemap_spiketrains variable at the time t """
        return [safe_pandas_get_group(self.all_time_filtered_spikes_df.spikes.time_sliced(0, t)[['aclu', self.all_time_filtered_spikes_df.spikes.time_variable_name]].groupby('aclu')[self.all_time_filtered_spikes_df.spikes.time_variable_name], neuron_id).to_numpy() for neuron_id in self.included_neuron_IDs] # dataframes split for each ID
    

    @property
    def ratemap(self):
        """The ratemap property is computed only as needed. Note, this might be the slowest way to get this data, it's like this just for compatibility with the other display functions."""
        # return Ratemap(self.curr_occupancy_weighted_tuning_maps_matrix, spikes_maps=self.curr_spikes_maps_matrix, xbin=self.xbin, ybin=self.ybin, neuron_ids=self.included_neuron_IDs, occupancy=self.curr_seconds_occupancy, neuron_extended_ids=self.frate_filter_fcn(self.all_time_filtered_spikes_df.spikes.neuron_probe_tuple_ids))
        # DO I need neuron_ids=self.frate_filter_fcn(self.included_neuron_IDs)?
        return Ratemap(self.curr_occupancy_weighted_tuning_maps_matrix[self._included_thresh_neurons_indx], spikes_maps=self.curr_spikes_maps_matrix[self._included_thresh_neurons_indx],
                       xbin=self.xbin, ybin=self.ybin, neuron_ids=self.included_neuron_IDs, occupancy=self.curr_seconds_occupancy, neuron_extended_ids=self.frate_filter_fcn(self.all_time_filtered_spikes_df.spikes.neuron_probe_tuple_ids))

        ## Passes self.included_neuron_IDs explicitly


    def __repr__(self):
        return f'{self.__class__.__qualname__.rsplit(">.", 1)[-1]}(spikes_df={self.spikes_df!r}, position={self.position!r}, epochs={self.epochs!r}, config={self.config!r}, position_srate={self.position_srate!r}, setup_on_init={self.setup_on_init!r}, compute_on_init={self.compute_on_init!r}, _save_intermediate_spikes_maps={self._save_intermediate_spikes_maps!r}, _included_thresh_neurons_indx={self._included_thresh_neurons_indx!r}, _peak_frate_filter_function={self._peak_frate_filter_function!r}, ratemap={self.ratemap!r}, _filtered_pos_df={self._filtered_pos_df!r}, _filtered_spikes_df={self._filtered_spikes_df!r}, ndim={self.ndim!r}, xbin={self.xbin!r}, ybin={self.ybin!r}, bin_info={self.bin_info!r})'
    
  
    # ==================================================================================================================== #
    # Initializer                                                                                                          #
    # ==================================================================================================================== #

    def __attrs_post_init__(self):
        """ called after initializer built by `attrs` library. """
        # Perform the primary setup to build the placefield
        if self.setup_on_init:
            self.setup(self.position, self.spikes_df, self.epochs)
            self._reset_after_neuron_index_update()
            ## Interpolate the spikes over positions
            self._filtered_spikes_df['x'] = np.interp(self._filtered_spikes_df[self.spikes_df.spikes.time_variable_name].to_numpy(), self.t, self.x) # self._filtered_spikes_df is empty -> ValueError: array of sample points is empty
            if 'binned_x' not in self._filtered_spikes_df:
                self._filtered_spikes_df['binned_x'] = pd.cut(self._filtered_spikes_df['x'].to_numpy(), bins=self.xbin, include_lowest=True, labels=self.xbin_labels) # same shape as the input data 
        
            if (self.ndim > 1):
                self._filtered_spikes_df['y'] = np.interp(self._filtered_spikes_df[self.spikes_df.spikes.time_variable_name].to_numpy(), self.t, self.y)
                if 'binned_y' not in self._filtered_spikes_df:
                    self._filtered_spikes_df['binned_y'] = pd.cut(self._filtered_spikes_df['y'].to_numpy(), bins=self.ybin, include_lowest=True, labels=self.ybin_labels)
    
            self._setup_time_varying()
            if self.compute_on_init:
                # Ignore self.compute() for time varying
                pass
        else:
            assert (not self.compute_on_init), f"compute_on_init can't be true if setup_on_init isn't true!"

    @classmethod
    def from_config_values(cls, spikes_df: pd.DataFrame, position: Position, epochs: Epoch = None, frate_thresh=1, speed_thresh=5, grid_bin=(1,1), grid_bin_bounds=None, smooth=(1,1), setup_on_init:bool=True, compute_on_init:bool=True):
        """ initialize from the explicitly listed arguments instead of a specified config. """
        return cls(spikes_df=spikes_df, position=position, epochs=epochs,
            config=PlacefieldComputationParameters(speed_thresh=speed_thresh, grid_bin=grid_bin, grid_bin_bounds=grid_bin_bounds, smooth=smooth, frate_thresh=frate_thresh),
            setup_on_init=setup_on_init, compute_on_init=compute_on_init, position_srate=position.sampling_rate, historical_snapshots={}, last_t=np.finfo('float').max)


# ==================================================================================================================== #
# Time-dependent state manipulation                                                                                    #
# ==================================================================================================================== #

    def reset(self):
        """ used to reset the calculations to an initial value. """
        self._setup_time_varying()

    def _reset_after_neuron_index_update(self):
        """ Called after an update to `self._filtered_spikes_df` that will effect the neuron_ids (such as `self.get_by_id(neuron_ids)`.
        Uses only `self._filtered_spikes_df` to rebuild the other various properties.
         
        """
        self._filtered_spikes_df, _reverse_cellID_index_map = self._filtered_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()
        self.fragile_linear_neuron_IDXs = np.unique(self._filtered_spikes_df.fragile_linear_neuron_IDX) # array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63])
        self.n_fragile_linear_neuron_IDXs = len(self.fragile_linear_neuron_IDXs)
        self._included_thresh_neurons_indx = np.arange(self.n_fragile_linear_neuron_IDXs)
        self._peak_frate_filter_function = lambda list_: [list_[_] for _ in self._included_thresh_neurons_indx] # filter_function: takes any list of length n_neurons (original number of neurons) and returns only the elements that met the firing rate criteria
        

    def _setup_time_varying(self):
        """ Initialize for the 0th timestamp 
        Depends on `self.n_fragile_linear_neuron_IDXs` and other values setup in `self._reset_after_neuron_index_update()`
        """
        if not hasattr(self, '_included_thresh_neurons_indx'):
            self._reset_after_neuron_index_update()

        dims_coord_tuple = self.dims_coord_tuple

        self.curr_spikes_maps_matrix = np.zeros((self.n_fragile_linear_neuron_IDXs, *dims_coord_tuple), dtype=int) # create an initially zero occupancy map
        self.curr_smoothed_spikes_maps_matrix = None
        self.curr_num_pos_samples_occupancy_map = np.zeros(dims_coord_tuple, dtype=int) # create an initially zero occupancy map
        self.curr_num_pos_samples_smoothed_occupancy_map = None
        self.last_t = 0.0
        self.curr_seconds_occupancy = np.zeros(dims_coord_tuple, dtype=float)
        self.curr_normalized_occupancy = self.curr_seconds_occupancy.copy()
        self.curr_occupancy_weighted_tuning_maps_matrix = np.zeros((self.n_fragile_linear_neuron_IDXs, *dims_coord_tuple), dtype=float) # will have units of # spikes/sec
        self.historical_snapshots = OrderedDict({})


    def step(self, num_seconds_to_advance, should_snapshot=False):
        """ advance the computed time by a fixed number of seconds. """
        next_t = self.last_t + num_seconds_to_advance # add one second
        self.update(next_t, should_snapshot=should_snapshot)
        return next_t
    
    
    def update(self, t, start_relative_t:bool=False, should_snapshot=False):
        """ updates all variables to the latest versions """
        if start_relative_t:
            # if start_relative_t then t is assumed to be specified relative to the earliest_valid_time
            t = self.earliest_valid_time + t # add self.earliest_valid_time to the relative_t value to get the absolute t value
        
        if self.is_additive_mode and (self.last_t > t):
            print(f'WARNING: update(t: {t}) called with t < self.last_t ({self.last_t}! Skipping.')
        else:
            # Otherwise update to this t.
            # with np.errstate(divide='ignore', invalid='ignore'):
            with np.errstate(divide='warn', invalid='raise'):
                if self.is_additive_mode:
                    self._minimal_additive_update(t)
                    self._display_additive_update(t)
                else:
                    # non-additive mode, recompute:
                    self.complete_time_range_computation(0.0, t, should_snapshot=False) # should_snapshot=False during this call because we snapshot ourselves in the next step if should_snapshot==True
                    
                if should_snapshot:
                    self.snapshot()
                    

    def batch_snapshotting(self, combined_records_list, reset_at_start:bool = True, debug_print=False) -> Dict[SnapshotTimestamp, PlacefieldSnapshot]:
        """ Updates sequentially, snapshotting each time.

        combined_records_list: can be an Epoch object, a epoch-formatted pd.DataFrame, or a series of tuples to convert into combined_records_list

        Usage:
            laps_df = deepcopy(global_any_laps_epochs_obj.to_dataframe())
            laps_df['epoch_type'] = 'lap'
            laps_df['interval_type_id'] = 666 # 'inter' # vs, 'intra'
            # laps_records_list = list(laps_df[['epoch_type','start','stop','lap_id']].itertuples(index=False, name='lap'))
            laps_df

        """
        if reset_at_start:
            self.reset() ## Reset completely to start

        if isinstance(combined_records_list, Epoch):
            # convert to pd.DataFrame:
            combined_records_list = combined_records_list.to_dataframe()

        if isinstance(combined_records_list, pd.DataFrame):
            combined_records_list = list(combined_records_list[['epoch_type','start','stop','interval_type_id']].itertuples(index=False, name='combined_epochs')) # len(intra_lap_interval_records) # 75

        ## dataframe:
        # initial_start_t = combined_records_list.start[0]

        # Tuples list:
        initial_start_t = float(combined_records_list[0][1])
        self.update(t=initial_start_t, start_relative_t=True, should_snapshot=True)

        ## Use the combined records list (which is a list of tuples) to update the self:
        for epoch_type, start_t, stop_t, item_id in combined_records_list: ## tuple list
            if debug_print:
                print(f'{epoch_type}, start_t: {start_t}, stop_t: {stop_t}, item_id: {item_id}')
            self.update(t=float(stop_t), start_relative_t=True, should_snapshot=True) # advance the self to the current start time
        if debug_print:
            print(f'done.')
        
        if debug_print:
            print(f'batch_snapshotting(...): took {len(self.historical_snapshots)} snapshots.') # 150 snapshots
            print(f'\t of {len(combined_records_list)} combined records') # 149 combined records
            # print_object_memory_usage(self) # object size: 204.464939 MB for 150 snapshots of a 1D track
        return self.historical_snapshots

    def compute(self):
        raise NotImplementedError('compute() is not implemented for PlacefieldTracker. Use batch_snapshotting() instead.')

    # ==================================================================================================================== #
    # Snapshotting and state restoration                                                                                   #
    # ==================================================================================================================== #
    
    def snapshot(self):
        """ takes a snapshot of the current values at this time."""    
        # Add this entry to the historical snapshot dict:
        self.historical_snapshots[self.last_t] = PlacefieldSnapshot(num_position_samples_occupancy=self.curr_num_pos_samples_occupancy_map.copy(), num_position_samples_smoothed_occupancy=copy_if_not_none(self.curr_num_pos_samples_smoothed_occupancy_map), seconds_occupancy=self.curr_seconds_occupancy.copy(), normalized_occupancy=copy_if_not_none(self.curr_normalized_occupancy),
            spikes_maps_matrix=self.curr_spikes_maps_matrix.copy(), smoothed_spikes_maps_matrix=copy_if_not_none(self.curr_smoothed_spikes_maps_matrix),
            occupancy_weighted_tuning_maps_matrix=self.curr_occupancy_weighted_tuning_maps_matrix.copy())
        return (self.last_t, self.historical_snapshots[self.last_t]) # return the (snapshot_time, snapshot_data) pair

    def _apply_snapshot_data(self, snapshot_t, snapshot_data):
        """ applys the snapshot_data to replace the current state of this object (except for historical_snapshots) """
        ## PlacefieldSnapshot class version:
        self.curr_spikes_maps_matrix = snapshot_data.spikes_maps_matrix
        self.curr_smoothed_spikes_maps_matrix = snapshot_data.smoothed_spikes_maps_matrix
        self.curr_num_pos_samples_occupancy_map = snapshot_data.num_position_samples_occupancy
        self.curr_num_pos_samples_smoothed_occupancy_map = snapshot_data.num_position_samples_smoothed_occupancy

        self.curr_seconds_occupancy = snapshot_data.seconds_occupancy
        self.curr_normalized_occupancy = snapshot_data.normalized_occupancy
        self.curr_occupancy_weighted_tuning_maps_matrix = snapshot_data.occupancy_weighted_tuning_maps_matrix
        # Common:
        self.last_t = snapshot_t
        
    def restore_from_snapshot(self, snapshot_t):
        """ restores the current state to that of a historic snapshot indexed by the time snapshot_t """
        snapshot_data = self.historical_snapshots[snapshot_t]
        self._apply_snapshot_data(snapshot_t, snapshot_data)
        
    def to_dict(self):
        # ['spikes_df', 'position', 'epochs', 'config', 'position_srate', 'setup_on_init', 'compute_on_init', '_save_intermediate_spikes_maps', '_included_thresh_neurons_indx', '_peak_frate_filter_function', '_ratemap', '_ratemap_spiketrains', '_ratemap_spiketrains_pos', '_filtered_pos_df', '_filtered_spikes_df', 'ndim', 'xbin', 'ybin', 'bin_info', 'last_t', 'historical_snapshots', 'fragile_linear_neuron_IDXs', 'n_fragile_linear_neuron_IDXs']

        included_key_names = ['spikes_df', 'position', 'epochs', 'config', 'position_srate', 'setup_on_init', 'compute_on_init',
          '_save_intermediate_spikes_maps', '_filtered_pos_df', '_filtered_spikes_df',
           'ndim', 'xbin', 'ybin', 'bin_info', 'last_t', 'historical_snapshots', 'fragile_linear_neuron_IDXs', 'n_fragile_linear_neuron_IDXs']

        # print(f'to_dict(...): {list(self.__dict__.keys())}')
        curr_snapshot_time, curr_snapshot_data = self.snapshot() # take a snapshot of the current state
        # self._setup_time_varying() # reset completely before saving. Throw out everything
        # Excluded from serialization: ['_included_thresh_neurons_indx', '_peak_frate_filter_function']
        # filter_fn = filters.exclude(fields(PfND)._included_thresh_neurons_indx, int)
        # [an_attr.name for an_attr in self.__attrs_attrs__]
        # print(f'{[an_attr.name for an_attr in self.__attrs_attrs__]}')

        # filter_fn = lambda attr, value: attr.name not in ["_included_thresh_neurons_indx", "_peak_frate_filter_function", "_ratemap", "_ratemap_spiketrains", "_ratemap_spiketrains_pos"]
        # if not hasattr(self, '_included_thresh_neurons_indx'):
        #     self.reset()
        # filter_fn = lambda attr, value: attr.name in included_key_names
        # _out = asdict(self, recurse=False, filter=filter_fn) # serialize using attrs.asdict but exclude the listed properties
        _out = {k:v for k,v in self.__dict__.items() if k in included_key_names}
        return _out
        

    ## For serialization/pickling:
    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, state):
        """ assumes state is a dict generated by calling self.__getstate__() previously"""        
        # print(f'__setstate__(self: {self}, state: {state})')
        # print(f'__setstate__(...): {list(self.__dict__.keys())}')
        self.__dict__ = state # set the dict
        self._save_intermediate_spikes_maps = True # False is not yet implemented

        # Convert back to PlacefieldSnapshot objects:
        # self.historical_snapshots = {k:PlacefieldSnapshot.from_dict(dict_v) for k, dict_v in self.historical_snapshots.items()}

        try:
            self.restore_from_snapshot(self.last_t) # after restoring the object's __dict__ from state, self.historical_snapshots is populated and the last entry can be used to restore all the last-computed properties. Note this requires at least one snapshot.
        except (AttributeError, KeyError) as e:
            print(f'WARNING: failed to unpickle PfND_TimeDependent - just resetting. Snapshots will not be loaded.')
            self.reset()

        except Exception as e:
            print(f'ERROR: unhandled exception: {e}')
            raise e
        
        # I think this is okay:
        if not hasattr(self, '_included_thresh_neurons_indx'):
            self._reset_after_neuron_index_update()
            
        # Rebuild the filter function from self._included_thresh_neurons_indx
        # self._included_thresh_neurons_indx = np.arange(self.n_fragile_linear_neuron_IDXs)
        # self._peak_frate_filter_function = lambda list_: [list_[_] for _ in self._included_thresh_neurons_indx] # filter_function: takes any list of length n_neurons (original number of neurons) and returns only the elements that met the firing rate criteria        
        
        
    # ==================================================================================================================== #
    # Common Methods                                                                                                       #
    # ==================================================================================================================== #
    # for NeuronUnitSlicableObjectProtocol:
    def get_by_id(self, ids):
        """Implementors return a copy of themselves with neuron_ids equal to ids
            Needs to update: spikes_maps_matrix=self.curr_spikes_maps_matrix.copy(), smoothed_spikes_maps_matrix=copy_if_not_none(self.curr_smoothed_spikes_maps_matrix),
            occupancy_weighted_tuning_maps_matrix=self.curr_occupancy_weighted_tuning_maps_matrix.copy()
        """
        ## Method 0: Completely new initialization of the PfND_TimeDependent:
        # copy_pf = PfND_TimeDependent(spikes_df=deepcopy(self._filtered_spikes_df[np.isin(self._filtered_spikes_df.aclu, ids)]), position=self.position, epochs=self.epochs, config=self.config)

        # ## Method 1: Exhaustive (inefficient) rebuild from scratch:
        # copy_pf = deepcopy(self)
        # # filter the spikes_df:
        # copy_pf._filtered_spikes_df = copy_pf._filtered_spikes_df[np.isin(copy_pf._filtered_spikes_df.aclu, ids)] # refers to `self.all_time_filtered_spikes_df`
        # # diverge from the non-time-dependent placefields here (which call self.compute() here):
        # self._reset_after_neuron_index_update()
        # self._setup_time_varying()


        ## Method 2: Updating `copy_pf._included_thresh_neurons_indx`:
        copy_pf = deepcopy(self)
        # Find the neuron_IDs that are included in the active_pf_2D for filtering the active_pf_2D_dt's results:
        is_included_neuron = np.isin(copy_pf.included_neuron_IDs, ids)
        included_neuron_IDXs = copy_pf._included_thresh_neurons_indx[is_included_neuron]
        # included_neuron_ids = copy_pf.included_neuron_IDs[is_included_neuron]

        # #NOTE: to reset and include all neurons:
        # copy_pf._included_thresh_neurons_indx = np.arange(copy_pf.n_fragile_linear_neuron_IDXs)

        copy_pf._included_thresh_neurons_indx = included_neuron_IDXs
        copy_pf._peak_frate_filter_function = lambda list_: [list_[_] for _ in copy_pf._included_thresh_neurons_indx]

        ## Backup the historical snapshots and slice them by the indicies to avoid recomputation:
        copy_pf.historical_snapshots = OrderedDict({k:v.get_by_IDX(included_neuron_IDXs) for k, v in copy_pf.historical_snapshots.items()})

        try:
            copy_pf.restore_from_snapshot(copy_pf.last_t) # after slicing the copy_pf's historical_snapshots, copy_pf.historical_snapshots is populated and the last entry can be used to restore all the last-computed properties. Note this requires at least one snapshot.
        except (AttributeError, KeyError) as e:
            # No previous snapshots, just reset:
            self.reset()
        except Exception as e:
            print(f'ERROR: unhandled exception: {e}')
            raise e

        # assert (copy_pf.ratemap.spikes_maps == active_pf_2D.ratemap.spikes_maps).all(), f"copy_pf.ratemap.spikes_maps: {copy_pf.ratemap.spikes_maps}\nactive_pf_2D.ratemap.spikes_maps: {active_pf_2D.ratemap.spikes_maps}"
        return copy_pf


    def conform_to_position_bins(self, target_pf1D, force_recompute=False):
        """ Allow overriding PfND's bins:
            # 2022-12-09 - We want to be able to have both long/short track placefields have the same spatial bins.
            This function standardizes the short pf1D's xbins to the same ones as the long_pf1D, and then recalculates it.
            Usage:
                short_pf1D, did_update_bins = short_pf1D.conform_to_position_bins(long_pf1D)
        """
        did_update_bins = False
        if force_recompute or (len(self.xbin) < len(target_pf1D.xbin)) or ((self.ndim > 1) and (len(self.ybin) < len(target_pf1D.ybin))):
            print(f'self will be re-binned to match target_pf1D...')
            # bak_self = deepcopy(self) # Backup the original first
            xbin, ybin, bin_info, grid_bin = target_pf1D.xbin, target_pf1D.ybin, target_pf1D.bin_info, target_pf1D.config.grid_bin
            ## Apply to the short dataframe:
            self.xbin, self.ybin, self.bin_info, self.config.grid_bin = xbin, ybin, bin_info, grid_bin
            ## Updates (replacing) the 'binned_x' (and if 2D 'binned_y') columns to the position dataframe:
            self._filtered_pos_df, _, _, _ = PfND.build_position_df_discretized_binned_positions(self._filtered_pos_df, self.config, xbin_values=self.xbin, ybin_values=self.ybin, debug_print=False) # Finishes setup
            # self.compute() # does compute
            self._reset_after_neuron_index_update()
            self._setup_time_varying()

            raise NotImplementedError # Not yet implemented for Pf1D_dt (time-dependent version) although it might be quite simple and similar to the non-time dependent implementation (copied here).

            print(f'done.') ## Successfully re-bins pf1D:
            did_update_bins = True # set the update flag
        else:
            # No changes needed:
            did_update_bins = False

        return self, did_update_bins



    @classmethod
    def compute_occupancy_weighted_tuning_map(cls, curr_seconds_occupancy_map, curr_spikes_maps_matrix, debug_print=False):
        """ Given the curr_occupancy_map and curr_spikes_maps_matrix for this timestamp, returns the occupancy weighted tuning map
        Inputs:
        # curr_seconds_occupancy_map: note that this is the occupancy map in seconds, not the raw counts
        """
        ## Simple occupancy shuffle:
        # occupancy_weighted_tuning_maps_matrix = curr_spikes_maps_matrix / curr_seconds_occupancy_map # dividing by positions with zero occupancy result in a warning and the result being set to NaN. Set to 0.0 instead.
        # occupancy_weighted_tuning_maps_matrix = np.nan_to_num(occupancy_weighted_tuning_maps_matrix, copy=True, nan=0.0) # set any NaN values to 0.0, as this is the correct weighted occupancy
        
        ## More advanced occumancy shuffle:
        curr_seconds_occupancy_map[curr_seconds_occupancy_map == 0.0] = np.nan # pre-set the zero occupancy locations to NaN to avoid a warning in the next step. They'll be replaced with zero afterwards anyway
        occupancy_weighted_tuning_maps_matrix = curr_spikes_maps_matrix / curr_seconds_occupancy_map # dividing by positions with zero occupancy result in a warning and the result being set to NaN. Set to 0.0 instead.
        occupancy_weighted_tuning_maps_matrix = np.nan_to_num(occupancy_weighted_tuning_maps_matrix, copy=True, nan=0.0) # set any NaN values to 0.0, as this is the correct weighted occupancy
        curr_seconds_occupancy_map[np.isnan(curr_seconds_occupancy_map)] = 0.0 # restore these entries back to zero
        return occupancy_weighted_tuning_maps_matrix
    

    # ==================================================================================================================== #
    # Additive update static methods:                                                                                      #
    # ==================================================================================================================== #
    def _minimal_additive_update(self, t):
        """ Updates the current_occupancy_map, curr_spikes_maps_matrix
        # t: the "current time" for which to build the best possible placefields
        
        Updates:
            self.curr_num_pos_samples_occupancy_map
            self.curr_spikes_maps_matrix
            self.last_t
        """
        # Post Initialization Update
        curr_t, self.curr_num_pos_samples_occupancy_map = PfND_TimeDependent.update_occupancy_map(self.last_t, self.curr_num_pos_samples_occupancy_map, t, self.all_time_filtered_pos_df)
        curr_t, self.curr_spikes_maps_matrix = PfND_TimeDependent.update_spikes_map(self.last_t, self.curr_spikes_maps_matrix, t, self.all_time_filtered_spikes_df)
        self.last_t = curr_t
    
    def _display_additive_update(self, t):
        """ updates the extended variables:


        TODO: MAKE_1D: remove 'binned_y' references, Refactor for 1D support if deciding to continue additive update mode.

        Using:
            self.position_srate
            self.curr_num_pos_samples_occupancy_map
            self.curr_spikes_maps_matrix
            
        Updates:
            self.curr_raw_smoothed_occupancy_map
            self.curr_smoothed_spikes_maps_matrix
            self.curr_seconds_occupancy
            self.curr_normalized_occupancy
            self.curr_occupancy_weighted_tuning_maps_matrix
        
        """
        # Smooth if needed: OH NO! Don't smooth the occupancy map!!
        ## Occupancy:
        # NOTE: usually don't smooth occupancy. Unless self.should_smooth_spatial_occupancy_map is True, and in that case use the same smoothing values that are used to smooth the firing rates
        if (self.should_smooth_spatial_occupancy_map and (self.smooth is not None) and ((self.smooth[0] > 0.0) & (self.smooth[1] > 0.0))): 
            # Smooth the occupancy map:
            self.curr_num_pos_samples_smoothed_occupancy_map = gaussian_filter(self.curr_num_pos_samples_occupancy_map, sigma=(self.smooth[1], self.smooth[0])) # 2d gaussian filter
            self.curr_seconds_occupancy, self.curr_normalized_occupancy = _normalized_occupancy(self.curr_num_pos_samples_smoothed_occupancy_map, position_srate=self.position_srate)
        else:
            self.curr_seconds_occupancy, self.curr_normalized_occupancy = _normalized_occupancy(self.curr_num_pos_samples_occupancy_map, position_srate=self.position_srate)
            
        ## Spikes:
        if ((self.smooth is not None) and ((self.smooth[0] > 0.0) & (self.smooth[1] > 0.0))): 
            # Smooth the firing map:
            self.curr_smoothed_spikes_maps_matrix = gaussian_filter(self.curr_spikes_maps_matrix, sigma=(0, self.smooth[1], self.smooth[0])) # 2d gaussian filter
            self.curr_occupancy_weighted_tuning_maps_matrix = PfND_TimeDependent.compute_occupancy_weighted_tuning_map(self.curr_seconds_occupancy, self.curr_smoothed_spikes_maps_matrix)

        else:
            self.curr_occupancy_weighted_tuning_maps_matrix = PfND_TimeDependent.compute_occupancy_weighted_tuning_map(self.curr_seconds_occupancy, self.curr_spikes_maps_matrix)
    
    @classmethod
    def update_occupancy_map(cls, last_t, last_occupancy_matrix, t, active_pos_df, debug_print=False):
        """ Given the last_occupancy_matrix computed at time last_t, determines the additional positional occupancy from active_pos_df and adds them producing an updated version
        Inputs:
            t: the "current time" for which to build the best possible placefields

        TODO: MAKE_1D: remove 'binned_y' references
        """
        active_current_pos_df = active_pos_df.position.time_sliced(last_t, t) # [active_pos_df.position.time<t]
        # Compute the updated counts:
        current_bin_counts = active_current_pos_df.value_counts(subset=['binned_x', 'binned_y'], normalize=False, sort=False, ascending=True, dropna=True) # dropna=True
        # current_bin_counts: a series with a MultiIndex index for each bin that has nonzero counts
        # binned_x  binned_y
        # 2         12           2
        # 3         11           1
        #           12          30
        # ...
        # 57        9            5
        #           12           1
        #           13           4
        #           14           1
        # 58        9            3
        #           10           2
        #           12           4
        # Length: 247, dtype: int64
        if debug_print:
            print(f'np.shape(current_bin_counts): {np.shape(current_bin_counts)}') # (247,)
        for (xbin_label, ybin_label), count in current_bin_counts.iteritems():
            if debug_print:
                print(f'xbin_label: {xbin_label}, ybin_label: {ybin_label}, count: {count}')
            
            # last_occupancy_matrix[xbin_label, ybin_label] += count
            try:
                last_occupancy_matrix[xbin_label-1, ybin_label-1] += count
                # last_occupancy_matrix[xbin_label, ybin_label] += count
            except IndexError as e:
                print(f'e: {e}\n active_current_pos_df: {np.shape(active_current_pos_df)}, current_bin_counts: {np.shape(current_bin_counts)}\n last_occupancy_matrix: {np.shape(last_occupancy_matrix)}\n count: {count}')
                raise e
        return t, last_occupancy_matrix

    @classmethod
    def update_spikes_map(cls, last_t, last_spikes_maps_matrix, t, active_spike_df, debug_print=False):
        """ Given the last_spikes_maps_matrix computed at time last_t, determines the additional updates (spikes) from active_spike_df and adds them producing an updated version
        Inputs:
        # t: the "current time" for which to build the best possible placefields

        TODO: MAKE_1D: remove 'binned_y' references
        """
        active_current_spike_df = active_spike_df.spikes.time_sliced(last_t, t)
        
        # Compute the updated counts:
        current_spike_per_unit_per_bin_counts = active_current_spike_df.value_counts(subset=['fragile_linear_neuron_IDX', 'binned_x', 'binned_y'], normalize=False, sort=False, ascending=True, dropna=True) # dropna=True
        
        if debug_print:
            print(f'np.shape(current_spike_per_unit_per_bin_counts): {np.shape(current_spike_per_unit_per_bin_counts)}') # (247,)
        for (fragile_linear_neuron_IDX, xbin_label, ybin_label), count in current_spike_per_unit_per_bin_counts.iteritems():
            if debug_print:
                print(f'fragile_linear_neuron_IDX: {fragile_linear_neuron_IDX}, xbin_label: {xbin_label}, ybin_label: {ybin_label}, count: {count}')
            try:
                last_spikes_maps_matrix[fragile_linear_neuron_IDX, xbin_label-1, ybin_label-1] += count
            except IndexError as e:
                print(f'e: {e}\n active_current_spike_df: {np.shape(active_current_spike_df)}, current_spike_per_unit_per_bin_counts: {np.shape(current_spike_per_unit_per_bin_counts)}\n last_spikes_maps_matrix: {np.shape(last_spikes_maps_matrix)}\n count: {count}')
                print(f' last_spikes_maps_matrix[fragile_linear_neuron_IDX: {fragile_linear_neuron_IDX}, (xbin_label-1): {xbin_label-1}, (ybin_label-1): {ybin_label-1}] += count: {count}')
                raise e
        return t, last_spikes_maps_matrix

    
    # ==================================================================================================================== #
    # 2022-08-02 - New Simple Time-Dependent Placefield Overhaul                                                           #
    # ==================================================================================================================== #
    # Idea: use simple dataframes and operations on them to easily get the placefield results for a given time range.
        
    def complete_time_range_computation(self, start_time, end_time, assign_results_to_member_variables=True, should_snapshot=True):
        """ recomputes the entire time period from start_time to end_time with few other assumptions """
        computed_out_results = PfND_TimeDependent.perform_time_range_computation(self.all_time_filtered_spikes_df, self.all_time_filtered_pos_df, position_srate=self.position_srate,
                                                             xbin=self.xbin, ybin=self.ybin,
                                                             start_time=start_time, end_time=end_time,
                                                             included_neuron_IDs=self.included_neuron_IDs, active_computation_config=self.config, override_smooth=self.smooth) # previously active_computation_config=None

        if assign_results_to_member_variables:
            # Should replace current time variables with those from this most recently added snapshot:
            self._apply_snapshot_data(end_time, computed_out_results)
            if should_snapshot:
                self.snapshot()

        else:
            assert not should_snapshot, "should_snapshot must be False if assign_results_to_member_variables is False, but instead should_snapshot == True!"
            # if assign_results_to_member_variables is False, don't update any of the member variables and just return the wrapped result.
            return computed_out_results

    @classmethod    
    def perform_time_range_computation(cls, spikes_df, pos_df, position_srate, xbin, ybin, start_time, end_time, included_neuron_IDs, active_computation_config=None, override_smooth=None):
        """ This method performs complete calculation witihin a single function. 
        
        Inputs:
        
        Note that active_computation_config can be None IFF xbin, ybin, and override_smooth are provided.
        
        Usage:
            # active_pf_spikes_df = deepcopy(sess.spikes_df)
            # active_pf_pos_df = deepcopy(sess.position.to_dataframe())
            # position_srate = sess.position_sampling_rate
            # active_computation_config = curr_active_config.computation_config
            # out_dict = PfND_TimeDependent.perform_time_range_computation(spikes_df, pos_df, position_srate, xbin, ybin, start_time, end_time, included_neuron_IDs, active_computation_config)
            out_dict = PfND_TimeDependent.perform_time_range_computation(sess.spikes_df, sess.position.to_dataframe(), position_srate=sess.position_sampling_rate,
                                                             xbin=active_pf_2D.xbin, ybin=active_pf_2D.ybin,
                                                             start_time=_test_arbitrary_start_time, end_time=_test_arbitrary_end_time,
                                                             included_neuron_IDs=active_pf_2D.included_neuron_IDs, active_computation_config=curr_active_config.computation_config, override_smooth=(0.0, 0.0))
                                                             
        TODO: MAKE_1D: remove 'binned_y' references

        """
        def _build_bin_pos_counts(active_pf_pos_df, xbin_values=None, ybin_values=None, active_computation_config=active_computation_config):
            ## This version was brought in from PfND.perform_time_range_computation(...):
            # If xbin_values is not None and ybin_values is None, assume 1D
            # if xbin_values is not None and ybin_values is None:
            if 'y' not in active_pf_pos_df.columns:
                # Assume 1D:
                ndim = 1
                pos_col_names = ('x',)
                binned_col_names = ('binned_x',)
                bin_values = (xbin_values,)
            else:
                # otherwise assume 2D:
                ndim = 2
                pos_col_names = ('x', 'y')
                binned_col_names = ('binned_x', 'binned_y')
                bin_values = (xbin_values, ybin_values)

            # bin the dataframe's x and y positions into bins, with binned_x and binned_y containing the index of the bin that the given position is contained within.
            active_pf_pos_df, out_bins, bin_info = build_df_discretized_binned_position_columns(active_pf_pos_df.copy(), bin_values=bin_values, position_column_names=pos_col_names, binned_column_names=binned_col_names, active_computation_config=active_computation_config, force_recompute=False, debug_print=False)
            
            if ndim == 1:
                # Assume 1D:
                xbin = out_bins[0]
                ybin = None
                n_xbins = len(xbin) - 1 # the -1 is to get the counts for the centers only
                num_position_samples_occupancy = np.zeros((n_xbins, ), dtype=int) # create an initially zero position occupancy map. # TODO: should it be NaN or np.masked where we haven't visisted at all yet?
                curr_counts_df = active_pf_pos_df.value_counts(subset=['binned_x'], normalize=False, sort=False, ascending=True, dropna=True).to_frame(name='counts').reset_index()
                xbin_indicies = curr_counts_df.binned_x.values.astype('int') - 1
                num_position_samples_occupancy[xbin_indicies] = curr_counts_df.counts.values # Assignment

            else:            
                (xbin, ybin) = out_bins
                n_xbins = len(xbin) - 1 # the -1 is to get the counts for the centers only
                n_ybins = len(ybin) - 1 # the -1 is to get the counts for the centers only
                num_position_samples_occupancy = np.zeros((n_xbins, n_ybins), dtype=int) # create an initially zero position occupancy map. # TODO: should it be NaN or np.masked where we haven't visisted at all yet?
                curr_counts_df = active_pf_pos_df.value_counts(subset=['binned_x', 'binned_y'], normalize=False, sort=False, ascending=True, dropna=True).to_frame(name='counts').reset_index()
                xbin_indicies = curr_counts_df.binned_x.values.astype('int') - 1
                ybin_indicies = curr_counts_df.binned_y.values.astype('int') - 1
                num_position_samples_occupancy[xbin_indicies, ybin_indicies] = curr_counts_df.counts.values # Assignment
                # num_position_samples_occupancy[xbin_indicies, ybin_indicies] += curr_counts_df.counts.values # Additive

            return curr_counts_df, num_position_samples_occupancy

        def _build_bin_spike_counts(active_pf_spikes_df, neuron_ids=None, xbin_values=None, ybin_values=None, active_computation_config=active_computation_config):
            ## This version was brought in from PfND.perform_time_range_computation(...):
            # If xbin_values is not None and ybin_values is None, assume 1D
            # if xbin_values is not None and ybin_values is None:

            assert np.all(active_pf_spikes_df.aclu.isin(neuron_ids)), f"active_pf_spikes_df.aclu: {active_pf_spikes_df.aclu.unique()}, neuron_ids: {neuron_ids}"
            # # Cut spikes_df down to only the neuron_IDs that appear at least in one decoder:
            # active_pf_spikes_df = active_pf_spikes_df.spikes.sliced_by_neuron_id(neuron_ids)
            # active_pf_spikes_df, active_aclu_to_fragile_linear_neuron_IDX_dict = active_pf_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()
            
            if ('y' not in active_pf_spikes_df.columns) or ((xbin_values is not None) and (ybin_values is None)):
                # Assume 1D:
                ndim = 1
                pos_col_names = ('x',)
                binned_col_names = ('binned_x',)
                bin_values = (xbin_values,)
            else:
                # otherwise assume 2D:
                assert ybin_values is not None
                ndim = 2
                pos_col_names = ('x', 'y')
                binned_col_names = ('binned_x', 'binned_y')
                bin_values = (xbin_values, ybin_values)

            # bin the dataframe's x and y positions into bins, with binned_x and binned_y containing the index of the bin that the given position is contained within.
            active_pf_spikes_df, out_bins, bin_info = build_df_discretized_binned_position_columns(active_pf_spikes_df.copy(), bin_values=bin_values, binned_column_names=binned_col_names, position_column_names=pos_col_names, active_computation_config=active_computation_config, force_recompute=False, debug_print=False) # removed , position_column_names=pos_col_names, binned_column_names=binned_col_names
                    
            if ndim == 1:
                # Assume 1D:
                xbin = out_bins[0]
                ybin = None
                n_xbins = len(xbin) - 1 # the -1 is to get the counts for the centers only
                n_neuron_ids = len(neuron_ids)
                curr_spikes_map_dict = {neuron_id:np.zeros((n_xbins, ), dtype=int) for neuron_id in neuron_ids} # create an initially zero spikes map, one for each possible neruon_id, even if there aren't spikes for that neuron yet
                curr_counts_df = active_pf_spikes_df.value_counts(subset=['aclu', 'binned_x'], sort=False).to_frame(name='counts').reset_index()
                for name, group in curr_counts_df.groupby('aclu'):
                    xbin_indicies = group.binned_x.values.astype('int') - 1
                    # curr_spikes_map_dict[name][xbin_indicies, ybin_indicies] += group.counts.values # Additive
                    curr_spikes_map_dict[name][xbin_indicies] = group.counts.values # Assignment

            else:
                # Regular 2D:
                (xbin, ybin) = out_bins
                n_xbins = len(xbin) - 1 # the -1 is to get the counts for the centers only
                n_ybins = len(ybin) - 1 # the -1 is to get the counts for the centers only
                n_neuron_ids = len(neuron_ids)
                curr_spikes_map_dict = {neuron_id:np.zeros((n_xbins, n_ybins), dtype=int) for neuron_id in neuron_ids} # create an initially zero spikes map, one for each possible neruon_id, even if there aren't spikes for that neuron yet
                curr_counts_df = active_pf_spikes_df.value_counts(subset=['aclu', 'binned_x', 'binned_y'], sort=False).to_frame(name='counts').reset_index()
                for name, group in curr_counts_df.groupby('aclu'):
                    xbin_indicies = group.binned_x.values.astype('int') - 1
                    ybin_indicies = group.binned_y.values.astype('int') - 1
                    # curr_spikes_map_dict[name][xbin_indicies, ybin_indicies] += group.counts.values # Additive
                    curr_spikes_map_dict[name][xbin_indicies, ybin_indicies] = group.counts.values # Assignment

            return curr_counts_df, curr_spikes_map_dict


        ## Only the spikes_df and pos_df are required, and are not altered by the analyses:
        active_pf_spikes_df = deepcopy(spikes_df)
        active_pf_pos_df = deepcopy(pos_df)

        ## NEEDS:
        # position_srate, xbin, ybin, included_neuron_IDs, active_computation_config
       
        if override_smooth is not None:
            smooth = override_smooth
        else:
            smooth = active_computation_config.pf_params.smooth

        ## Test arbitrarily slicing by first _test_arbitrary_end_time seconds
        active_pf_spikes_df = active_pf_spikes_df.spikes.time_sliced(start_time, end_time)
        active_pf_pos_df = active_pf_pos_df.position.time_sliced(start_time, end_time)
        
        active_pf_spikes_df = active_pf_spikes_df.spikes.sliced_by_neuron_id(included_neuron_IDs)
        active_pf_spikes_df, active_aclu_to_fragile_linear_neuron_IDX_dict = active_pf_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()


        counts_df, num_position_samples_occupancy = _build_bin_pos_counts(active_pf_pos_df, xbin_values=xbin, ybin_values=ybin, active_computation_config=active_computation_config)
        spikes_counts_df, spikes_map_dict = _build_bin_spike_counts(active_pf_spikes_df, neuron_ids=included_neuron_IDs, xbin_values=xbin, ybin_values=ybin, active_computation_config=active_computation_config)
        # Convert curr_spikes_map_dict from a dict into the expected 3-dim matrix:
        spikes_maps_matrix = np.array([spikes_matrix for an_aclu, spikes_matrix in spikes_map_dict.items()])  # spikes_maps_matrix.shape # (40, 64, 29) (len(curr_spikes_map_dict), n_xbins, n_ybins)

        # active_computation_config.grid_bin, smooth=active_computation_config.smooth
        seconds_occupancy, normalized_occupancy = _normalized_occupancy(num_position_samples_occupancy, position_srate=position_srate)

        ## TODO: Copy the 1D Gaussian filter code here. Currently it always does 2D:
        if (ybin is None) or ('y' not in active_pf_spikes_df.columns):
            # Assume 1D:
            ndim = 1
            smooth_criteria_fn = lambda smooth: (smooth[0] > 0.0)
            occupancy_smooth_gaussian_filter_fn = lambda x, smooth: gaussian_filter1d(x, sigma=smooth[0]) 
            spikes_maps_smooth_gaussian_filter_fn = lambda x, smooth: gaussian_filter1d(x, sigma=smooth[0]) 
        else:
            # otherwise assume 2D:
            ndim = 2
            smooth_criteria_fn = lambda smooth: ((smooth[0] > 0.0) & (smooth[1] > 0.0))
            occupancy_smooth_gaussian_filter_fn = lambda x, smooth: gaussian_filter(x, sigma=(smooth[1], smooth[0])) 
            spikes_maps_smooth_gaussian_filter_fn = lambda x, smooth: gaussian_filter(x, sigma=(0, smooth[1], smooth[0])) 

        # Smooth the final tuning map if needed and valid smooth parameter. Default FALSE.
        if (cls.should_smooth_spatial_occupancy_map and (smooth is not None) and smooth_criteria_fn(smooth)):
            num_position_samples_occupancy = occupancy_smooth_gaussian_filter_fn(num_position_samples_occupancy, smooth) 
            seconds_occupancy = occupancy_smooth_gaussian_filter_fn(seconds_occupancy, smooth)

        # Smooth the spikes maps if needed and valid smooth parameter. Default False.
        if (cls.should_smooth_spikes_map and (smooth is not None) and smooth_criteria_fn(smooth)): 
            smoothed_spikes_maps_matrix = spikes_maps_smooth_gaussian_filter_fn(spikes_maps_matrix, smooth)
            occupancy_weighted_tuning_maps_matrix = PfND_TimeDependent.compute_occupancy_weighted_tuning_map(seconds_occupancy, smoothed_spikes_maps_matrix)
        else:
            smoothed_spikes_maps_matrix = None
            occupancy_weighted_tuning_maps_matrix = PfND_TimeDependent.compute_occupancy_weighted_tuning_map(seconds_occupancy, spikes_maps_matrix)

        # Smooth the final tuning map if needed and valid smooth parameter. Default True.            
        if (cls.should_smooth_final_tuning_map and (smooth is not None) and smooth_criteria_fn(smooth)): 
            occupancy_weighted_tuning_maps_matrix = spikes_maps_smooth_gaussian_filter_fn(occupancy_weighted_tuning_maps_matrix, smooth)
        
        return PlacefieldSnapshot(num_position_samples_occupancy=num_position_samples_occupancy, num_position_samples_smoothed_occupancy=None, seconds_occupancy=seconds_occupancy, normalized_occupancy=normalized_occupancy,
            spikes_maps_matrix=spikes_maps_matrix, smoothed_spikes_maps_matrix=smoothed_spikes_maps_matrix, 
            occupancy_weighted_tuning_maps_matrix=occupancy_weighted_tuning_maps_matrix)
        

    @classmethod
    def validate_pf_dt_equivalent(cls, active_pf_nD, active_pf_nD_dt):
        """ Asserts to make sure that the fully-updated dt is equal to the normal 

        Usage:
            from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent

            computation_result = curr_active_pipeline.computation_results[global_epoch_name]
            active_session, pf_computation_config = computation_result.sess, computation_result.computation_config.pf_params
            active_session_spikes_df, active_pos, computation_config, active_epoch_placefields1D, active_epoch_placefields2D, included_epochs, should_force_recompute_placefields = active_session.spikes_df, active_session.position, pf_computation_config, None, None, pf_computation_config.computation_epochs, True
            active_pf_1D_dt = PfND_TimeDependent(deepcopy(active_session_spikes_df), deepcopy(active_pos.linear_pos_obj), epochs=included_epochs,
                                                speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
                                                grid_bin=computation_config.grid_bin, grid_bin_bounds=computation_config.grid_bin_bounds, smooth=computation_config.smooth)


            ## Update the time-dependent pf:
            active_pf_1D_dt.reset()

            ## Find the neuron_IDs that are included in the active_pf_1D for filtering the active_pf_1D_dt's results:
            is_pf_1D_included_neuron = np.isin(active_pf_1D_dt.included_neuron_IDs, active_pf_1D.included_neuron_IDs)
            pf_1D_included_neuron_indx = active_pf_1D_dt._included_thresh_neurons_indx[is_pf_1D_included_neuron]
            active_pf_1D_dt._included_thresh_neurons_indx = pf_1D_included_neuron_indx
            active_pf_1D_dt._peak_frate_filter_function = lambda list_: [list_[_] for _ in active_pf_1D_dt._included_thresh_neurons_indx]

            earliest_pos_t = active_pf_1D_dt.all_time_filtered_pos_df['t'].values[0]
            print(f'earliest_pos_t: {earliest_pos_t}')
            # active_pf_1D_dt.update(t=earliest_pos_t)
            # active_pf_1D_dt.step(num_seconds_to_advance=6000.0)

            last_pos_t = active_pf_1D_dt.all_time_filtered_pos_df['t'].values[-1]
            print(f'last_pos_t: {last_pos_t}')
            # active_pf_1D_dt.update(t=last_pos_t)

            # active_pf_1D_dt.update(t=earliest_pos_t+(60.0 * 5.0))
            active_pf_1D_dt.update(t=3000000.0)
            print(f'post-update time: {active_pf_1D_dt.last_t}')

            # earliest_pos_t: 22.26785500004189
            # last_pos_t: 1739.1316560000414
            # post-update time: 322.2678550000419

            validate_pf_dt_equivalent(active_pf_1D, active_pf_1D_dt)

        """
        assert active_pf_nD_dt.all_time_filtered_pos_df.shape == active_pf_nD_dt.filtered_pos_df.shape, f"active_pf_nD_dt.all_time_filtered_pos_df.shape: {active_pf_nD_dt.all_time_filtered_pos_df.shape}\nactive_pf_nD_dt.filtered_pos_df.shape: {active_pf_nD_dt.filtered_pos_df.shape} "
        assert active_pf_nD_dt.all_time_filtered_spikes_df.shape == active_pf_nD_dt.filtered_spikes_df.shape, f"active_pf_nD_dt.all_time_filtered_spikes_df.shape: {active_pf_nD_dt.all_time_filtered_spikes_df.shape}\nactive_pf_nD_dt.filtered_spikes_df.shape: {active_pf_nD_dt.filtered_spikes_df.shape} "
        # Occupancies are equal:

        assert np.isclose(active_pf_nD_dt.ratemap.occupancy, active_pf_nD.ratemap.occupancy).all(), f"active_pf_nD_dt.ratemap.occupancy: {active_pf_nD_dt.ratemap.occupancy}\nactive_pf_nD.ratemap.occupancy: {active_pf_nD.ratemap.occupancy}"
        # assert (active_pf_nD_dt.ratemap.occupancy == active_pf_nD.ratemap.occupancy).all(), f"active_pf_nD_dt.ratemap.occupancy: {active_pf_nD_dt.ratemap.occupancy}\nactive_pf_nD.ratemap.occupancy: {active_pf_nD.ratemap.occupancy}"
        
        # assert (active_pf_nD_dt.ratemap.spikes_maps == active_pf_nD.ratemap.spikes_maps).all(), f"active_pf_nD_dt.ratemap.spikes_maps: {active_pf_nD_dt.ratemap.spikes_maps}\nactive_pf_nD.ratemap.spikes_maps: {active_pf_nD.ratemap.spikes_maps}"
        assert np.isclose(active_pf_nD_dt.ratemap.spikes_maps, active_pf_nD.ratemap.spikes_maps).all(), f"active_pf_nD_dt.ratemap.spikes_maps: {active_pf_nD_dt.ratemap.spikes_maps}\nactive_pf_nD.ratemap.spikes_maps: {active_pf_nD.ratemap.spikes_maps}"
        


    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path
        Usage:
            hdf5_output_path: Path = curr_active_pipeline.get_output_path().joinpath('test_data.h5')
            _pfnd_obj: PfND = long_one_step_decoder_1D.pf
            _pfnd_obj.to_hdf(hdf5_output_path, key='test_pfnd')
            
            included_key_names = ['spikes_df', 'position', 'epochs', 'config', 'position_srate', 'setup_on_init', 'compute_on_init',
          '_save_intermediate_spikes_maps', '_filtered_pos_df', '_filtered_spikes_df',
           'ndim', 'xbin', 'ybin', 'bin_info', 'last_t', 'historical_snapshots', 'fragile_linear_neuron_IDXs', 'n_fragile_linear_neuron_IDXs']

        # print(f'to_dict(...): {list(self.__dict__.keys())}')
        curr_snapshot_time, curr_snapshot_data = self.snapshot() # take a snapshot of the current state
        # self._setup_time_varying() # reset completely before saving. Throw out everything
        # Excluded from serialization: ['_included_thresh_neurons_indx', '_peak_frate_filter_function']
        # filter_fn = filters.exclude(fields(PfND)._included_thresh_neurons_indx, int)
        # [an_attr.name for an_attr in self.__attrs_attrs__]
        # print(f'{[an_attr.name for an_attr in self.__attrs_attrs__]}')

        # filter_fn = lambda attr, value: attr.name not in ["_included_thresh_neurons_indx", "_peak_frate_filter_function", "_ratemap", "_ratemap_spiketrains", "_ratemap_spiketrains_pos"]
        # if not hasattr(self, '_included_thresh_neurons_indx'):
        #     self.reset()
        # filter_fn = lambda attr, value: attr.name in included_key_names
        # _out = asdict(self, recurse=False, filter=filter_fn) # serialize using attrs.asdict but exclude the listed properties
        _out = {k:v for k,v in self.__dict__.items() if k in included_key_names}
        
        
        """
    
        self.position.to_hdf(file_path=file_path, key=f'{key}/pos')
        if self.epochs is not None:
            self.epochs.to_hdf(file_path=file_path, key=f'{key}/epochs') #TODO 2023-07-30 11:13: - [ ] What if self.epochs is None?
        else:
            # if self.epochs is None
            pass
        self.spikes_df.spikes.to_hdf(file_path, key=f'{key}/spikes')
        self.ratemap.to_hdf(file_path, key=f'{key}/ratemap')


        # Do the dt specific ones:
        #'_filtered_pos_df', '_filtered_spikes_df', 'xbin', 'ybin', 'bin_info', 'last_t', 'historical_snapshots', 'fragile_linear_neuron_IDXs', 'n_fragile_linear_neuron_IDXs'
        snapshots_xarray: xr.DataArray = self.prepare_snapshots_for_export_as_xarray()
        snapshots_np_array = snapshots_xarray.to_numpy() # .shape # (5104, 49, 112)
        # Write the xr.DataArray to .h5 file:
        # snapshots_xarray.to_netcdf(file_path, kwargs.get('mode', 'a'), group=f'{key}/snapshots', ) # "output/test_xr_DataArray.h5"
        # snapshots_np_array

        # Open the file with h5py to add attributes to the group. The pandas.HDFStore object doesn't provide a direct way to manipulate groups as objects, as it is primarily intended to work with datasets (i.e., pandas DataFrames)
        with h5py.File(file_path, 'r+') as f:
            ## Unfortunately, you cannot directly assign a dictionary to the attrs attribute of an h5py group or dataset. The attrs attribute is an instance of a special class that behaves like a dictionary in some ways but not in others. You must assign attributes individually
            group = f[key]

            ## Assign by numpy array first:
            group.create_dataset('snapshots_array', data=snapshots_np_array)
            

            group.attrs['position_srate'] = self.position_srate
            group.attrs['ndim'] = self.ndim

            # can't just set the dict directly
            # group.attrs['config'] = str(self.config.to_dict())  # Store as string if it's a complex object
            # Manually set the config attributes
            config_dict = self.config.to_dict()
            group.attrs['config/speed_thresh'] = config_dict['speed_thresh']
            group.attrs['config/grid_bin'] = config_dict['grid_bin']
            group.attrs['config/grid_bin_bounds'] = config_dict['grid_bin_bounds']
            group.attrs['config/smooth'] = config_dict['smooth']
            group.attrs['config/frate_thresh'] = config_dict['frate_thresh']
            
            # Do the dt-specific ones:
            group.attrs['last_t'] = self.last_t
            group.attrs['n_fragile_linear_neuron_IDXs'] = self.n_fragile_linear_neuron_IDXs





    @classmethod
    def read_hdf(cls, file_path, key: str, **kwargs) -> "PfND":
        """ Reads the data from the key in the hdf5 file at file_path
        Usage:
            _reread_pfnd_obj = PfND.read_hdf(hdf5_output_path, key='test_pfnd')
            _reread_pfnd_obj
        """
        # Read DataFrames using pandas
        position = Position.read_hdf(file_path, key=f'{key}/pos')
        try:
            epochs = Epoch.read_hdf(file_path, key=f'{key}/epochs')
        except KeyError as e:
            # epochs can be None, in which case the serialized object will not contain the f'{key}/epochs' key.  'No object named test_pfnd/epochs in the file'
            epochs = None
        except Exception as e:
            # epochs can be None, in which case the serialized object will not contain the f'{key}/epochs' key
            print(f'Unhandled exception {e}')
            raise e
        
        spikes_df = SpikesAccessor.read_hdf(file_path, key=f'{key}/spikes')

        # Open the file with h5py to read attributes
        with h5py.File(file_path, 'r') as f:
            group = f[key]
            position_srate = group.attrs['position_srate']
            ndim = group.attrs['ndim'] # Assuming you'll use it somewhere else if needed

            # Read the config attributes
            config_dict = {
                'speed_thresh': group.attrs['config/speed_thresh'],
                'grid_bin': tuple(group.attrs['config/grid_bin']),
                'grid_bin_bounds': tuple(group.attrs['config/grid_bin_bounds']),
                'smooth': tuple(group.attrs['config/smooth']),
                'frate_thresh': group.attrs['config/frate_thresh']
            }

        # Create a PlacefieldComputationParameters object from the config_dict
        config = PlacefieldComputationParameters(**config_dict)

        #TODO 2023-09-26 08:47: - [ ] Needs UPDATE for dt-version. This was copied literally from non-time dependent version


        # Reconstruct the object using the from_config_values class method
        return cls(spikes_df=spikes_df, position=position, epochs=epochs, config=config, position_srate=position_srate)
    
    
    def prepare_snapshots_for_export_as_xarray(self) -> xr.DataArray:
        """ exports all snapshots as an xarray """
        return prepare_snapshots_for_export_as_xarray(historical_snapshots=self.historical_snapshots, ndim=self.ndim)



def perform_compute_time_dependent_placefields(active_session_spikes_df, active_pos, computation_config: PlacefieldComputationParameters, active_epoch_placefields1D=None, active_epoch_placefields2D=None, included_epochs=None, should_force_recompute_placefields=True):
    """ Most general computation function. Computes both 1D and 2D time-dependent placefields.
    active_epoch_session_Neurons: 
    active_epoch_pos: a Position object
    included_epochs: a Epoch object to filter with, only included epochs are included in the PF calculations
    active_epoch_placefields1D (Pf1D, optional) & active_epoch_placefields2D (Pf2D, optional): allow you to pass already computed Pf1D and Pf2D objects from previous runs and it won't recompute them so long as should_force_recompute_placefields=False, which is useful in interactive Notebooks/scripts
    Usage:
        active_epoch_placefields1D, active_epoch_placefields2D = perform_compute_time_dependent_placefields(active_epoch_session_Neurons, active_epoch_pos, active_epoch_placefields1D, active_epoch_placefields2D, active_config.computation_config, should_force_recompute_placefields=True)
    """
    ## Linearized (1D) Position Placefields:
    if ((active_epoch_placefields1D is None) or should_force_recompute_placefields):
        print('Recomputing active_epoch_time_dependent_placefields...', end=' ')
        spikes_df = deepcopy(active_session_spikes_df).spikes.sliced_by_neuron_type('PYRAMIDAL') # Only use PYRAMIDAL neurons
        active_epoch_placefields1D = PfND_TimeDependent.from_config_values(spikes_df, deepcopy(active_pos.linear_pos_obj), epochs=included_epochs,
                                        speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
                                        grid_bin=computation_config.grid_bin, grid_bin_bounds=computation_config.grid_bin_bounds, smooth=computation_config.smooth)

        print('\t done.')
    else:
        print('active_epoch_placefields1D already exists, reusing it.')

    ## 2D Position Placemaps:
    if ((active_epoch_placefields2D is None) or should_force_recompute_placefields):
        print('Recomputing active_epoch_time_dependent_placefields2D...', end=' ')
        spikes_df = deepcopy(active_session_spikes_df).spikes.sliced_by_neuron_type('PYRAMIDAL') # Only use PYRAMIDAL neurons
        active_epoch_placefields2D = PfND_TimeDependent.from_config_values(spikes_df, deepcopy(active_pos), epochs=included_epochs,
                                        speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
                                        grid_bin=computation_config.grid_bin, grid_bin_bounds=computation_config.grid_bin_bounds, smooth=computation_config.smooth)

        print('\t done.')
    else:
        print('active_epoch_placefields2D already exists, reusing it.')
    
    return active_epoch_placefields1D, active_epoch_placefields2D
