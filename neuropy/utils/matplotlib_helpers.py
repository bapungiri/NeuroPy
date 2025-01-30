from __future__ import annotations # otherwise have to do type like 'Ratemap'

from enum import Enum, IntEnum, auto, unique
from collections import namedtuple
import numpy as np
import contextlib
from copy import deepcopy
from attrs import define, field, Factory

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
from matplotlib.collections import BrokenBarHCollection # for draw_epoch_regions
from matplotlib.widgets import RectangleSelector # required for `add_rectangular_selector`
from matplotlib.widgets import SpanSelector

from neuropy.utils.misc import AutoNameEnum, compute_paginated_grid_config, RowColTuple

from typing import TYPE_CHECKING, Dict, List, Tuple, Optional, Callable, Union, Any
from nptyping import NDArray

if TYPE_CHECKING:
    from neuropy.core.neuron_identities import PlotStringBrevityModeEnum # needed for _build_neuron_identity_label
    from neuropy.core.neuron_identities import NeuronExtendedIdentity # needed for _build_neuron_identity_label
    


""" Note that currently the only Matplotlib-specific functions here are add_inner_title(...) and draw_sizebar(...). The rest have general uses! """

# refactored to pyPhoCoreHelpers.geometery_helpers but had to be bring back in explicitly
Width_Height_Tuple = namedtuple('Width_Height_Tuple', 'width height')
def compute_data_extent(xpoints, *other_1d_series):
    """Computes the outer bounds, or "extent" of one or more 1D data series.

    Args:
        xpoints ([type]): [description]
        other_1d_series: any number of other 1d data series

    Returns:
        xmin, xmax, ymin, ymax, imin, imax, ...: a flat list of paired min, max values for each data series provided.
        
    Usage:
        # arbitrary number of data sequences:        
        xmin, xmax, ymin, ymax, x_center_min, x_center_max, y_center_min, y_center_max = compute_data_extent(active_epoch_placefields2D.ratemap.xbin, active_epoch_placefields2D.ratemap.ybin, active_epoch_placefields2D.ratemap.xbin_centers, active_epoch_placefields2D.ratemap.ybin_centers)
        print(xmin, xmax, ymin, ymax, x_center_min, x_center_max, y_center_min, y_center_max)

        # simple 2D extent:
        extent = compute_data_extent(active_epoch_placefields2D.ratemap.xbin, active_epoch_placefields2D.ratemap.ybin)
        print(extent)
    """
    num_total_series = len(other_1d_series) + 1 # + 1 for the x-series
    # pre-allocate output:     
    extent = np.empty(((2 * num_total_series),))
    # Do first-required series:
    xmin, xmax = min_max_1d(xpoints)
    extent[0], extent[1] = [xmin, xmax]
    # finish remaining series passed as inputs.
    for (i, a_series) in enumerate(other_1d_series):
        curr_min, curr_xmax = min_max_1d(a_series)
        curr_start_idx = 2 * (i + 1)
        extent[curr_start_idx] = curr_min
        extent[curr_start_idx+1] = curr_xmax
    return extent

def compute_data_aspect_ratio(xbin, ybin, sorted_inputs=True):
    """Computes the aspect ratio of the provided data

    Args:
        xbin ([type]): [description]
        ybin ([type]): [description]
        sorted_inputs (bool, optional): whether the input arrays are pre-sorted in ascending order or not. Defaults to True.

    Returns:
        float: The aspect ratio of the data such that multiplying any height by the returned float would result in a width in the same aspect ratio as the data.
    """
    if sorted_inputs:
        xmin, xmax, ymin, ymax = (xbin[0], xbin[-1], ybin[0], ybin[-1]) # assumes-pre-sourced events, which is valid for bins but not general
    else:
        xmin, xmax, ymin, ymax = compute_data_extent(xbin, ybin) # more general form.

    # The extent keyword arguments controls the bounding box in data coordinates that the image will fill specified as (left, right, bottom, top) in data coordinates, the origin keyword argument controls how the image fills that bounding box, and the orientation in the final rendered image is also affected by the axes limits.
    # extent = (xmin, xmax, ymin, ymax)
    
    width = xmax - xmin
    height = ymax - ymin
    
    aspect_ratio = width / height
    return aspect_ratio, Width_Height_Tuple(width, height)

@unique
class enumTuningMap2DPlotMode(AutoNameEnum):
    PCOLORFAST = auto() # DEFAULT prior to 2021-12-24
    PCOLORMESH = auto() # UNTESTED
    PCOLOR = auto() # UNTESTED
    IMSHOW = auto() # New Default as of 2021-12-24

@unique
class enumTuningMap2DPlotVariables(AutoNameEnum):
    TUNING_MAPS = auto() # DEFAULT
    SPIKES_MAPS = auto()
    OCCUPANCY = auto()
    

    
def _build_neuron_identity_label(neuron_extended_id: NeuronExtendedIdentity=None, brev_mode: PlotStringBrevityModeEnum=None, formatted_max_value_string=None, use_special_overlayed_title=True):
    """ builds the subplot title for 2D PFs that displays the neuron identity and other important info. """    
    from neuropy.core.neuron_identities import PlotStringBrevityModeEnum # needed for _build_neuron_identity_label
    from neuropy.core.neuron_identities import NeuronExtendedIdentity # needed for _build_neuron_identity_label

    if brev_mode is None:
        brev_mode = PlotStringBrevityModeEnum.CONCISE

    if neuron_extended_id is not None:
        if isinstance(neuron_extended_id, (tuple, )):
            ## convert to modern
            neuron_extended_id = NeuronExtendedIdentity.init_from_NeuronExtendedIdentityTuple(neuron_extended_id)
        
        full_extended_id_string = brev_mode.extended_identity_formatting_string(neuron_extended_id)
    else:
        full_extended_id_string = ''
    
    final_string_components = [full_extended_id_string]
    
    if formatted_max_value_string is not None:
        final_string_components.append(formatted_max_value_string)
    
    if use_special_overlayed_title:
        final_title = ' - '.join(final_string_components)
    else:
        # conventional way:
        final_title = '\n'.join(final_string_components) # f"Cell {ratemap.neuron_ids[cell]} - {ratemap.get_extended_neuron_id_string(neuron_i=cell)} \n{round(np.nanmax(pfmap),2)} Hz"
    return final_title
    
def _build_variable_max_value_label(plot_variable: enumTuningMap2DPlotVariables):
    """  Builds a label that displays the max value with the appropriate unit suffix for the title
    if brev_mode.should_show_firing_rate_label:
        pf_firing_rate_string = f'{round(np.nanmax(pfmap),2)} Hz'
        final_string_components.append(pf_firing_rate_string)
    """
    if plot_variable.name is enumTuningMap2DPlotVariables.TUNING_MAPS.name:
        return lambda value: f'{round(value,2)} Hz'
    elif plot_variable.name == enumTuningMap2DPlotVariables.SPIKES_MAPS.name:
        return lambda value: f'{round(value,2)} Spikes'
    else:
        raise NotImplementedError

def _determine_best_placefield_2D_layout(xbin, ybin, included_unit_indicies, subplots:RowColTuple=(40, 3), fig_column_width:float=8.0, fig_row_height:float=1.0, resolution_multiplier:float=1.0, max_screen_figure_size=(None, None), last_figure_subplots_same_layout=True, debug_print:bool=False):
    """ Computes the optimal sizes, number of rows and columns, and layout of the individual 2D placefield subplots in terms of the overarching pf_2D figure
    
    Interally Calls:
        neuropy.utils.misc.compute_paginated_grid_config(...)


    Known Uses:
        display_all_pf_2D_pyqtgraph_binned_image_rendering
        plot_advanced_2D
    
    Major outputs:
    
    
    (curr_fig_page_grid_size.num_rows, curr_fig_page_grid_size.num_columns)
    
    
    Usage Example:
        nfigures, num_pages, included_combined_indicies_pages, page_grid_sizes, data_aspect_ratio, page_figure_sizes = _final_wrapped_determine_placefield_2D_layout(xbin=active_pf_2D.xbin, ybin=active_pf_2D.ybin, included_unit_indicies=np.arange(active_pf_2D.ratemap.n_neurons), subplots=(40, 3), fig_column_width=8.0, fig_row_height=1.0, resolution_multiplier=1.0, max_screen_figure_size=(None, None), last_figure_subplots_same_layout=True, debug_print=True)
        
        print(f'nfigures: {nfigures}\ndata_aspect_ratio: {data_aspect_ratio}')
        # Loop through each page/figure that's required:
        for page_fig_ind, page_fig_size, page_grid_size in zip(np.arange(nfigures), page_figure_sizes, page_grid_sizes):
            print(f'\tpage_fig_ind: {page_fig_ind}, page_fig_size: {page_fig_size}, page_grid_size: {page_grid_size}')
               
        
    """
    from neuropy.plotting.figure import compute_figure_size_pixels, compute_figure_size_inches # needed for _determine_best_placefield_2D_layout(...)'s internal _perform_compute_required_figure_sizes(...) function

    def _perform_compute_optimal_paginated_grid_layout(xbin, ybin, included_unit_indicies, subplots:RowColTuple=(40, 3), last_figure_subplots_same_layout=True, debug_print:bool=False):
        if not isinstance(subplots, RowColTuple):
            subplots = RowColTuple(subplots[0], subplots[1])
        
        nMapsToShow = len(included_unit_indicies)
        data_aspect_ratio = compute_data_aspect_ratio(xbin, ybin)
        if debug_print:
            print(f'data_aspect_ratio: {data_aspect_ratio}')
        
        if (subplots.num_columns is None) or (subplots.num_rows is None):
            # This will disable pagination by setting an arbitrarily high value
            max_subplots_per_page = nMapsToShow
            if debug_print:
                print('Pagination is disabled because one of the subplots values is None. Output will be in a single figure/page.')
        else:
            # valid specified maximum subplots per page
            max_subplots_per_page = int(subplots.num_columns * subplots.num_rows)
        
        if debug_print:
            print(f'nMapsToShow: {nMapsToShow}, subplots: {subplots}, max_subplots_per_page: {max_subplots_per_page}')
            
        # Paging Management: Constrain the subplots values to just those that you need
        subplot_no_pagination_configuration, included_combined_indicies_pages, page_grid_sizes = compute_paginated_grid_config(nMapsToShow, max_num_columns=subplots.num_columns, max_subplots_per_page=max_subplots_per_page, data_indicies=included_unit_indicies, last_figure_subplots_same_layout=last_figure_subplots_same_layout)
        num_pages = len(included_combined_indicies_pages)
        nfigures = num_pages
        # nfigures = nMapsToShow // np.prod(subplots) + 1 # "//" is floor division (rounding result down to nearest whole number)
        return nfigures, num_pages, included_combined_indicies_pages, page_grid_sizes, data_aspect_ratio
    
    def _perform_compute_required_figure_sizes(curr_fig_page_grid_size, data_aspect_ratio, fig_column_width:float=None, fig_row_height:float=None, resolution_multiplier:float=1.0, max_screen_figure_size=(None, None), debug_print:bool=False):
        if resolution_multiplier is None:
            resolution_multiplier = 1.0
        if (fig_column_width is not None) and (fig_row_height is not None):
            desired_single_map_width = fig_column_width * resolution_multiplier
            desired_single_map_height = fig_row_height * resolution_multiplier
        else:
            ## TODO: I think this hardcoded 4.0 should be set to data_aspect_ratio: (1.0167365776358197 for square maps)
            desired_single_map_width = data_aspect_ratio[0] * resolution_multiplier
            desired_single_map_height = 1.0 * resolution_multiplier
            
        # Computes desired_single_map_width and desired_signle_map_height
            
        ## Figure size should be (Width, height)
        required_figure_size = ((float(curr_fig_page_grid_size.num_columns) * float(desired_single_map_width)), (float(curr_fig_page_grid_size.num_rows) * float(desired_single_map_height))) # (width, height)
        required_figure_size_px = compute_figure_size_pixels(required_figure_size)
        if debug_print:
            print(f'resolution_multiplier: {resolution_multiplier}, required_figure_size: {required_figure_size}, required_figure_size_px: {required_figure_size_px}') # this is figure size in inches

        active_figure_size = required_figure_size
        
        # If max_screen_figure_size is not None (it should be a two element tuple, specifying the max width and height in pixels for the figure:
        if max_screen_figure_size is not None:
            required_figure_size_px = list(required_figure_size_px) # convert to a list instead of a tuple to make it mutable
            if max_screen_figure_size[0] is not None:
                required_figure_size_px[0] = min(required_figure_size_px[0], max_screen_figure_size[0])
            if max_screen_figure_size[1] is not None:
                required_figure_size_px[1] = min(required_figure_size_px[1], max_screen_figure_size[1])

        required_figure_size_px = tuple(required_figure_size_px)
        # convert back to inches from pixels to constrain the figure size:
        required_figure_size = compute_figure_size_inches(required_figure_size_px) # Convert back from pixels to inches when done
        # Update active_figure_size again:
        active_figure_size = required_figure_size
        
        # active_figure_size=figsize
        # active_figure_size=required_figure_size
        if debug_print:
            print(f'final active_figure_size: {active_figure_size}, required_figure_size_px: {required_figure_size_px} (after constraining by max_screen_figure_size, etc)')

        return active_figure_size

    # BEGIN MAIN FUNCTION BODY ___________________________________________________________________________________________ #
    nfigures, num_pages, included_combined_indicies_pages, page_grid_sizes, data_aspect_ratio = _perform_compute_optimal_paginated_grid_layout(xbin=xbin, ybin=ybin, included_unit_indicies=included_unit_indicies, subplots=subplots, last_figure_subplots_same_layout=last_figure_subplots_same_layout, debug_print=debug_print)
    if resolution_multiplier is None:
        resolution_multiplier = 1.0

    page_figure_sizes = []
    for fig_ind in range(nfigures):
        # Dynamic Figure Sizing: 
        curr_fig_page_grid_size = page_grid_sizes[fig_ind]
        ## active_figure_size is the primary output
        active_figure_size = _perform_compute_required_figure_sizes(curr_fig_page_grid_size, data_aspect_ratio=data_aspect_ratio, fig_column_width=fig_column_width, fig_row_height=fig_row_height, resolution_multiplier=resolution_multiplier, max_screen_figure_size=max_screen_figure_size, debug_print=debug_print)
        page_figure_sizes.append(active_figure_size)
        
    return nfigures, num_pages, included_combined_indicies_pages, page_grid_sizes, data_aspect_ratio, page_figure_sizes

def _scale_current_placefield_to_acceptable_range(image, occupancy, drop_below_threshold: float=0.0000001):
    """ Universally used to prepare the pfmap to be displayed (across every plot time)
    
    Regardless of `occupancy` and `drop_below_threshold`, the image is rescaled by its maximum (meaning the output will be normalized between zero and one).
    `occupancy` is not used unless `drop_below_threshold` is non-None
    

    Input:
        drop_below_threshold: if None, no indicies are dropped. Otherwise, values of occupancy less than the threshold specified are used to build a mask, which is subtracted from the returned image (the image is NaN'ed out in these places).

    Known Uses:
            NeuroPy.neuropy.plotting.ratemaps.plot_single_tuning_map_2D(...)
            pyphoplacecellanalysis.Pho2D.PyQtPlots.plot_placefields.pyqtplot_plot_image_array(...)
            
     # image = np.squeeze(images[a_linear_index,:,:])
    """
    # Pre-filter the data:
    with np.errstate(divide='ignore', invalid='ignore'):
        image = np.array(image.copy()) / np.nanmax(image) # note scaling by maximum here!
        if (drop_below_threshold is not None) and (occupancy is not None):
            image[np.where(occupancy < drop_below_threshold)] = np.nan # null out the occupancy
        return image # return the modified and masked image

    
def _build_square_checkerboard_image(extent, num_checkerboard_squares_short_axis:int=10, debug_print=False):
    """ builds a background checkerboard image used to indicate opacity
    Usage:
    # Updating Existing:
    background_chessboard = _build_square_checkerboard_image(active_ax_main_image.get_extent(), num_checkerboard_squares_short_axis=8)
    active_ax_bg_image.set_data(background_chessboard) # updating mode
    
    # Creation:
    background_chessboard = _build_square_checkerboard_image(active_ax_main_image.get_extent(), num_checkerboard_squares_short_axis=8)
    bg_im = ax.imshow(background_chessboard, cmap=plt.cm.gray, interpolation='nearest', **imshow_shared_kwargs, label='background_image')
    
    """
    left, right, bottom, top = extent
    width = np.abs(left - right)
    height = np.abs(top - bottom) # width: 241.7178791533281, height: 30.256480996256016
    if debug_print:
        print(f'width: {width}, height: {height}')
    
    if width >= height:
        short_axis_length = float(height)
        long_axis_length = float(width)
    else:
        short_axis_length = float(width)
        long_axis_length = float(height)
    
    checkerboard_square_side_length = short_axis_length / float(num_checkerboard_squares_short_axis) # checkerboard_square_side_length is the same along all axes
    frac_num_checkerboard_squares_long_axis = long_axis_length / float(checkerboard_square_side_length)
    num_checkerboard_squares_long_axis = int(np.round(frac_num_checkerboard_squares_long_axis))
    if debug_print:
        print(f'checkerboard_square_side: {checkerboard_square_side_length}, num_checkerboard_squares_short_axis: {num_checkerboard_squares_short_axis}, num_checkerboard_squares_long_axis: {num_checkerboard_squares_long_axis}')
    # Grey checkerboard background:
    background_chessboard = np.add.outer(range(num_checkerboard_squares_short_axis), range(num_checkerboard_squares_long_axis)) % 2  # chessboard
    return background_chessboard





# ==================================================================================================================== #
# These are the only Matplotlib-specific functions here: add_inner_title(...) and draw_sizebar(...).                              #
# ==================================================================================================================== #

# @function_attributes(short_name=None, tags=['matplotlib', 'margin', 'layout', 'constrained_layout'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-29 08:16', related_items=[])
def set_margins(fig, left=0, right=0, top=0, bottom=0, is_in_inches: bool=False):
    """Set figure margins as [left, right, top, bottom] in inches
    from the edges of the figure.
    

    You can set the rectangle that the layout engine operates within. See the rect parameter for each engine at https://matplotlib.org/stable/api/layout_engine_api.html.
    It's unfortunately not a very friendly part of the API, especially because TightLayoutEngine and ConstrainedLayoutEngine have different semantics for rect: TightLayoutEngine uses rect = (left, bottom, right, top) and ConstrainedLayoutEngine uses rect = (left, bottom, width, height).

    Usage:
        
        from neuropy.utils.matplotlib_helpers import set_margins
        
        #your margins were [0.2, 0.8, 0.2, 0.8] in figure coordinates
        #which are 0.2*11 and 0.2*8.5 in inches from the edge
        set_margins(a_fig, top=0.2) # [0.2*11, 0.2*11, 0.2*8.5, 0.2*8.5]
            
    
    """
    if is_in_inches:
        width, height = fig.get_size_inches()
        #convert to figure coordinates:
        left, right = left/width, 1-right/width
        bottom, top = bottom/height, 1-top/height
    else:
        # already in figure coordinates
        pass
    
    #get the layout engine and convert to its desired format
    engine = fig.get_layout_engine()
    if isinstance(engine, matplotlib.layout_engine.TightLayoutEngine):
        rect = (left, bottom, right, top)
    elif isinstance(engine, matplotlib.layout_engine.ConstrainedLayoutEngine):
        rect = (left, bottom, right-left, top-bottom)
    else:
        raise RuntimeError('Cannot adjust margins of unsupported layout engine')
    #set and recompute the layout
    engine.set(rect=rect)
    engine.execute(fig)



def add_text_with_stroke(ax, text: str, x_pos: float, y_pos: float, strokewidth=3, stroke_foreground='w', stroke_alpha=0.9, text_foreground='k', font_size=None, text_alpha=1.0, **kwargs):
    """
    Add a new ax.text(...) object to the axes but with an outline.

    Args:
        ax (matplotlib.axes.Axes): The axes object where the title should be added.
        title (str): The title text.
        loc (str or int): The location code for the title placement.
        strokewidth (int, optional): The line width for the stroke around the text. Default is 3.
        stroke_foreground (str, optional): The color for the stroke around the text. Default is 'w' (white).
        stroke_alpha (float, optional): The alpha value for the stroke. Default is 0.9.
        text_foreground (str, optional): The color for the text. Default is 'k' (black).
        font_size (int, optional): The font size for the title text. If not provided, it will use the value from plt.rcParams['legend.title_fontsize'].
        text_alpha (float, optional): The alpha value for the text itself. Default is 1.0 (opaque).
        **kwargs: Additional keyword arguments to be passed to AnchoredText.

    Returns:
        matplotlib.offsetbox.AnchoredText: The AnchoredText object containing the title.
    """
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnchoredText
    # from matplotlib.patheffects import withStroke
    import matplotlib.patheffects as path_effects

    # List of keys to be removed from kwargs and added to text_prop_kwargs if present
    # text_pop_key_name_list = ('horizontalalignment', 'verticalalignment', 'multialignment', 'rotation', 'fontproperties')
    text_pop_key_name_list = ('horizontalalignment', 'verticalalignment', 'multialignment', 'ha', 'va', 'rotation', 'fontproperties')

    # Get any text property keyword arguments from kwargs, or use an empty dictionary if not present
    text_prop_kwargs = kwargs.pop('text_prop_kwargs', {})

    # Add the path effect for the stroke, font size, text color, and any relevant keyword arguments from text_pop_key_name_list
    text_prop_kwargs = text_prop_kwargs | dict(
        # path_effects=[withStroke(foreground=stroke_foreground, linewidth=strokewidth, alpha=stroke_alpha)],
        size=(font_size or plt.rcParams['legend.title_fontsize']),
        color=text_foreground,
        **{k: kwargs.pop(k) for k in text_pop_key_name_list if k in kwargs}
    )

    # Create the AnchoredText object with the specified properties
    # if use_AnchoredCustomText:
    #     at = AnchoredCustomText(text, loc=loc, prop=text_prop_kwargs, pad=0., borderpad=0.5, frameon=False, **kwargs)
    # else:
    #     ## cannot have custom_value_formatter
    #     custom_value_formatter = kwargs.pop('custom_value_formatter', None)
    #     assert custom_value_formatter is None, f"custom_value_formatter should be None for non-custom anchored text but custom_value_formatter: {custom_value_formatter}"
    #     at = AnchoredText(text, loc=loc, prop=text_prop_kwargs, pad=0., borderpad=0.5, frameon=False, **kwargs)

    # ax.add_artist(at)

    # default_text_kwargs = dict(ha='center', va='center', color='white')
    
    # Add text with stroke outline
    text = ax.text(x_pos, y_pos, text, **text_prop_kwargs)
        # fontsize=(font_size or plt.rcParams['legend.title_fontsize']), ha='center', va='center', color=subseq_idx_text_color)

    # Apply path effects for the outline
    text.set_path_effects([
        path_effects.Stroke(linewidth=strokewidth, foreground=stroke_foreground, alpha=stroke_alpha),  # Outline (black)
        path_effects.Normal()  # Fill (white)
    ])

    # # Set the alpha value for the text itself, if specified
    # if text_alpha < 1.0:
    #     if use_AnchoredCustomText:
    #         at.update_text_alpha(text_alpha)
    #     else:
    #         at.txt._text.set_alpha(text_alpha)

    return text



def add_inner_title(ax, title, loc, strokewidth=3, stroke_foreground='w', stroke_alpha=0.9, text_foreground='k', font_size=None, text_alpha=1.0, use_AnchoredCustomText: bool=False, **kwargs):
    """
    Add a figure title inside the border of the figure (instead of outside).

    Args:
        ax (matplotlib.axes.Axes): The axes object where the title should be added.
        title (str): The title text.
        loc (str or int): The location code for the title placement.
        strokewidth (int, optional): The line width for the stroke around the text. Default is 3.
        stroke_foreground (str, optional): The color for the stroke around the text. Default is 'w' (white).
        stroke_alpha (float, optional): The alpha value for the stroke. Default is 0.9.
        text_foreground (str, optional): The color for the text. Default is 'k' (black).
        font_size (int, optional): The font size for the title text. If not provided, it will use the value from plt.rcParams['legend.title_fontsize'].
        text_alpha (float, optional): The alpha value for the text itself. Default is 1.0 (opaque).
        **kwargs: Additional keyword arguments to be passed to AnchoredText.

    Returns:
        matplotlib.offsetbox.AnchoredText: The AnchoredText object containing the title.
    """
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke

    # List of keys to be removed from kwargs and added to text_prop_kwargs if present
    text_pop_key_name_list = ('horizontalalignment', 'verticalalignment', 'multialignment', 'rotation', 'fontproperties')

    # Get any text property keyword arguments from kwargs, or use an empty dictionary if not present
    text_prop_kwargs = kwargs.pop('text_prop_kwargs', {})

    # Add the path effect for the stroke, font size, text color, and any relevant keyword arguments from text_pop_key_name_list
    text_prop_kwargs = text_prop_kwargs | dict(
        path_effects=[withStroke(foreground=stroke_foreground, linewidth=strokewidth, alpha=stroke_alpha)],
        size=(font_size or plt.rcParams['legend.title_fontsize']),
        color=text_foreground,
        **{k: kwargs.pop(k) for k in text_pop_key_name_list if k in kwargs}
    )

    # Create the AnchoredText object with the specified properties
    if use_AnchoredCustomText:
        at = AnchoredCustomText(title, loc=loc, prop=text_prop_kwargs, pad=0., borderpad=0.5, frameon=False, **kwargs)
    else:
        ## cannot have custom_value_formatter
        custom_value_formatter = kwargs.pop('custom_value_formatter', None)
        assert custom_value_formatter is None, f"custom_value_formatter should be None for non-custom anchored text but custom_value_formatter: {custom_value_formatter}"
        at = AnchoredText(title, loc=loc, prop=text_prop_kwargs, pad=0., borderpad=0.5, frameon=False, **kwargs)

    ax.add_artist(at)

    # Set the alpha value for the text itself, if specified
    if text_alpha < 1.0:
        if use_AnchoredCustomText:
            at.update_text_alpha(text_alpha)
        else:
            at.txt._text.set_alpha(text_alpha)

    return at
        


    
## TODO: Not currently used, but looks like it can add anchored scale/size bars to matplotlib figures pretty easily:
def draw_sizebar(ax):
    """
    Draw a horizontal bar with length of 0.1 in data coordinates,
    with a fixed label underneath.
    """
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    asb = AnchoredSizeBar(ax.transData,
                          0.1,
                          r"1$^{\prime}$",
                          loc='lower center',
                          pad=0.1, borderpad=0.5, sep=5,
                          frameon=False)
    ax.add_artist(asb)
    

def set_ax_emphasis_color(ax, emphasis_color = 'green', defer_draw:bool=False):
    """ for the provided axis: changes the spine color, the x/y tick/labels color to the emphasis color. 
    """
    # Change the spine color for all spines
    for spine in ax.spines.values():
        spine.set_color(emphasis_color)  # Change color to red

    # Set the color of the tick labels
    for label in ax.get_xticklabels():
        label.set_color(emphasis_color)
    for label in ax.get_yticklabels():
        label.set_color(emphasis_color)

    # Set the color of the axis labels
    ax.set_xlabel(ax.get_xlabel(), color=emphasis_color)
    ax.set_ylabel(ax.get_ylabel(), color=emphasis_color)

    ## This works to actually redraw:
    if not defer_draw:
        a_fig = ax.get_figure()
        a_fig.canvas.draw()

    
# Add check buttons to toggle axis properties
# def add_check_buttons(fig, button_positions, labels):
#     from matplotlib.widgets import CheckButtons

#     # Create check buttons
#     # Create an axes for the check button
#     col = (0.95, 0.95, 0.95, 0.6)
#     # col = 'none'
#     check_ax = fig.add_axes(button_positions, facecolor=col)  # This should be a list of positions if multiple buttons are added. E.g. [0.7, 0.05, 0.1, 0.1]

#     # check_ax = ax.inset_axes(button_positions, facecolor='none')

#     check = CheckButtons(check_ax, labels, [False]*len(labels)) # , frame_props={'linecolor': None}
#     # for r in check.rectangles:
#     #     r.set_facecolor("red") 
#     #     r.set_edgecolor("w")
#     #     r.set_alpha(0.3)

#     # [ll.set_color("white") for l in check.lines for ll in l]
#     # [ll.set_linewidth(3) for l in check.lines for ll in l]

#     # [lbl.set_alpha(0.9) for lbl in check.labels]

#     # for i, c in enumerate(["r", "b", "g"]):
#     #     check.labels[i].set_color(c)
#     #     check.labels[i].set_alpha(0.7)
                
#     # axis_check_buttons = []

#     # # Create CheckButtons for each label, aligning their left edge with x1
#     # for i, label in enumerate(labels):
#     #     # Create an axes for the check button
#     #     check_ax = a_fig.add_axes([ax_pos.x1, button_y_start, button_width, button_height], facecolor='none')
        
#     #     # Create the check button with a transparent background
#     #     check = CheckButtons(check_ax, [label], [False])
    
#     #     # Remove the background of the CheckButtons axes
#     #     check_ax.patch.set_alpha(0.0) # Make CheckButtons background fully transparent
        
#     #     # Optionally, adjust the color of the lines for the checkboxes and the text
#     #     for line in check.lines:  # These are the lines that make up the checkmarks
#     #         line.set_color('black')
#     #     for chk_text in check.labels:
#     #         chk_text.set_color('black')  # Set the color of the text labels to black
        
#     #     check.label = label  # Custom attribute (if you need it)
#     #     axis_check_buttons.append(check)

        
#     # Remove the background of the CheckButtons axes
#     # check_ax.patch.set_alpha(0.0) # Make CheckButtons background fully transparent
    
#     # # Optionally, adjust the color of the lines for the checkboxes and the text
#     # for line in check.lines:  # These are the lines that make up the checkmarks
#     #     line.set_color('black')
#     # for chk_text in check.labels:
#     #     chk_text.set_color('black')  # Set the color of the text labels to black
    
#     # Define an event for toggling the properties with check buttons
#     def toggle_label(label):
#         # Here you would toggle the associated property
#         print(f"Property '{label}' toggled")

#     # Connect the event handler
#     check.on_clicked(toggle_label)

#     return check
        


def add_selection_patch(ax, selection_color = 'green', alpha=0.6, zorder=-1, action_button_configs=None, debug_print=False, defer_draw:bool=False):
    """ adds a rectangle behind the ax, sticking out to the right side by default.
    
    Can be toggled on/off via 
    `rectangle.set_visible(not rectangle.get_visible())`

    """
    from matplotlib.patches import Rectangle

    rect_kwargs = dict(color=selection_color, alpha=alpha, zorder=zorder)

    # Get the position of the ax in figure coordinates
    ax_pos = ax.get_position()
    if debug_print:
        print("Bottom-left corner (x0, y0):", ax_pos.x0, ax_pos.y0)
        print("Width and Height (width, height):", ax_pos.width, ax_pos.height)

    ## Get the figure from the axes:
    a_fig = ax.get_figure()
    ## Fill up to the right edge of the figure:
    right_margin = (1.0 - (ax_pos.x1))
    selection_rect_width = ax_pos.width + (right_margin * 0.75) # fill 75% of the remaining right margin with the box
    rectangle = Rectangle((ax_pos.x0, ax_pos.y0), selection_rect_width, ax_pos.height, transform=a_fig.transFigure, **rect_kwargs)

    ax_xmid = (ax_pos.x0 + (ax_pos.width/2.0))

    # Add the rectangle directly to the figure, not to the ax
    a_fig.add_artist(rectangle)

    enable_action_buttons = False
    if action_button_configs is not None:
        num_buttons = len(action_button_configs)
        enable_action_buttons = num_buttons > 0
    
      # Number of buttons
    if enable_action_buttons:
        from matplotlib.widgets import CheckButtons
        # Coordinates for the buttons
        # button_width = selection_rect_width * (1.0 - ax_pos.x1)  # Use a portion of the rect for the buttons
        # button_width = selection_rect_width * 0.9
        # single_button_width = (ax_pos.width * 0.9) #0.2

        # single_button_width = (ax_pos.width * 0.9)/float(num_buttons) #0.2
        # single_button_height = (ax_pos.height * 0.085)  # Set button height
        # single_button_height = (ax_pos.height * 0.18)  # Set button height

        # button_x_start = ax_pos.x1+0.005 # Start where the axis ends less a tiny amount
        # button_x_start = ax_xmid - (float(single_button_width)/2.0) # centered
        
        # button_y_start = ax_pos.y0 + ax_pos.height - button_height  # Start at the top of the axis
        
        # rax = ax.inset_axes([1.0, 0.0, 0.1, 0.12])
        # button_positions = [(button_x_start, (button_y_start - i * button_height), button_width, button_height) for i in range(num_buttons)]

        ## Vertical ascending from bottom:
        # single_button_width = (ax_pos.width * 0.9)
        single_button_width = (ax_pos.width * 0.2)
        single_button_height = (ax_pos.height * 0.6)/float(num_buttons)
        button_x_start = ax_xmid - (float(single_button_width)/2.0) # centered
        button_y_start = ax_pos.y0
        # button_positions = [(button_x_start, (button_y_start + (i * single_button_height)), single_button_width, single_button_height) for i in range(num_buttons)]
        # print(f'button_positions: {button_positions}')

        # ## Horizontal stack:
        # single_button_width = (ax_pos.width * 0.9)/float(num_buttons)
        # single_button_height = (ax_pos.height * 0.3)
        # button_x_start = ax_pos.x0+0.05 # left
        # button_y_start = ax_pos.y0  # Start at the bottom of the figure
        # button_positions = [((button_x_start + (i * single_button_width)), button_y_start, single_button_width, single_button_height) for i in range(num_buttons)]

        button_stack_total_width = single_button_width  
        button_stack_total_height =  single_button_height * float(num_buttons)
        button_stack_group_position = (button_x_start, button_y_start, button_stack_total_width, button_stack_total_height)

        # Create an axes for the check button
        col = (0.95, 0.65, 0.65, 0.6)
        # col = 'none'
        check_ax = a_fig.add_axes(button_stack_group_position, facecolor=col)  # This should be a list of positions if multiple buttons are added. E.g. [0.7, 0.05, 0.1, 0.1]
        # check_ax = ax.inset_axes(button_positions, facecolor='none')
        action_buttons = CheckButtons(check_ax,
                            [a_btn_cfg['name'] for a_btn_cfg in action_button_configs],
                            [a_btn_cfg.get('value', False) for a_btn_cfg in action_button_configs]) 

        
        print(f'action_buttons: {action_buttons}')

        # for i, (a_btn_cfg, a_btn) in enumerate(zip(action_button_configs, action_buttons)):
        
        # Connect the event handler
        # action_buttons.on_clicked(toggle_label)
        
        # for i, (a_btn_cfg, a_pos) in enumerate(zip(action_button_configs, a_button_positions)):
        #     a_label = a_btn_cfg['name']

        #     # Create check buttons
        #     # Create an axes for the check button
        #     col = (0.95, 0.95, 0.95, 0.6)
        #     # col = 'none'
        #     check_ax = a_fig.add_axes(button_positions, facecolor=col)  # This should be a list of positions if multiple buttons are added. E.g. [0.7, 0.05, 0.1, 0.1]

        #     # check_ax = ax.inset_axes(button_positions, facecolor='none')

        #     check = CheckButtons(check_ax, labels, [False]*len(labels)) 

        #     action_buttons.append(add_check_buttons(a_fig, a_pos, labels=a_label))
    
    else:
        action_buttons = None

    if not defer_draw:
        a_fig.canvas.draw()

    return rectangle, action_buttons
        


def build_or_reuse_figure(fignum=1, fig=None, fig_idx:int=0, **kwargs):
    """ Reuses a Matplotlib figure if it exists, or creates a new one if needed
    Inputs:
        fignum - an int or str that identifies a figure
        fig - an existing Matplotlib figure
        fig_idx:int - an index to identify this figure as part of a series of related figures, e.g. plot_pf_1D[0], plot_pf_1D[1], ... 
        **kwargs - are passed as kwargs to the plt.figure(...) command when creating a new figure
    Outputs:
        fig: a Matplotlib figure object

    History: factored out of `plot_ratemap_2D`

    Usage:
        from neuropy.utils.matplotlib_helpers import build_or_reuse_figure
        
    Example 1:
        ## Figure Setup:
        fig = build_or_reuse_figure(fignum=kwargs.pop('fignum', None), fig=kwargs.pop('fig', None), fig_idx=kwargs.pop('fig_idx', 0), figsize=kwargs.pop('figsize', (10, 4)), dpi=kwargs.pop('dpi', None), constrained_layout=True) # , clear=True
        subfigs = fig.subfigures(actual_num_subfigures, 1, wspace=0.07)
        ##########################

    Example 2:
        
        if fignum is None:
            if f := plt.get_fignums():
                fignum = f[-1] + 1
            else:
                fignum = 1

        ## Figure Setup:
        if ax is None:
            fig = build_or_reuse_figure(fignum=fignum, fig=fig, fig_idx=0, figsize=(12, 4.2), dpi=None, clear=True, tight_layout=False)
            gs = GridSpec(1, 1, figure=fig)

            if use_brokenaxes_method:
                # `brokenaxes` method: DOES NOT YET WORK!
                from brokenaxes import brokenaxes ## Main brokenaxes import 
                pad_size: float = 0.1
                # [(a_tuple.start, a_tuple.stop) for a_tuple in a_test_epoch_df.itertuples(index=False, name="EpochTuple")]
                lap_start_stop_tuples_list = [((a_tuple.start - pad_size), (a_tuple.stop + pad_size)) for a_tuple in ensure_dataframe(laps_Epoch_obj).itertuples(index=False, name="EpochTuple")]
                # ax = brokenaxes(xlims=((0, .1), (.4, .7)), ylims=((-1, .7), (.79, 1)), hspace=.05, subplot_spec=gs[0])
                ax = brokenaxes(xlims=lap_start_stop_tuples_list, hspace=.05, subplot_spec=gs[0])
            else:
                ax = plt.subplot(gs[0])

        else:
            # otherwise get the figure from the passed axis
            fig = ax.get_figure()
                    
            
    """
    if fignum is None:
        if f := plt.get_fignums():
            fignum = f[-1] + 1
        else:
            fignum = 1

    ## Figure Setup:
    if fig is not None:
        # provided figure
        extant_fig = fig
    else:
        extant_fig = None # is this okay?
        
    if fig is not None:
        # provided figure
        active_fig_id = fig
    else:
        if isinstance(fignum, int):
            # a numeric fignum that can be incremented
            active_fig_id = fignum + fig_idx
        elif isinstance(fignum, str):
            # a string-type fignum.
            # TODO: deal with inadvertant reuse of figure? perhaps by appending f'{fignum}[{fig_ind}]'
            if fig_idx > 0:
                active_fig_id = f'{fignum}[{fig_idx}]'
            else:
                active_fig_id = fignum
        else:
            raise NotImplementedError
    
    if extant_fig is None:
        fig = plt.figure(active_fig_id, **({'dpi': None, 'clear': True} | kwargs)) # , 'tight_layout': False - had to remove 'tight_layout': False because it can't coexist with 'constrained_layout'
            #  UserWarning: The Figure parameters 'tight_layout' and 'constrained_layout' cannot be used together.
    else:
        fig = extant_fig
    return fig

def scale_title_label(ax, curr_title_obj, curr_im, debug_print=False):
    """ Scales some matplotlib-based figures titles to be reasonable. I remember that this was important and hard to make, but don't actually remember what it does as of 2022-10-24. It needs to be moved in to somewhere else.
    

    History: From PendingNotebookCode's 2022-11-09 section


    Usage:

        from neuropy.utils.matplotlib_helpers import scale_title_label

        ## Scale all:
        _display_outputs = widget.last_added_display_output
        curr_graphics_objs = _display_outputs.graphics[0]

        ''' curr_graphics_objs is:
        {2: {'axs': [<Axes:label='2'>],
        'image': <matplotlib.image.AxesImage at 0x1630c4556d0>,
        'title_obj': <matplotlib.offsetbox.AnchoredText at 0x1630c4559a0>},
        4: {'axs': [<Axes:label='4'>],
        'image': <matplotlib.image.AxesImage at 0x1630c455f70>,
        'title_obj': <matplotlib.offsetbox.AnchoredText at 0x1630c463280>},
        5: {'axs': [<Axes:label='5'>],
        'image': <matplotlib.image.AxesImage at 0x1630c463850>,
        'title_obj': <matplotlib.offsetbox.AnchoredText at 0x1630c463b20>},
        ...
        '''
        for aclu, curr_neuron_graphics_dict in curr_graphics_objs.items():
            curr_title_obj = curr_neuron_graphics_dict['title_obj'] # matplotlib.offsetbox.AnchoredText
            curr_title_text_obj = curr_title_obj.txt.get_children()[0] # Text object
            curr_im = curr_neuron_graphics_dict['image'] # matplotlib.image.AxesImage
            curr_ax = curr_neuron_graphics_dict['axs'][0]
            scale_title_label(curr_ax, curr_title_obj, curr_im)

    
    """
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke
    transform = ax.transData
    curr_label_extent = curr_title_obj.get_window_extent(ax.get_figure().canvas.get_renderer()) # Bbox([[1028.49144862968, 2179.0555555555566], [1167.86644862968, 2193.0555555555566]])
    curr_label_width = curr_label_extent.width # 139.375
    # curr_im.get_extent() # (-85.75619321393464, 112.57838773103435, -96.44772761274268, 98.62205280781535)
    img_extent = curr_im.get_window_extent(ax.get_figure().canvas.get_renderer()) # Bbox([[1049.76842294452, 2104.7727272727284], [1146.5894743148401, 2200.000000000001]])
    curr_img_width = img_extent.width # 96.82105137032022
    needed_scale_factor = curr_img_width / curr_label_width
    if debug_print:
        print(f'curr_label_width: {curr_label_width}, curr_img_width: {curr_img_width}, needed_scale_factor: {needed_scale_factor}')
    needed_scale_factor = min(needed_scale_factor, 1.0) # Only scale up, don't scale down
    
    curr_font_props = curr_title_obj.prop # FontProperties
    curr_font_size_pts = curr_font_props.get_size_in_points() # 10.0
    curr_scaled_font_size_pts = needed_scale_factor * curr_font_size_pts
    if debug_print:
        print(f'curr_font_size_pts: {curr_font_size_pts}, curr_scaled_font_size_pts: {curr_scaled_font_size_pts}')

    if isinstance(curr_title_obj, AnchoredText):
        curr_title_text_obj = curr_title_obj.txt.get_children()[0] # Text object
    else:
        curr_title_text_obj = curr_title_obj
    
    curr_title_text_obj.set_fontsize(curr_scaled_font_size_pts)
    font_foreground = 'white'
    # font_foreground = 'black'
    curr_title_text_obj.set_color(font_foreground)
    # curr_title_text_obj.set_fontsize(6)
    
    stroke_foreground = 'black'
    # stroke_foreground = 'gray'
    # stroke_foreground = 'orange'
    strokewidth = 4
    curr_title_text_obj.set_path_effects([withStroke(foreground=stroke_foreground, linewidth=strokewidth)])
    ## Disable path effects:
    # curr_title_text_obj.set_path_effects([])


def add_value_labels(ax, spacing=5, labels=None):
    """Add labels to the end (top) of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes of the plot to annotate.
        spacing (int): The distance between the labels and the bars.

    History:
        Factored out of `plot_short_v_long_pf1D_scalar_overlap_comparison` on 2023-03-28

    Usage:
        from neuropy.utils.matplotlib_helpers import add_value_labels
        # Call the function above. All the magic happens there.
        add_value_labels(ax, labels=x_labels) # 

    """

    # For each bar: Place a label
    for i, rect in enumerate(ax.patches):
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        if labels is None:
            label = "{:.1f}".format(y_value)
            # # Use cell ID (given by x position) as the label
            label = "{}".format(x_value)
        else:
            label = str(labels[i])
            
        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va,                      # Vertically align label differently for positive and negative values.
            color=rect.get_facecolor(),
            rotation=90)                      
                                        # 


def fit_both_axes(ax_lhs, ax_rhs):
    """ 2023-05-25 - Computes the x and y bounds needed to fit all data on both axes, and the actually applies these bounds to each. """
    def _subfn_compute_fitting_both_axes(ax_lhs, ax_rhs):
        """ computes the fitting x and y bounds for both axes to fit all the data. 
        
        >>> ((0.8970694235737637, 95.79803141394544),
            (-1.3658343711184302, 32.976028484630994))
        """
        fitting_xbounds = (min(*ax_lhs.get_xbound(), *ax_rhs.get_xbound()), max(*ax_lhs.get_xbound(), *ax_rhs.get_xbound())) 
        fitting_ybounds = (min(*ax_lhs.get_ybound(), *ax_rhs.get_ybound()), max(*ax_lhs.get_ybound(), *ax_rhs.get_ybound())) 
        return (fitting_xbounds, fitting_ybounds)

    fitting_xbounds, fitting_ybounds = _subfn_compute_fitting_both_axes(ax_lhs, ax_rhs)
    ax_lhs.set_xbound(*fitting_xbounds)
    ax_lhs.set_ybound(*fitting_ybounds)
    ax_rhs.set_xbound(*fitting_xbounds)
    ax_rhs.set_ybound(*fitting_ybounds)
    return (fitting_xbounds, fitting_ybounds)


# ==================================================================================================================== #
# Advanced Text Helpers via `flexitext` library                                                                        #
# ==================================================================================================================== #
from attrs import define, field
from flexitext import flexitext ## flexitext is an advanced text library used in `FormattedFigureText`

@define(slots=False)
class FigureMargins:
    top_margin: float = 0.8
    left_margin: float = 0.15
    right_margin: float = 0.85 # (1.0-0.15)
    bottom_margin: float = 0.150

    
    

# left_margin: float = 0.090
# right_margin: float = 0.91 # (1.0-0.090)


# left_margin: float = 0.090
# right_margin: float = 0.91 # (1.0-0.090)

@define(slots=False)
class FormattedFigureText:
    """ builds flexitext matplotlib figure title and footers 

    Consistent color scheme:
        Long: Red
        Short: Blue

        Context footer is along the bottom of the figure in gray.


    Usage:
        
        from neuropy.utils.matplotlib_helpers import FormattedFigureText

        # `flexitext` version:
        text_formatter = FormattedFigureText()
        plt.title('')
        plt.suptitle('')
        text_formatter.setup_margins(fig)

        ## Need to extract the track name ('maze1') for the title in this plot. 
        track_name = active_context.get_description(subset_includelist=['filter_name'], separator=' | ') # 'maze1'
        # TODO: do we want to convert this into "long" or "short"?
        header_text_obj = flexitext(text_formatter.left_margin, text_formatter.top_margin, f'<size:22><weight:bold>{track_name}</> replay|laps <weight:bold>firing rate</></>', va="bottom", xycoords="figure fraction")
        footer_text_obj = flexitext((text_formatter.left_margin*0.1), (text_formatter.bottom_margin*0.25), text_formatter._build_footer_string(active_context=active_context), va="top", xycoords="figure fraction")



    """
    # fig.subplots_adjust(top=top_margin, left=left_margin, bottom=bottom_margin)

    margins: FigureMargins = field(factory=FigureMargins)

    @property
    def top_margin(self):
        return self.margins.top_margin
    @top_margin.setter
    def top_margin(self, value):
        self.margins.top_margin = value

    @property
    def left_margin(self):
        return self.margins.left_margin
    @left_margin.setter
    def left_margin(self, value):
        self.margins.left_margin = value

    @property
    def right_margin(self):
        return self.margins.right_margin
    @right_margin.setter
    def right_margin(self, value):
        self.margins.right_margin = value

    @property
    def bottom_margin(self):
        return self.margins.bottom_margin
    @bottom_margin.setter
    def bottom_margin(self, value):
        self.margins.bottom_margin = value


    @classmethod
    def init_from_margins(cls, top_margin=None, left_margin=None, right_margin=None, bottom_margin=None) -> "FormattedFigureText":
        """ allows initializing while overriding specific margins 
        
        text_formatter = FormattedFigureText.init_from_margins(left_margin=0.01)
        
        """
        _obj = cls()
        if top_margin is not None:
            _obj.top_margin = top_margin
        if left_margin is not None:
            _obj.left_margin = left_margin
        if right_margin is not None:
            _obj.right_margin = right_margin
        if bottom_margin is not None:
            _obj.bottom_margin = bottom_margin
        return _obj
        
    @classmethod
    def _build_formatted_title_string(cls, epochs_name) -> str:
        """ buidls the two line colored string figure's footer that is passed into `flexitext`.
        """
        return (f"<size:22><weight:bold>{epochs_name}</> Firing Rates\n"
                "<size:14>for the "
                "<color:crimson, weight:bold>Long</>/<color:royalblue, weight:bold>Short</> eXclusive Cells on each track</></>"
                )

    @classmethod
    def _build_footer_string(cls, active_context) -> str:
        """ buidls the dim, grey string for the figure's footer that is passed into `flexitext`.
        Usage:
            footer_text_obj = flexitext((left_margin*0.1), (bottom_margin*0.25), cls._build_footer_string(active_context=active_context), va="top", xycoords="figure fraction")
        """
        if hasattr(active_context, 'get_specific_purpose_description'):
            footer_string = active_context.get_specific_purpose_description(specific_purpose='flexitext_footer')
            return footer_string
        else:
            first_portion_sess_ctxt_str = active_context.get_description(subset_includelist=['format_name', 'animal', 'exper_name'], separator=' | ')
            session_name_sess_ctxt_str = active_context.get_description(subset_includelist=['session_name'], separator=' | ') # 2006-6-08_14-26-15
            return (f"<color:silver, size:10>{first_portion_sess_ctxt_str} | <weight:bold>{session_name_sess_ctxt_str}</></>")


    def setup_margins(self, fig, **kwargs):
        top_margin, left_margin, right_margin, bottom_margin = kwargs.get('top_margin', self.top_margin), kwargs.get('left_margin', self.left_margin), kwargs.get('right_margin', self.right_margin), kwargs.get('bottom_margin', self.bottom_margin)

        layout_engine = fig.get_layout_engine()
        if (layout_engine is None) or (fig.get_layout_engine().adjust_compatible):
            fig.subplots_adjust(top=top_margin, left=left_margin, right=right_margin, bottom=bottom_margin) # perform the adjustment on the figure
        else:
            # new function works for 'constrained' and 'tight' layouts
            set_margins(fig, top=top_margin, left=left_margin, right=right_margin, bottom=bottom_margin) # perform the adjustment on the figure
    

    def add_flexitext_context_footer(self, active_context, override_left_margin_multipler:float=0.1, override_bottom_margin_multiplier:float=0.25):
        """ adds the default footer  """
        return flexitext((self.left_margin*float(override_left_margin_multipler)), (self.bottom_margin*float(override_bottom_margin_multiplier)), self._build_footer_string(active_context=active_context), va="top", xycoords="figure fraction")



    def add_flexitext(self, fig, active_context, **kwargs):
        self.setup_margins(fig, **kwargs)
        # Add flexitext
        top_margin, left_margin, bottom_margin = kwargs.get('top_margin', self.top_margin), kwargs.get('left_margin', self.left_margin), kwargs.get('bottom_margin', self.bottom_margin)
        title_text_obj = flexitext(left_margin, top_margin, 'long ($L$)|short($S$) firing rate indicies', va="bottom", xycoords="figure fraction")
        footer_text_obj = flexitext((self.left_margin*0.1), (self.bottom_margin*0.25), self._build_footer_string(active_context=active_context), va="top", xycoords="figure fraction")
        return title_text_obj, footer_text_obj


    @classmethod
    def clear_basic_titles(self, fig):
        """ clears the basic title and suptitle in preparation for the flexitext version. """
        plt.figure(fig)
        plt.title('')
        plt.suptitle('')
        






#     ## Figure computation
#     fig: plt.Figure = ax.get_figure()
#     dpi = fig.dpi
#     rect_height_inch = rect_height / dpi
#     # Initial fontsize according to the height of boxes
#     fontsize = rect_height_inch * 72
#     print(f'rect_height_inch: {rect_height_inch}, fontsize: {fontsize}')

# #     text: Annotation = ax.annotate(txt, xy, ha=ha, va=va, xycoords=transform, **kwargs)

# #     # Adjust the fontsize according to the box size.
# #     text.set_fontsize(fontsize)
#     bbox: Bbox = text.get_window_extent(fig.canvas.get_renderer())
#     adjusted_size = fontsize * rect_width / bbox.width
#     print(f'bbox: {bbox}, adjusted_size: {adjusted_size}')
#     text.set_fontsize(adjusted_size)


def plot_position_curves_figure(position_obj, include_velocity=True, include_accel=False, figsize=(24, 10)):
    """ Renders a figure with a position curve and optionally its higher-order derivatives """
    num_subplots = 1
    out_axes_list = []
    if include_velocity:
        num_subplots = num_subplots + 1
    if include_accel:
        num_subplots = num_subplots + 1
    subplots=(num_subplots, 1)
    fig = plt.figure(figsize=figsize, clear=True)
    gs = plt.GridSpec(subplots[0], subplots[1], figure=fig, hspace=0.02)
    
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(position_obj.time, position_obj.x, 'k')
    ax0.set_ylabel('pos_x')
    out_axes_list.append(ax0)
    
    prev_axis = ax0

    if include_velocity:
        ax1 = fig.add_subplot(gs[1])
        # ax1.plot(position_obj.time, pos_df['velocity_x'], 'grey')
        # ax1.plot(position_obj.time, pos_df['velocity_x_smooth'], 'r')
        ax1.plot(position_obj.time, position_obj._data['velocity_x_smooth'], 'k')
        ax1.set_ylabel('Velocity_x')
        ax0.set_xticklabels([]) # this is intensionally ax[i-1], as we want to disable the tick labels on above plots        
        out_axes_list.append(ax1)
        # share x axis
        ax1.sharex(prev_axis)
        prev_axis = ax1

    if include_accel:  
        ax2 = fig.add_subplot(gs[2])
        # ax2.plot(position_obj.time, position_obj.velocity)
        # ax2.plot(position_obj.time, pos_df['velocity_x'])
        ax2.plot(position_obj.time, position_obj._data['acceleration_x'], 'k')
        # ax2.plot(position_obj.time, pos_df['velocity_y'])
        ax2.set_ylabel('Higher Order Terms')
        ax1.set_xticklabels([]) # this is intensionally ax[i-1], as we want to disable the tick labels on above plots
        out_axes_list.append(ax2)
        # share x axis
        ax2.sharex(prev_axis)
        prev_axis = ax2

    # Shared:
    # ax0.get_shared_x_axes().join(ax0, ax1)
    # ax0.get_shared_x_axes().join(*out_axes_list) # this was removed for some reason! AttributeError: 'GrouperView' object has no attribute 'join'
    ax0.set_xticklabels([])
    ax0.set_xlim([position_obj.time[0], position_obj.time[-1]])

    return fig, out_axes_list

    


# ==================================================================================================================== #
# 2022-12-14 Batch Surprise Recomputation                                                                              #
# ==================================================================================================================== #



def _subfn_build_epoch_region_label(xy, text, ax, **labels_kwargs):
    """ places a text label inside a square area the top, just inside of it 
    the epoch at

    Used by:
        draw_epoch_regions: to draw the epoch name inside the epoch
    """
    if labels_kwargs is None:
        labels_kwargs = {}
    labels_y_offset = labels_kwargs.pop('y_offset', -0.05)
    # y = xy[1]
    y = xy[1] + labels_y_offset  # shift y-value for label so that it's below the artist
    return ax.text(xy[0], y, text, **({'ha': 'center', 'va': 'top', 'family': 'sans-serif', 'size': 14, 'rotation': 0} | labels_kwargs)) # va="top" places it inside the box if it's aligned to the top

# @function_attributes(short_name='draw_epoch_regions', tags=['epoch','matplotlib','helper'], input_requires=[], output_provides=[], uses=['BrokenBarHCollection'], used_by=[], creation_date='2023-03-28 14:23')
def draw_epoch_regions(epoch_obj, curr_ax, facecolor=('green','red'), edgecolors=("black",), alpha=0.25, labels_kwargs=None, defer_render=False, debug_print=False, **kwargs):
    """ plots epoch rectangles with customizable color, edgecolor, and labels on an existing matplotlib axis
    2022-12-14

    Info:
    
    https://matplotlib.org/stable/tutorials/intermediate/autoscale.html
    
    Usage:
        from neuropy.utils.matplotlib_helpers import draw_epoch_regions
        epochs_collection, epoch_labels = draw_epoch_regions(curr_active_pipeline.sess.epochs, ax, defer_render=False, debug_print=False)

    Full Usage Examples:

    ## Example 1:
        active_filter_epochs = curr_active_pipeline.sess.replay
        active_filter_epochs

        if not 'stop' in active_filter_epochs.columns:
            # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
            active_filter_epochs['stop'] = active_filter_epochs['end'].copy()
            
        if not 'label' in active_filter_epochs.columns:
            # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
            active_filter_epochs['label'] = active_filter_epochs['flat_replay_idx'].copy()

        active_filter_epoch_obj = Epoch(active_filter_epochs)
        active_filter_epoch_obj


        fig, ax = plt.subplots()
        ax.plot(post_update_times, flat_surprise_across_all_positions)
        ax.set_ylabel('Relative Entropy across all positions')
        ax.set_xlabel('t (seconds)')
        epochs_collection, epoch_labels = draw_epoch_regions(curr_active_pipeline.sess.epochs, ax, facecolor=('red','cyan'), alpha=0.1, edgecolors=None, labels_kwargs={'y_offset': -0.05, 'size': 14}, defer_render=True, debug_print=False)
        laps_epochs_collection, laps_epoch_labels = draw_epoch_regions(curr_active_pipeline.sess.laps.as_epoch_obj(), ax, facecolor='red', edgecolors='black', labels_kwargs={'y_offset': -16.0, 'size':8}, defer_render=True, debug_print=False)
        replays_epochs_collection, replays_epoch_labels = draw_epoch_regions(active_filter_epoch_obj, ax, facecolor='orange', edgecolors=None, labels_kwargs=None, defer_render=False, debug_print=False)
        fig.show()


    ## Example 2:

        # Show basic relative entropy vs. time plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(post_update_times, flat_relative_entropy_results)
        ax.set_ylabel('Relative Entropy')
        ax.set_xlabel('t (seconds)')
        epochs_collection, epoch_labels = draw_epoch_regions(curr_active_pipeline.sess.epochs, ax, defer_render=False, debug_print=False)
        fig.show()

    """
    # epoch_obj
    def _subfn_perform_plot_epochs(an_epoch_tuples, an_ax):
        curr_span_ymin = an_ax.get_ylim()[0]
        curr_span_ymax = an_ax.get_ylim()[1]
        curr_span_height = curr_span_ymax - curr_span_ymin
        # xrange: list of (float, float) The sequence of (left-edge-position, width) pairs for each bar.
        # yrange: (lower-edge, height) 
        epochs_collection = BrokenBarHCollection(xranges=an_epoch_tuples, yrange=(curr_span_ymin, curr_span_height), facecolor=facecolor, alpha=alpha, edgecolors=edgecolors, linewidths=(1,), **kwargs) # , offset_transform=curr_ax.transData
        if debug_print:
            print(f'(curr_span_ymin, curr_span_ymax): ({curr_span_ymin}, {curr_span_ymax}), an_epoch_tuples: {an_epoch_tuples}')
        an_ax.add_collection(epochs_collection)
        return epochs_collection
        

    epoch_tuples = [(start_t, width_duration) for start_t, width_duration in zip(epoch_obj.starts, epoch_obj.durations)] # [(0.0, 1211.5580800310709), (1211.5580800310709, 882.3397767931456)]
    epoch_mid_t = [a_tuple[0]+(0.5*a_tuple[1]) for a_tuple in epoch_tuples] # used for labels
    num_epochs: int = len(epoch_tuples)


    if isinstance(curr_ax.get_ylim()[0], tuple):
        ## strange case for brokenaxes
        a_ylim_tuple = curr_ax.get_ylim()[0]
        curr_span_ymin = a_ylim_tuple[0]
        curr_span_ymax = a_ylim_tuple[1]

        ## num xlims:
        # num_brokenaxes: int = len(curr_ax.get_ylim())
        num_brokenaxes: int = len(curr_ax.axs)
        assert num_epochs == num_brokenaxes, f"num_epochs: {num_epochs} != num_brokenaxes: {num_brokenaxes}"

        epoch_collection_list = [] # a list
        # epoch_labels_list = [] # a list
        for i, an_ax in enumerate(curr_ax.axs):
            # can plot on a single axis:
            epochs_collection = _subfn_perform_plot_epochs(an_epoch_tuples=[epoch_tuples[i]], an_ax=an_ax) # single element lists, ValueError: Can not reset the axes.  You are probably trying to re-use an artist in more than one Axes which is not supported
            # epochs_collection = _subfn_perform_plot_epochs(an_epoch_tuples=[epoch_tuples[i]], an_ax=curr_ax.big_ax) # same single (big axis: curr_ax.big_ax)
            epoch_collection_list.append(epochs_collection)
        
            # epoch_mid_t = [epoch_mid_t[i]]

            # if labels_kwargs is not None:
            #     epoch_labels = [_subfn_build_epoch_region_label((a_mid_t, curr_span_ymax), a_label, an_ax, **labels_kwargs) for a_label, a_mid_t in zip(epoch_obj.labels, epoch_mid_t)]
            # else:
            #     epoch_labels = None
            # epoch_labels_list.append(epoch_labels)
        # END FOR curr_ax.axs
            
        if labels_kwargs is not None:
            epoch_labels = [_subfn_build_epoch_region_label((a_mid_t, curr_span_ymax), a_label, curr_ax.big_ax, **labels_kwargs) for a_label, a_mid_t in zip(epoch_obj.labels, epoch_mid_t)]
        else:
            epoch_labels = None
            
        if not defer_render:
            curr_ax.get_figure().canvas.draw()

        return epoch_collection_list, epoch_labels


    else:

        # can plot on a single axis:
        epochs_collection = _subfn_perform_plot_epochs(an_epoch_tuples=epoch_tuples, an_ax=curr_ax)
        if labels_kwargs is not None:
            a_ylim_tuple = curr_ax.get_ylim()
            curr_span_ymin = a_ylim_tuple[0]
            curr_span_ymax = a_ylim_tuple[1]
            epoch_labels = [_subfn_build_epoch_region_label((a_mid_t, curr_span_ymax), a_label, curr_ax, **labels_kwargs) for a_label, a_mid_t in zip(epoch_obj.labels, epoch_mid_t)]
        else:
            epoch_labels = None
        

        if not defer_render:
            curr_ax.get_figure().canvas.draw()

        return epochs_collection, epoch_labels


def plot_overlapping_epoch_analysis_diagnoser(position_obj, epoch_obj):
    """ builds a MATPLOTLIB figure showing the position and velocity overlayed by the epoch intervals in epoch_obj. Useful for diagnosing overlapping epochs.
    Usage:
        from neuropy.utils.matplotlib_helpers import plot_overlapping_epoch_analysis_diagnoser
        fig, out_axes_list = plot_overlapping_epoch_analysis_diagnoser(sess.position, curr_active_pipeline.sess.laps.as_epoch_obj())
    """
    fig, out_axes_list = plot_position_curves_figure(position_obj, include_velocity=True, include_accel=False, figsize=(24, 10))
    for ax in out_axes_list:
        laps_epochs_collection, laps_epoch_labels = draw_epoch_regions(epoch_obj, ax, facecolor=[(255, 0, 0), (0, 255, 0)], edgecolors=(0,0,0), labels_kwargs={'y_offset': -16.0, 'size':8, 'rotation':90}, defer_render=False, debug_print=False)
    fig.show()
    return fig, out_axes_list


# ==================================================================================================================== #
# 2023-05-09 Misc Utility Functions                                                                                    #
# ==================================================================================================================== #

from matplotlib.figure import Figure # used in `MatplotlibFigureExtractors`

class MatplotlibFigureExtractors:
    """ 2023-06-26 - Unfinished class that aims to extract matplotlib.figure properties and settings.
    Usage:
        from neuropy.utils.matplotlib_helpers import MatplotlibFigureExtractors
    
    """
    @staticmethod
    def extract_figure_properties(fig):
        """ UNTESTED, UNFINISHED
        Extracts styles, formatting, and set options from a matplotlib Figure object.
        Returns a dictionary with the following keys:
            - 'title': the Figure title (if any)
            - 'xlabel': the label for the x-axis (if any)
            - 'ylabel': the label for the y-axis (if any)
            - 'xlim': the limits for the x-axis (if any)
            - 'ylim': the limits for the y-axis (if any)
            - 'xscale': the scale for the x-axis (if any)
            - 'yscale': the scale for the y-axis (if any)
            - 'legend': the properties of the legend (if any)
            - 'grid': the properties of the grid (if any)
            
        TO ADD:
            -   fig.get_figwidth()
                fig.get_figheight()
                # fig.set_figheight()

                print(f'fig.get_figwidth(): {fig.get_figwidth()}\nfig.get_figheight(): {fig.get_figheight()}')


            
            Usage:        
                curr_fig = plt.gcf()
                curr_fig = out.figures[0]
                curr_fig_properties = extract_figure_properties(curr_fig)
                curr_fig_properties

        """
        properties = {}
        
        # Extract title
        properties['title'] = fig._suptitle.get_text() if fig._suptitle else None
        
        # Extract axis labels and limits
        for ax in fig.get_axes():
            if ax.get_label() == 'x':
                properties['xlabel'] = ax.get_xlabel()
                properties['xlim'] = ax.get_xlim()
                properties['xscale'] = ax.get_xscale()
            elif ax.get_label() == 'y':
                properties['ylabel'] = ax.get_ylabel()
                properties['ylim'] = ax.get_ylim()
                properties['yscale'] = ax.get_yscale()
        
        # Extract legend properties
        if hasattr(fig, 'legend_'):
            legend = fig.legend_
            if legend:
                properties['legend'] = {
                    'title': legend.get_title().get_text(),
                    'labels': [t.get_text() for t in legend.get_texts()],
                    'loc': legend._loc,
                    'frameon': legend.get_frame_on(),
                }
        
        # Extract grid properties
        first_ax = fig.axes[0]
        grid = first_ax.get_gridlines()[0] if first_ax.get_gridlines() else None
        if grid:
            properties['grid'] = {
                'color': grid.get_color(),
                'linestyle': grid.get_linestyle(),
                'linewidth': grid.get_linewidth(),
            }
        
        return properties

    @classmethod
    def extract_fig_suptitle(cls, fig: Figure):
        """To get the figure's suptitle Text object: https://stackoverflow.com/questions/48917631/matplotlib-how-to-return-figure-suptitle

        Usage:
            from matplotlib.figure import Figure
            from neuropy.utils.matplotlib_helpers import MatplotlibFigureExtractors

            sup, suptitle_string = MatplotlibFigureExtractors.extract_fig_suptitle(fig)
            suptitle_string

        """
        sup = fig._suptitle # Text(0.5, 0.98, 'kdiba/gor01/one/2006-6-08_14-26-15/long_short_firing_rate_indicies/display_long_short_laps')
        if sup is not None:
            suptitle_string: str = sup._text # 'kdiba/gor01/one/2006-6-08_14-26-15/long_short_firing_rate_indicies/display_long_short_laps'
        else: 
            suptitle_string = None
            
        return sup, suptitle_string

    @classmethod
    def extract_titles(cls, fig: Optional[Figure]=None):
        """ 
        # Call the function to extract titles
            captured_titles = extract_titles()
            print(captured_titles)
        """
        fig = fig or (plt.gcf())
        titles = {}
        
        # Get the window title
        # fig.canvas.manager # .set_window_title(title_string) # sets the window's title
        

        # titles['window_title'] = fig.canvas.manager.window.title()

        
        # titles['window_title'] = plt.gcf().canvas.get_window_title()
        try:
            titles['window_title'] = fig.canvas.manager.window.windowTitle()
        except AttributeError as e:
            try:
                titles['window_title'] = fig.canvas.get_window_title() # try this one
            except Exception as e:
                try:
                    titles['window_title'] = f"{fig.number or ''}"
                except Exception as e:
                    raise e # unhandled exception
        except Exception as e:
            raise e
        
        # Get the suptitle
        suptitle = fig._suptitle.get_text() if fig._suptitle else None
        titles['suptitle'] = suptitle
        
        # Get the titles of each axis
        axes = fig.get_axes()
        for i, ax in enumerate(axes):
            title = ax.get_title()
            titles[f'axis_title_{i+1}'] = title
        
        return titles

# ==================================================================================================================== #
# 2023-06-05 Interactive Selection Helpers                                                                             #
# ==================================================================================================================== #
def add_range_selector(fig, ax, initial_selection=None, orientation="horizontal", on_selection_changed=None) -> SpanSelector:
    """ 2023-06-06 - a 1D version of `add_rectangular_selector` which adds a selection band to an existing axis

    from neuropy.utils.matplotlib_helpers import add_range_selector
    curr_pos = deepcopy(curr_active_pipeline.sess.position)
    curr_pos_df = curr_pos.to_dataframe()

    curr_pos_df.plot(x='t', y=['lin_pos'])
    fig, ax = plt.gcf(), plt.gca()
    range_selector, set_extents = add_range_selector(fig, ax, orientation="vertical", initial_selection=None) # (-86.91, 141.02)

    """
    assert orientation in ["horizontal", "vertical"]
    use_midline = False

    if use_midline:
        def update_mid_line(xmin, xmax):
            xmid = np.mean([xmin, xmax])
            mid_line.set_ydata(xmid)

        def on_move_callback(xmin, xmax):
            """ Callback whenever the range is moved. 

            """
            print(f'on_move_callback(xmin: {xmin}, xmax: {xmax})')
            update_mid_line(xmin, xmax)
    else:
        on_move_callback = None

    def select_callback(xmin, xmax):
        """
        Callback for range selection.
        """
        # indmin, indmax = np.searchsorted(x, (xmin, xmax))
        # indmax = min(len(x) - 1, indmax)
        print(f"({xmin:3.2f}, {xmax:3.2f})")
        if on_selection_changed is not None:
            """ call the user-provided callback """
            on_selection_changed(xmin, xmax)
        
    if initial_selection is not None:
        # convert to extents:
        (x0, x1) = initial_selection # initial_selection should be `(xmin, xmax)`
        extents = (min(x0, x1), max(x0, x1))
    else:
        extents = None
        
    props=dict(alpha=0.5, facecolor="tab:red")
    selector = SpanSelector(ax, select_callback, orientation, useblit=True, props=props, interactive=True, drag_from_anywhere=True, onmove_callback=on_move_callback) # Set useblit=True on most backends for enhanced performance.
    if extents is not None:
        selector.extents = extents
    
    ## Add midpoint line:
    if use_midline:
        mid_line = ax.axhline(linewidth=1, alpha=0.6, color='r', label='midline', linestyle="--")
        update_mid_line(*selector.extents)

    def set_extents(selection):
        """ can be called to set the extents on the selector object. Captures `selector` """
        if selection is not None:
            (x0, x1) = selection # initial_selection should be `(xmin, xmax)`
            extents = (min(x0, x1), max(x0, x1))
            selector.extents = extents
            
    return selector, set_extents

def add_rectangular_selector(fig, ax, initial_selection=None, on_selection_changed=None, selection_rect_props=None, **kwargs) -> RectangleSelector:
    """ 2023-05-16 - adds an interactive rectangular selector to an existing matplotlib figure/ax.
    
    Usage:
    
        from neuropy.utils.matplotlib_helpers import add_rectangular_selector

        fig, ax = curr_active_pipeline.computation_results['maze'].computed_data.pf2D.plot_occupancy()
        rect_selector, set_extents = add_rectangular_selector(fig, ax, initial_selection=grid_bin_bounds) # (24.82, 257.88), (125.52, 149.19)

    
    The returned RectangleSelector object can have its selection accessed via:
        rect_selector.extents # (25.508610487986658, 258.5627661142404, 128.10121504465053, 150.48449186696848)
    
    Or updated via:
        rect_selector.extents = (25, 258, 128, 150)

    """
    def select_callback(eclick, erelease):
        """
        Callback for line selection.

        *eclick* and *erelease* are the press and release events.
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
        print(f'({x1:3.2f}, {x2:3.2f}), ({y1:3.2f}, {y2:3.2f})')
        print(f"The buttons you used were: {eclick.button} {erelease.button}")
        if on_selection_changed is not None:
            """ call the user-provided callback """
            extents = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
            on_selection_changed(extents)

        
    if initial_selection is not None:
        if len(initial_selection) == 4:
            # extents format `(xmin, xmax, ymin, ymax)`
            x0, x1, y0, y1 = initial_selection
        elif len(initial_selection) == 2:
            # pairs format: `((xmin, xmax), (ymin, ymax))`
            assert len(initial_selection[0]) == 2, f"initial_selection should be `((xmin, xmax), (ymin, ymax))` but it is: {initial_selection}"
            assert len(initial_selection[1]) == 2, f"initial_selection should be `((xmin, xmax), (ymin, ymax))` but it is: {initial_selection}"
            # convert to extents:
            (x0, x1), (y0, y1) = initial_selection # initial_selection should be `((xmin, xmax), (ymin, ymax))`
        extents = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
    else:
        extents = None
        
    initial_extents = extents  # Store the initial extents
    # ax = axs[0]
    # props = dict(facecolor='blue', alpha=0.5)
    if selection_rect_props is None:
        selection_rect_props = None
    selector = RectangleSelector(ax, select_callback, props=selection_rect_props, **({'useblit': True, 'button': [1, 3], 'minspanx': 5, 'minspany': 5, 'spancoords': 'data', 'interactive': True, 'ignore_event_outside': True} | kwargs)) # spancoords='pixels', button=[1, 3]: disable middle button 
    # useblit=True, button=[1, 3], minspanx=5, minspany=5, spancoords='data', interactive=True, ignore_event_outside=True
    if extents is not None:
        selector.extents = extents
    # fig.canvas.mpl_connect('key_press_event', toggle_selector)
    def set_extents(selection):
        """ can be called to set the extents on the selector object. Captures `selector` """
        if selection is not None:
            (x0, x1), (y0, y1) = selection # initial_selection should be `((xmin, xmax), (ymin, ymax))`
            extents = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
            selector.extents = extents
            
    def reset_extents():
        """Reset the selector to the initial extents. Captures `initial_extents`."""
        selector.extents = initial_extents
        
    return selector, set_extents, reset_extents


# grid_bin_bounds updating versions __________________________________________________________________________________ #

def interactive_select_grid_bin_bounds_1D(curr_active_pipeline, epoch_name='maze'):
    """ allows the user to interactively select the grid_bin_bounds for the pf1D
    
    Usage:
        from neuropy.utils.matplotlib_helpers import interactive_select_grid_bin_bounds_1D
        fig, ax, range_selector, set_extents = interactive_select_grid_bin_bounds_1D(curr_active_pipeline, epoch_name='maze')
    """
    # from neuropy.utils.matplotlib_helpers import add_range_selector
    computation_result = curr_active_pipeline.computation_results[epoch_name]
    grid_bin_bounds_1D = computation_result.computation_config['pf_params'].grid_bin_bounds_1D
    fig, ax = computation_result.computed_data.pf1D.plot_occupancy() #plot_occupancy()
    # curr_pos = deepcopy(curr_active_pipeline.sess.position)
    # curr_pos_df = curr_pos.to_dataframe()
    # curr_pos_df.plot(x='t', y=['lin_pos'])
    # fig, ax = plt.gcf(), plt.gca()

    def _on_range_changed(xmin, xmax):
        # print(f'xmin: {xmin}, xmax: {xmax}')
        # xmid = np.mean([xmin, xmax])
        # print(f'xmid: {xmid}')
        print(f'new_grid_bin_bounds_1D: ({xmin}, {xmax})')

    # range_selector, set_extents = add_range_selector(fig, ax, orientation="vertical", initial_selection=grid_bin_bounds_1D, on_selection_changed=_on_range_changed) # (-86.91, 141.02)
    range_selector, set_extents = add_range_selector(fig, ax, orientation="horizontal", initial_selection=grid_bin_bounds_1D, on_selection_changed=_on_range_changed)
    return fig, ax, range_selector, set_extents


def interactive_select_grid_bin_bounds_2D(curr_active_pipeline, epoch_name='maze', should_block_for_input:bool=True, should_apply_updates_to_pipeline=True, selection_rect_props=None, **kwargs):
    """ allows the user to interactively select the grid_bin_bounds for the pf2D
    Uses:
        plot_occupancy, add_rectangular_selector


    Usage:
        from neuropy.utils.matplotlib_helpers import interactive_select_grid_bin_bounds_2D
        fig, ax, rect_selector, set_extents, reset_extents = interactive_select_grid_bin_bounds_2D(curr_active_pipeline, epoch_name='maze')
    """
    # from neuropy.utils.matplotlib_helpers import add_rectangular_selector # interactive_select_grid_bin_bounds_2D
    computation_result = curr_active_pipeline.computation_results[epoch_name]
    grid_bin_bounds = computation_result.computation_config['pf_params'].grid_bin_bounds
    epoch_context = curr_active_pipeline.filtered_contexts[epoch_name]
                    
    fig, ax = computation_result.computed_data.pf2D.plot_occupancy(identifier_details_list=[epoch_name], active_context=epoch_context) 

    rect_selector, set_extents, reset_extents = add_rectangular_selector(fig, ax, initial_selection=grid_bin_bounds, selection_rect_props=selection_rect_props, **kwargs) # (24.82, 257.88), (125.52, 149.19)
    
    # Add a close event handler to break the while loop when the figure is manually closed
    was_closed = False
    def on_close(event):
        nonlocal was_closed
        was_closed = True
    fig.canvas.mpl_connect('close_event', on_close)
    
    # Add a key press event handler to reset the selector when 'r' is pressed
    def on_key_press(event):
        if event.key == 'r':
            print(f'resetting extents.')
            reset_extents()
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    def _on_update_grid_bin_bounds(new_grid_bin_bounds):
        """ called to update the grid_bin_bounds for all filtered_epochs with the new values (new_grid_bin_bounds) 
        Captures: `curr_active_pipeline`
        """
        print(f'_on_update_grid_bin_bounds(new_grid_bin_bounds: {new_grid_bin_bounds})')
        # does it also need to update session or anything?
        for epoch_name, computation_result in curr_active_pipeline.computation_results.items():
            computation_result.computation_config['pf_params'].grid_bin_bounds = new_grid_bin_bounds
                
    if should_block_for_input:
        print(f'blocking and waiting for user input. Press [enter] to confirm selection change or [esc] to revert with no change.')
        # hold plot until a keyboard key is pressed
        keyboardClick = False
        while keyboardClick != True:
            if was_closed:
                # Figure was manually closed, break the loop
                print(f'Figure was manually closed, break the loop.')
                break
            keyboardClick = plt.waitforbuttonpress() # plt.waitforbuttonpress() exits the inactive state as soon as either a key is pressed or the Mouse is clicked. However, the function returns True if a keyboard key was pressed and False if a Mouse was clicked
            if keyboardClick:
                # Button was pressed
                # if plt.get_current_fig_manager().toolbar.mode == '':
                # [Enter] was pressed
                confirmed_extents = rect_selector.extents
                print(f'user confirmed extents: {confirmed_extents}')
                if confirmed_extents is not None:
                    if should_apply_updates_to_pipeline:
                        _on_update_grid_bin_bounds(confirmed_extents) # update the grid_bin_bounds.
                    x0, x1, y0, y1 = confirmed_extents
                    print(f"Add this to `specific_session_override_dict`:\n\n{curr_active_pipeline.get_session_context().get_initialization_code_string()}:dict(grid_bin_bounds=({(x0, x1), (y0, y1)})),\n")
                    
                plt.close() # close the figure
                return confirmed_extents
                # elif plt.get_current_fig_manager().toolbar.mode == '':
                #     # [Esc] was pressed
                #     print(f'user canceled selection with [Esc].')
                #     plt.close()
                #     return grid_bin_bounds
    else:
        return fig, ax, rect_selector, set_extents, reset_extents


# Title Helpers ______________________________________________________________________________________________________ #
def perform_update_title_subtitle(fig=None, ax=None, title_string:Optional[str]=None, subtitle_string:Optional[str]=None, active_context=None, use_flexitext_titles=False):
    """ Only updates the title/subtitle if the value is not None
    
    Usage:
    
    from neuropy.utils.matplotlib_helpers import perform_update_title_subtitle
    perform_update_title_subtitle(fig=fig_long_pf_1D, ax=ax_long_pf_1D, title_string="TEST - 1D Placemaps", subtitle_string="TEST - SUBTITLE")
    
    """
    if fig is None:
        fig = plt.gcf()
        
    if ax is None:
        ax = plt.gca()
            

    if title_string is not None:
        fig.canvas.manager.set_window_title(title_string) # sets the window's title

    if (active_context is None) or (not use_flexitext_titles):
        if title_string is not None:
            fig.suptitle(title_string, fontsize='14', wrap=True)
        if (subtitle_string is not None) and (ax is not None):
            ax.set_title(subtitle_string, fontsize='10', wrap=True) # this doesn't appear to be visible, so what is it used for?

        footer_text_obj = None
    else:
        from flexitext import flexitext ## flexitext version
        from neuropy.utils.matplotlib_helpers import FormattedFigureText

        text_formatter = FormattedFigureText()
        # text_formatter.bottom_margin = 0.0 # No margin on the bottom
        # text_formatter.top_margin = 0.6 # doesn't change anything. Neither does subplot_adjust
        text_formatter.setup_margins(fig)

        # ## Header:
        # # Clear the normal text:
        # fig.suptitle('')
        # ax.set_title('')
        # # header_text_obj = flexitext(text_formatter.left_margin, 0.90, f'<size:22><weight:bold>{title_string}</></>\n<size:10>{subtitle_string}</>', va="bottom", xycoords="figure fraction")

        ## Footer only:
        if title_string is not None:
            fig.suptitle(title_string, fontsize='14', wrap=True)
        if (subtitle_string is not None) and (ax is not None):
            ax.set_title(subtitle_string, fontsize='10', wrap=True) # this doesn't appear to be visible, so what is it used for?

        footer_text_obj = text_formatter.add_flexitext_context_footer(active_context=active_context, override_left_margin_multipler=0.1, override_bottom_margin_multiplier=0.1) # flexitext((text_formatter.left_margin*0.1), (text_formatter.bottom_margin*0.25), text_formatter._build_footer_string(active_context=active_context), va="top", xycoords="figure fraction")

        # label_objects = {'header': header_text_obj, 'footer': footer_text_obj, 'formatter': text_formatter}
    return footer_text_obj


def matplotlib_configuration_update(is_interactive:bool, backend:Optional[str]=None):
    """Non-Context manager version for configuring Matplotlib interactivity, backend, and toolbar.
    
    The context-manager version notabily doesn't work for making the figures visible, I think because when it leaves the context handler the variables assigned within go away and thus the references to the Figures are lost.
    
    # Example usage:

        from neuropy.utils.matplotlib_helpers import matplotlib_configuration_update
        
        _restore_previous_matplotlib_settings_callback = matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')
        _restore_previous_matplotlib_settings_callback = matplotlib_configuration_update(is_interactive=False, backend='AGG')
            
            
            
        with matplotlib_configuration_update(is_interactive=False, backend='AGG'):
            # Perform non-interactive Matplotlib operations with 'AGG' backend
            plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title('Non-interactive Mode with AGG Backend')
            plt.savefig('plot.png')  # Save the plot to a file (non-interactive mode)

        with matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg'):
            # Perform interactive Matplotlib operations with 'Qt5Agg' backend
            plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title('Interactive Mode with Qt5Agg Backend')
            plt.show()  # Display the plot interactively (interactive mode)
    """
    if backend is None:
        if is_interactive:
            backend='Qt5Agg'
        else:
            backend='AGG'
    # Backup the current rcParams
    prev_rcParams = matplotlib.rcParams.copy()
    # Backup the current backend
    prev_backend = matplotlib.get_backend()
    # Backup the current interactive mode
    prev_interactivity = plt.isinteractive()
    # Backup the current plt.show command:
    prev_plt_show_command = getattr(plt, "show")

    def _restore_previous_settings_callback():
        """ restore previous settings when done.
        
        Captures: 
            prev_backend, prev_interactivity, prev_rcParams, prev_plt_show_command
        """
        # Restore the previous backend
        matplotlib.use(prev_backend, force=True)
        plt.switch_backend(prev_backend)
        plt.interactive(prev_interactivity)     # Restore the previous interactive mode
        # Restore the previous rcParams
        matplotlib.rcParams.clear()
        matplotlib.rcParams.update(prev_rcParams)
        setattr(plt, "show", prev_plt_show_command) # restore plt.show()

    try:
        # Configure toolbar based on interactivity mode
        if is_interactive:
            matplotlib.rcParams['toolbar'] = 'toolbar2'
        else:
            matplotlib.rcParams['toolbar'] = 'None'

        # Switch to the desired backend
        matplotlib.use(backend, force=True)

        # Initialize the new backend (if needed)
        plt.switch_backend(backend)

        # Switch to the desired interactivity mode
        plt.interactive(is_interactive)

        if not is_interactive:
            # Non-blocking
            # setattr(plt, "show", lambda: None)
            setattr(plt, "show", lambda: print(f'plt.show() was overriden by a call to `matplotlib_configuration_update(...)`'))

    except BaseException as e:
        # Exception occurred while switching the backend
        print(f"An exception occurred: {str(e)}\n Trying to revert settings using `_restore_previous_settings_callback()`...`")
        _restore_previous_settings_callback()
        print(f'revert complete.')
        # You can choose to handle the exception here or re-raise it if needed
        # If you choose to re-raise, make sure to restore the previous backend before doing so
        raise


    return _restore_previous_settings_callback




# ==================================================================================================================== #
# Context Managers for Switching Interactivity and Backend                                                             #
# ==================================================================================================================== #



@contextlib.contextmanager
def matplotlib_backend(backend:str):
    """Context manager for switching Matplotlib backend and safely restoring it to its previous value when done.
        # Example usage:
        with matplotlib_backend('AGG'):
            # Perform non-interactive Matplotlib operations
            plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title('Non-interactive Mode')
            plt.savefig('plot.png')  # Save the plot to a file (non-interactive mode)

        with matplotlib_backend('Qt5Agg'):
            # Perform interactive Matplotlib operations
            plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title('Interactive Mode')
            plt.show()  # Display the plot interactively (interactive mode)
    """
    # Backup the current backend
    prev_backend = matplotlib.get_backend()
    try:
        # Switch to the desired backend
        matplotlib.use(backend, force=True)

        # Initialize the new backend (if needed)
        plt.switch_backend(backend)

        # Yield control back to the caller
        yield
        
    except BaseException as e:
        # Exception occurred while switching the backend
        print(f"An exception occurred: {str(e)}")
        # You can choose to handle the exception here or re-raise it if needed
        # If you choose to re-raise, make sure to restore the previous backend before doing so
        raise
    finally:
        # Restore the previous backend
        matplotlib.use(prev_backend, force=True)
        plt.switch_backend(prev_backend)


@contextlib.contextmanager
def matplotlib_interactivity(is_interactive:bool):
    """Context manager for switching Matplotlib interactivity mode and safely restoring it to its previous value when done.

    # Example usage:
    with matplotlib_interactivity(is_interactive=False):
        # Perform non-interactive Matplotlib operations
        plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Non-interactive Mode')
        plt.show()  # Display the plot (if desired)


    with matplotlib_interactivity(is_interactive=True):
        # Perform interactive Matplotlib operations
        plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Interactive Mode')
        plt.show()  # Display the plot immediately (if desired)
    """
    # Backup the current interactive mode
    prev_interactivity = plt.isinteractive()
    try:
        # Switch to the desired interactivity mode
        plt.interactive(is_interactive)

        # Yield control back to the caller
        yield
        
    except BaseException as e:
        # Exception occurred while switching the backend
        print(f"An exception occurred: {str(e)}")
        # You can choose to handle the exception here or re-raise it if needed
        # If you choose to re-raise, make sure to restore the previous backend before doing so
        raise
    finally:
        # Restore the previous interactive mode
        plt.interactive(prev_interactivity)


@contextlib.contextmanager
def disable_function_context(obj, fn_name: str):
    """ Disables a function within a context manager

    https://stackoverflow.com/questions/10388411/possible-to-globally-replace-a-function-with-a-context-manager-in-python

    Could be used for plt.show().
    ```python
    
    from neuropy.utils.matplotlib_helpers import disable_function_context
    import matplotlib.pyplot as plt
    with disable_function_context(plt, "show"):
        run_me(x)
    
    """
    temp = getattr(obj, fn_name)
    setattr(obj, fn_name, lambda: None)
    yield
    setattr(obj, fn_name, temp)





@contextlib.contextmanager
def matplotlib_configuration(is_interactive:bool, backend:Optional[str]=None):
    """Context manager for configuring Matplotlib interactivity, backend, and toolbar.
    # Example usage:

        from neuropy.utils.matplotlib_helpers import matplotlib_configuration
        with matplotlib_configuration(is_interactive=False, backend='AGG'):
            # Perform non-interactive Matplotlib operations with 'AGG' backend
            plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title('Non-interactive Mode with AGG Backend')
            plt.savefig('plot.png')  # Save the plot to a file (non-interactive mode)

        with matplotlib_configuration(is_interactive=True, backend='Qt5Agg'):
            # Perform interactive Matplotlib operations with 'Qt5Agg' backend
            plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title('Interactive Mode with Qt5Agg Backend')
            plt.show()  # Display the plot interactively (interactive mode)
    """
    if backend is None:
        if is_interactive:
            backend='Qt5Agg'
        else:
            backend='AGG'
    # Backup the current rcParams
    prev_rcParams = matplotlib.rcParams.copy()
    try:
        # Configure toolbar based on interactivity mode
        if is_interactive:
            matplotlib.rcParams['toolbar'] = 'toolbar2'
        else:
            matplotlib.rcParams['toolbar'] = 'None'

        # Enter the backend and interactivity context managers
        with contextlib.ExitStack() as stack:
            stack.enter_context(matplotlib_interactivity(is_interactive))
            stack.enter_context(matplotlib_backend(backend))
            ## Non-blocking in non-interactive mode:
            if not is_interactive:
                stack.enter_context(disable_function_context(plt, "show")) # Non-blocking

            yield
            
    except BaseException as e:
        # Exception occurred while switching the backend
        print(f"An exception occurred: {str(e)}")
        # You can choose to handle the exception here or re-raise it if needed
        # If you choose to re-raise, make sure to restore the previous backend before doing so
        raise
    
    finally:
        # Restore the previous rcParams
        matplotlib.rcParams.clear()
        matplotlib.rcParams.update(prev_rcParams)



@contextlib.contextmanager
def matplotlib_file_only():
    """Context manager for configuring Matplotlib to only render to file, using the 'AGG' backend, no interactivity, and no plt.show()
    # Example usage:
        from neuropy.utils.matplotlib_helpers import matplotlib_file_only
        with matplotlib_file_only():
            # Perform non-interactive Matplotlib operations with 'AGG' backend
            plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title('Non-interactive Mode with AGG Backend')
            plt.savefig('plot.png')  # Save the plot to a file (non-interactive mode)
    """
    # Enter the backend and interactivity context managers
    with contextlib.ExitStack() as stack:
        stack.enter_context(matplotlib_configuration(is_interactive=False, backend='AGG'))
        yield



def resize_window_to_inches(window, width_inches, height_inches, dpi=96):
    """ takes a matplotlib figure size (specified in inches) and the figure dpi to compute the matching pixel size. # If you render a matplotlib figure in a pyqt5 backend window, you can appropriately set the size of this window using this function.

    # Example usage:
        # Assuming you have a QMainWindow instance named 'main_window'
        size=(5,12)
        resize_window_to_inches(mw.window(), *size)

    """
    width_pixels = int(width_inches * dpi)
    height_pixels = int(height_inches * dpi)
    window.resize(width_pixels, height_pixels)


# # ==================================================================================================================== #
# # 2024-03-12 - Multi-color/multi-line labels                                                                           #
# # ==================================================================================================================== #

## FLEXITEXT-version
from flexitext import FlexiText, Style
from flexitext.textgrid import make_text_grid, make_grid
from flexitext.text import Text as StyledText

import matplotlib as mpl
import matplotlib.colors
from matplotlib import cm
from neuropy.utils.mathutil import bounded

def get_heatmap_cmap(cmap: Union[str, mpl.colors.Colormap]='viridis', bad_color='black', under_color='white', over_color='red') -> mpl.colors.Colormap:
    """ 
    from neuropy.utils.matplotlib_helpers import get_heatmap_cmap
     # cmap = get_heatmap_cmap(cmap='viridis', bad_color='black', under_color='white', over_color='red')
    cmap = get_heatmap_cmap(cmap='Oranges', bad_color='black', under_color='white', over_color='red')
    
    """
    # Get the colormap to use and set the bad color
    if isinstance(cmap, str):
        ## convert to real colormap
        cmap = mpl.colormaps.get_cmap(cmap)  # viridis is the default colormap for imshow

    cmap.set_bad(color=bad_color, alpha=0.95)
    cmap.set_under(color=under_color, alpha=0.0)
    cmap.set_over(color=over_color, alpha=1.0)
    # cmap = 'turbo'
    return cmap
    



import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def modify_colormap_alpha(cmap: Union[str, mpl.colors.Colormap], alpha: float) -> ListedColormap:
    """
    Create a copy of the given colormap with a specified alpha.

    Args:
        cmap_name (str): Name of the original colormap.
        alpha (float): Desired alpha value (0.0 to 1.0).

    Returns:
        ListedColormap: A colormap with the specified alpha.
    
    Usage:

        from neuropy.utils.matplotlib_helpers import modify_colormap_alpha
        # Example usage
        custom_cmap = modify_colormap_alpha('viridis', 0.5)
        # Define a colormap
        cmap = plt.get_cmap('tab10')
        custom_cmap = modify_colormap_alpha(cmap=cmap, alpha=subsequence_line_color_alpha)
        num_colors: int = custom_cmap.N
        
        # Visualizing the custom colormap
        plt.imshow([np.linspace(0, 1, 100)], aspect='auto', cmap=custom_cmap)
        plt.colorbar()
        plt.show()

    """
    # Get the colormap to use and set the bad color
    if isinstance(cmap, str):
        ## convert to real colormap
        cmap = mpl.colormaps.get_cmap(cmap)  # viridis is the default colormap for imshow
            
    colors = cmap(np.arange(cmap.N))
    colors[:, 3] = alpha  # Modify alpha channel
    return ListedColormap(colors)


            

    
    
@define(slots=False)
class ValueFormatter:
    """ builds text formatting (for example larger values being rendered larger or more red) 

    Usage:
        from neuropy.utils.matplotlib_helpers import ValueFormatter
        a_val_formatter = ValueFormatter()
        a_val_formatter(0.934)
    """
    NONE_fallback_color: str = field(default="#000000") # black
    nan_fallback_color: str = field(default="#000000") # black
    out_of_range_fallback_color: str = field(default="#00FF00") # lime green

    # select a divergent colormap
    cmap: Union[str, mpl.colors.Colormap] = field(factory=(lambda *args, **kwargs: cm.coolwarm))
    norm: Optional[mpl.colors.Normalize] = field(factory=(lambda *args, **kwargs: mpl.colors.Normalize(vmin=-1, vmax=1)))
    
    coloring_function: Callable = field(default=None)


    # cmap = matplotlib.cm.get_cmap('Spectral')
    # cmap = cm.coolwarm
    # norm = matplotlib.colors.Normalize(vmin=10.0, vmax=20.0)

    def __attrs_post_init__(self):
        if self.cmap is None:
            self.cmap = cm.coolwarm
        if isinstance(self.cmap, str):
            ## convert to real colormap
            self.cmap = mpl.colormaps.get_cmap(self.cmap)  # viridis is the default colormap for imshow
            
            
        if self.norm is None:
            self.norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        if self.coloring_function is None:
            self.coloring_function = self.matplotlib_colormap_value_to_color_fn
        self.cmap.set_bad(self.NONE_fallback_color)
        self.cmap.set_under(self.out_of_range_fallback_color)
        self.cmap.set_over(self.out_of_range_fallback_color)


    def value_to_color(self, value, debug_print=True) -> str:
        """ Maps a value between -1.0 and 1.0 to an RGB color code. Returns a hex-formatted color string
        """
        assert self.coloring_function is not None
        return self.coloring_function(value, debug_print=debug_print)
    
            
    def value_to_format_dict(self, value, debug_print=False) -> Dict[str, Any]:
        """ Returns a formatting dict for rendering the value text suitable for use with flexitext_value_textprops

        Returns a formatting dict for rendering the value text suitable for use with flexitext_value_textprops

        """
        return {
            # 'color': self.value_to_color(value=value, debug_print=debug_print),
            'color': self.coloring_function(value=value, debug_print=debug_print),
             
        }
    
    # ==================================================================================================================== #
    # Specific coloring functions to use for self.coloring_function                                                        #
    # ==================================================================================================================== #
    def matplotlib_colormap_value_to_color_fn(self, value, debug_print=True) -> str:
        """ uses self.cmap and self.norm to format the value. """
        if value is None:
            return self.NONE_fallback_color
        elif np.isnan(value):
            return self.nan_fallback_color
        else:
            norm_value = self.norm(value)
            color = self.cmap(norm_value)
            if debug_print:
                print(f'value: {value}')
                print(f'norm_value: {norm_value}')
                print(f'color: {color}')

            return color

    # @function_attributes(short_name=None, tags=['cmap', 'diverging-cmap'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-25 09:52', related_items=[])
    def blue_grey_red_custom_value_to_color_fn(self, value, debug_print=True) -> str:
        """
        Maps a value between -1.0 and 1.0 to an RGB color code.
        -1.0 maps to bright blue, 0.0 maps to dark gray, and 1.0 maps to bright red.

        Returns a hex-formatted color string
        Does not use the matplotlib colormap properties.
        """
        import colorsys

        if value is None:
            return self.NONE_fallback_color
        elif np.isnan(value):
            return self.nan_fallback_color
        else:
            # valid color
            #TODO 2024-03-13 09:44: - [ ] Range bounds between 0-1 implicitly
            value = bounded(value, vmin=-1.0, vmax=1.0)
            magnitude_value: float = np.abs(value)
            # norm_value: float = map_to_fixed_range(magnitude_value, x_min=0.0, x_max=1.0)
            saturation_component = magnitude_value
            # saturation_component = norm_value

            if value <= 0:
                # Map values from -1.0 to 0.0 to shades of blue
                # norm = (value + 1) / 2  # Normalize to [0, 1] range
                rgb = colorsys.hsv_to_rgb(0.67, saturation_component, magnitude_value)  # Blue to dark gray
            else:
                # Map values from 0.0 to 1.0 to shades of red
                # norm = value  # No need to normalize
                rgb = colorsys.hsv_to_rgb(0.0, saturation_component, magnitude_value)  # Dark gray to red

            if debug_print:
                print(f'value: {value}')
                # print(f'norm_value: {norm_value}')
                print(f'magnitude_value: {magnitude_value}')
                print(f'saturation_component: {saturation_component}')
                print(f'rgb: {rgb}')

            # assert ((rgb[0] <= 1.0) and (rgb[0] >= 0.0))
            assert (np.all((np.array(rgb) <= 1.0)) and np.all((np.array(rgb) >= 0.0))), f"rgb: {rgb}, value: {value}"
            
            return '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)) # ValueError: cannot convert float NaN to integer
    


def find_first_available_matplotlib_font_name(desired_fonts_list):
    """ returns the first font that is available to matplotlib
    
        from neuropy.utils.matplotlib_helpers import find_first_available_matplotlib_font_name

        found_font_name = find_first_available_matplotlib_font_name(desired_fonts_list=['Source Sans Pro', 'Source Sans 3', 'DejaVu Sans Mono'])

    """
    import matplotlib.font_manager as fm

    known_font_names = sorted(fm.get_font_names())
    _found_font_name = None
    for a_font in desired_fonts_list:
        if (_found_font_name is None) and (a_font in known_font_names):
            ## found font
            _found_font_name = a_font
            return _found_font_name

    return _found_font_name






# @function_attributes(short_name=None, tags=['flexitext', 'matplotlib'], input_requires=[], output_provides=[], uses=['flexitext'], used_by=[], creation_date='2024-03-13 10:44', related_items=['AnchoredCustomText'])
def parse_and_format_unformatted_values_text(unformatted_text_block: str, key_value_split: str = ":", desired_label_value_sep: str = ": ", a_val_formatter: Optional[ValueFormatter]=None) -> Tuple[List[StyledText], ValueFormatter]:
    """ takes a potentially multi-line string containing keys and values like:
        unformatted_text_block: str = "wcorr: -0.754\n$P_i$: 0.052\npearsonr: -0.76"
    to produce a list of flexitext.Text objects that contain styled text that can be rendered.

    
    desired_label_value_sep: str - the desired label/value separator to be rendered in the final string:

    
    Usage:
        ## FLEXITEXT-version
        from flexitext import FlexiText, Style
        from flexitext.textgrid import make_text_grid, make_grid
        from flexitext.text import Text as StyledText

        unformatted_text_block: str = "wcorr: -0.754\n$P_i$: 0.052\npearsonr: -0.76"
        texts: List[StyledText] = parse_and_format_unformatted_values_text(test_test)
        text_grid: VPacker = make_text_grid(texts, ha="right")
        text_grid


    """
    if a_val_formatter is None:
        a_val_formatter = ValueFormatter()

    # splits into new lines first, then splits into (label, value) pairs on the `key_value_split`
    split_label_value_tuples = [[part.strip() for part in line.split(key_value_split)] for line in unformatted_text_block.splitlines()] # [['wcorr', '-0.754'], ['$P_i$', '0.052'], ['pearsonr', '-0.76']]

    found_font_name = find_first_available_matplotlib_font_name(desired_fonts_list=['Source Sans Pro', 'Source Sans 3', 'DejaVu Sans Mono'])    

    texts: List[StyledText] = []
    for (label, value) in split_label_value_tuples:
        flexitext_label_text_props = dict(color="black", name=found_font_name, size=9)
        float_val = float(value)
        # formatted_val_color: str = a_val_formatter.value_to_color(float_val, debug_print=False)
        flexitext_value_textprops = dict(color='grey', weight="bold", name=found_font_name, size=10) | a_val_formatter.value_to_format_dict(float_val, debug_print=False) # merge the formatting dict, using the value returned from the formatter preferentially
        
        label_formatted_text: StyledText = Style(**flexitext_label_text_props)(label + desired_label_value_sep)
        value_formatted_text: StyledText = Style(**flexitext_value_textprops)(value + "\n")
        texts.append(label_formatted_text)
        texts.append(value_formatted_text)

    return texts, a_val_formatter


# @metadata_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=['flexitext', 'parse_and_format_unformatted_values_text'], used_by=[], creation_date='2024-03-13 10:45', related_items=[])
class AnchoredCustomText(AnchoredOffsetbox):
    """
    AnchoredOffsetbox with Text.

    
    Usage:
        from typing import Tuple
        import matplotlib.pyplot as plt
        from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
        from neuropy.utils.matplotlib_helpers import AnchoredCustomText, build_formatted_label_values_stack, build_formatted_label_values_stack, value_to_color
                                
        # Create a figure and axis
        fig, ax = plt.subplots()
        formated_text_list = [("wcorr: ", -0.754),
                                ("$P_i$: ", 0.052), 
                                ("pearsonr: ", -0.76),
                            ]

        text_kwargs = _helper_build_text_kwargs_flat_top(a_curr_ax=ax)

        anchored_custom_text = AnchoredCustomText(formated_text_list=formated_text_list, pad=0., frameon=False,**text_kwargs, borderpad=0.)
        # anchored_box = AnchoredOffsetbox(child=stack_box, pad=0., frameon=False,**text_kwargs, borderpad=0.)

        # Add the offset box to the axes
        ax.add_artist(anchored_custom_text)

        # Display the plot
        plt.show()
            
            
    """

    def __init__(self, unformatted_text_block: str, custom_value_formatter: Optional[ValueFormatter]=None, **kwargs):
        """
        Parameters
        ----------
        **kwargs
            All other parameters are passed to `AnchoredOffsetbox`.
        """
        text_area_props = kwargs.pop('prop', {})

        # self.txtAreaItems = []
        # self.stack_box = build_formatted_label_values_stack(formated_text_list)
        self._unformatted_text_block = unformatted_text_block
        self._custom_value_formatter = custom_value_formatter
        self._texts, self._custom_value_formatter = parse_and_format_unformatted_values_text(self._unformatted_text_block, a_val_formatter=custom_value_formatter)
        self.stack_box: VPacker = make_text_grid(self._texts, ha="right")

        # first_stack_area = self.stack_box.get_children()[0][0]
        first_stack_area = self.stack_box.findobj(match=TextArea, include_self=False)[0]
        # self.txt = TextArea(s, textprops=prop)
        fp = first_stack_area._text.get_fontproperties() # passed to AnchoredOffsetbox, This is only used as a reference for paddings.
        super().__init__(child=self.stack_box, prop=fp, **kwargs)



    @property
    def text_areas(self) -> List[TextArea]:
        """The text_areas property."""
        return self.findobj(match=TextArea, include_self=False)

    @property
    def text_objs(self) -> List[Text]:
        """The matplotlib.Text objects. """
        return [a_text_area._text for a_text_area in self.text_areas]

    def update_text_alpha(self, value: float):
        for a_text_area in self.text_areas:
            a_text_area._text.set_alpha(value)
            
    def update_text(self, unformatted_text_block: str) -> bool:
        """ not yet working """
        did_text_change = (unformatted_text_block != self._unformatted_text_block)
        raise NotImplementedError("just remove from parent and build a new one. `custom_anchored_text.remove()`")
        if not did_text_change:
            print(f'text did not change!')
            return False
        ## otherwise text did change
        self._unformatted_text_block = unformatted_text_block
        self._texts, self._custom_value_formatter = parse_and_format_unformatted_values_text(self._unformatted_text_block, a_val_formatter=self._custom_value_formatter)

        ax = deepcopy(self.axes)

        if self.stack_box is not None:
            # self.stack_box.remove() ## remove the old one -- can't remove artists
            self.remove()
            self.stack_box = None
        
        ## Create the new one:
        self.stack_box: VPacker = make_text_grid(self._texts, ha="right")
        ax.add_artist(self)
        return True



# @function_attributes(short_name=None, tags=['colormap', 'image', 'colorbar'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-09-18 19:06', related_items=[])
def save_colormap_image(filepath: str, colormap: str = 'viridis', orientation: str = 'horizontal'):
    """
    Saves a colorbar image of the specified colormap to the given file path.

    Args:
        filepath (str): The path where the image will be saved.
        colormap (str): The name of the colormap (default is 'viridis').
        orientation (str): Orientation of the colorbar, either 'horizontal' or 'vertical' (default is 'horizontal').

    Usage:
        from neuropy.utils.matplotlib_helpers import save_colormap_image
        
        save_colormap_image('path/to/save/colorbar.png', 'plasma', 'vertical')
    """
    fig, ax = plt.subplots(figsize=(6, 1) if orientation == 'horizontal' else (1, 6))
    fig.subplots_adjust(bottom=0.5 if orientation == 'horizontal' else 0.05, left=0.05 if orientation == 'vertical' else 0.5)

    cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(vmin=0, vmax=1)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation=orientation)

    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)



            
class TabbedMatplotlibFigures:
    """ 
    Example:
        from neuropy.utils.matplotlib_helpers import TabbedMatplotlibFigures
        
        def _perform_plot(x_data, y_data, extant_fig=None, extant_axes=None, **kwargs):
            if extant_fig is not None:
                fig = extant_fig # use the existing passed figure
                # fig.set_size_inches([23, 9.7])
            else:
                if extant_axes is None:
                    fig = plt.figure(figsize=(23, 9.7))
                else:
                    fig = extant_axes.get_figure()
                    
            if extant_axes is not None:
                # user-supplied extant axes [axs0, axs1]
                assert isinstance(extant_axes, dict), f"extant_axes should be a dict but is instead of type: {type(extant_axes)}\n\textant_axes: {extant_axes}"
                assert len(extant_axes) == 1
                ax_scatter = extant_axes["ax_scatter"]
            else:
                # Need axes:
                # Layout Subplots in Figure:
                extant_axes = fig.subplot_mosaic(
                    [   
                        ["ax_scatter"],
                    ],
                )
                ax_scatter = extant_axes["ax_scatter"]
                # gs = fig.add_gridspec(1, 8)
                # gs.update(wspace=0, hspace=0.05) # set the spacing between axes. # `wspace=0`` is responsible for sticking the pf and the activity axes together with no spacing
                # ax_activity_v_time = fig.add_subplot(gs[0, :-1]) # all except the last element are the trajectory over time
                # ax_pf_tuning_curve = fig.add_subplot(gs[0, -1], sharey=ax_activity_v_time) # The last element is the tuning curve
                # ax_pf_tuning_curve.set_title('Normalized Placefield', fontsize='14')
                ax_scatter.set_xticklabels([])
                ax_scatter.set_yticklabels([])
            # end else
            
            ## BEGIN FUNCTION BODY
            ax_scatter.plot(x_data, y_data)
            
            return fig, extant_axes


        ## INPUTS: arrays, plot_data_dict

        plot_subplot_mosaic_dict = {f"arr[{i}]":dict(sharex=True, sharey=True,) for i, arr in enumerate(arrays)}
        ui, figures_dict, axs_dict = TabbedMatplotlibFigures._build_tabbed_multi_figure(plot_subplot_mosaic_dict)

        for a_name, ax in axs_dict.items():
            plot_arr = plot_data_dict[a_name]
            fig = figures_dict[a_name]
            fig, extant_axes = _perform_plot(x_data=np.arange(len(plot_arr)), y_data=plot_arr, extant_fig=fig, extant_axes=ax)
        ui.show()


    """
    @classmethod
    def build_tabbed_multi_figure(cls, plot_subplot_mosaic_dict, obj_class = None):
        """ 
        plot_subplot_mosaic_dict = {f"arr[{i}]":dict(sharex=True, sharey=True,) for i, arr in enumerate(arrays)}
        ui, plots_dict, axs_dict = TabbedMatplotlibFigures.build_tabbed_multi_figure(plot_subplot_mosaic_dict)
        ui.show()
        
                
        """
        from mpl_multitab import MplMultiTab, MplMultiTab2D ## Main imports
        from pyphoplacecellanalysis.Pho2D.matplotlib.CustomMatplotlibTabbedWidget import CustomMplMultiTab
        
        if obj_class is None:
            obj_class = CustomMplMultiTab

        # ui = MplMultiTab()
        # ui = CustomMplMultiTab()
        ui = obj_class()
        figures_dict = {}
        axs_dict = {}
        # for plot_name, plot_arr in plot_data_dict.items():
        for plot_name, plot_subplot_mosaic_kwargs in plot_subplot_mosaic_dict.items():
            fig = ui.add_tab(plot_name)
            figures_dict[plot_name] = fig
            if 'mosaic' not in plot_subplot_mosaic_kwargs:
                ## add the required 'mosaic' parameter if it's missing, this specifies the layout
                plot_subplot_mosaic_kwargs['mosaic'] = [   
                    ["ax_scatter"],
                ]
            extant_axes = fig.subplot_mosaic(**plot_subplot_mosaic_kwargs)
            # extant_axes = fig.figure.subplot_mosaic(**plot_subplot_mosaic_kwargs)
            


            # fig, extant_axes = _perform_plot(x_data=np.arange(len(plot_arr)), y_data=plot_arr, extant_fig=fig)
            # extant_axes = fig.subplot_mosaic(mosaic=
            #     [   
            #         ["ax_scatter"],
            #     ],
            # **plot_subplot_mosaic_kwargs,
            # )
            
            axs_dict[plot_name] = extant_axes

        return ui, figures_dict, axs_dict
        # return MatplotlibRenderPlots(name=f'TabbedMatplotlibFigures.build_tabbed_multi_figure', figures=figures_dict, axes=axs_dict, ui=ui)

    