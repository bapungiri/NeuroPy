import matplotlib.pyplot as plt
import numpy as np

from neuropy.utils.mixins.dict_representable import overriding_dict_with # required for plot_placefield_occupancy
from neuropy.utils.mixins.unwrap_placefield_computation_parameters import unwrap_placefield_computation_parameters

def plot_all_placefields(active_placefields1D, active_placefields2D, active_config, variant_identifier_label=None, should_save_to_disk=True, **kwargs):
    """ Main function to plot all aspects of 1D and 2D placefields
    active_placefields1D: (Pf1D)
    active_placefields2D: (Pf2D)
    active_config:
    Usage:
        ax_pf_1D, occupancy_fig, active_pf_2D_figures = plot_all_placefields(active_epoch_placefields1D, active_epoch_placefields2D, active_config)
    """
    active_epoch_name = active_config.active_epochs.name
    active_pf_computation_params = unwrap_placefield_computation_parameters(active_config.computation_config)
    common_parent_foldername = active_pf_computation_params.str_for_filename(True)
    
    ## Linearized (1D) Position Placefields:
    if active_placefields1D is not None:
        ax_pf_1D = active_placefields1D.plot_ratemaps_1D(sortby='id') # by passing a string ('id') the plot_ratemaps_1D function chooses to use the normal range as the sort index (instead of sorting by the default)
        active_pf_1D_identifier_string = '1D Placefields - {}'.format(active_epoch_name)
        if variant_identifier_label is not None:
            active_pf_1D_identifier_string = ' - '.join([active_pf_1D_identifier_string, variant_identifier_label])

        
        title_string = ' '.join([active_pf_1D_identifier_string])
        subtitle_string = ' '.join([f'{active_placefields1D.config.str_for_display(False)}'])
        
        plt.gcf().suptitle(title_string, fontsize='14')
        plt.gca().set_title(subtitle_string, fontsize='10')
        
        if should_save_to_disk:
            active_pf_1D_filename_prefix_string = f'Placefield1D-{active_epoch_name}'
            if variant_identifier_label is not None:
                active_pf_1D_filename_prefix_string = '-'.join([active_pf_1D_filename_prefix_string, variant_identifier_label])
            active_pf_1D_filename_prefix_string = f'{active_pf_1D_filename_prefix_string}-' # it always ends with a '-' character
            common_basename = active_placefields1D.str_for_filename(prefix_string=active_pf_1D_filename_prefix_string)
            active_pf_1D_output_filepath = active_config.plotting_config.get_figure_save_path(common_parent_foldername, common_basename).with_suffix('.png')
            print('Saving 1D Placefield image out to "{}"...'.format(active_pf_1D_output_filepath), end='')
            plt.savefig(active_pf_1D_output_filepath)
            print('\t done.')
            
    else:
        print('plot_all_placefields(...): active_epoch_placefields1D does not exist. Skipping it.')
        ax_pf_1D = None

    ## 2D Position Placemaps:
    if active_placefields2D is not None:
        active_2D_occupancy_variant_identifier_list = [active_epoch_name]
        if variant_identifier_label is not None:
            active_2D_occupancy_variant_identifier_list.append(variant_identifier_label)
        occupancy_fig, occupancy_ax = active_placefields2D.plot_occupancy(identifier_details_list=active_2D_occupancy_variant_identifier_list)
        
        # Save ocupancy figure out to disk:
        if should_save_to_disk:
            active_2D_occupancy_filename_prefix_string = f'Occupancy-{active_epoch_name}'
            if variant_identifier_label is not None:
                active_2D_occupancy_filename_prefix_string = '-'.join([active_2D_occupancy_filename_prefix_string, variant_identifier_label])
            active_2D_occupancy_filename_prefix_string = f'{active_2D_occupancy_filename_prefix_string}-' # it always ends with a '-' character
            common_basename = active_placefields2D.str_for_filename(prefix_string=active_2D_occupancy_filename_prefix_string)
            active_pf_occupancy_2D_output_filepath = active_config.plotting_config.get_figure_save_path(common_parent_foldername, common_basename).with_suffix('.png')
            print('Saving 2D Placefield image out to "{}"...'.format(active_pf_occupancy_2D_output_filepath), end='')
            occupancy_fig.savefig(active_pf_occupancy_2D_output_filepath)
            print('\t done.')
            
        ## 2D Tuning Curves Figure:
        active_pf_2D_identifier_string = '2D Placefields - {}'.format(active_epoch_name)
        if variant_identifier_label is not None:
            active_pf_2D_identifier_string = ' - '.join([active_pf_2D_identifier_string, variant_identifier_label])
        title_string = ' '.join([active_pf_2D_identifier_string])
        subtitle_string = ' '.join([f'{active_placefields2D.config.str_for_display(True)}'])
        
        extended_overlay_points_datasource_dicts = kwargs.get('extended_overlay_points_datasource_dicts', None)
        active_pf_2D_figures, active_pf_2D_gs, active_pf_2D_graphics_obj_dicts = active_placefields2D.plot_ratemaps_2D(subplots=(80, 3), resolution_multiplier=2.5, extended_overlay_points_datasource_dicts=extended_overlay_points_datasource_dicts)

        if should_save_to_disk:
            active_pf_2D_filename_prefix_string = f'Placefields-{active_epoch_name}'
            if variant_identifier_label is not None:
                active_pf_2D_filename_prefix_string = '-'.join([active_pf_2D_filename_prefix_string, variant_identifier_label])
            active_pf_2D_filename_prefix_string = f'{active_pf_2D_filename_prefix_string}-' # it always ends with a '-' character
            common_basename = active_placefields2D.str_for_filename(prefix_string=active_pf_2D_filename_prefix_string)
            active_pf_2D_output_filepath = active_config.plotting_config.get_figure_save_path(common_parent_foldername, common_basename).with_suffix('.png')
            print('Saving 2D Placefield image out to "{}"...'.format(active_pf_2D_output_filepath), end='')
            for aFig in active_pf_2D_figures:
                aFig.savefig(active_pf_2D_output_filepath)
            print('\t done.')
    else:
        print('plot_all_placefields(...): active_epoch_placefields2D does not exist. Skipping it.')
        occupancy_fig = None
        active_pf_2D_figures = None
    
    return ax_pf_1D, occupancy_fig, active_pf_2D_figures, active_pf_2D_gs



def plot_placefield_occupancy(active_epoch_placefields, fig=None, ax=None, plot_pos_bin_axes: bool=False, **kwargs):
    """ plots the placefield occupancy in a matplotlib figure. 
    Works for both 1D and 2D.
    
    from neuropy.plotting.placemaps import plot_placefield_occupancy
    
    """
    def _subfn_plot_occupancy_1D(active_epoch_placefields1D, max_normalized, drop_below_threshold=None, fig=None, ax=None):
        """ Draws an occupancy curve showing the relative proprotion of the recorded positions that occured in a given position bin. """
        should_fill = False
        
        if fig is None:
            occupancy_fig = plt.figure()
        else:
            occupancy_fig = fig
        
        if ax is None:
            occupancy_ax = occupancy_fig.gca()
        else:
            occupancy_ax = ax

        only_visited_occupancy = active_epoch_placefields1D.occupancy.copy()
        # print('only_visited_occupancy: {}'.format(only_visited_occupancy))
        if drop_below_threshold is not None:
            only_visited_occupancy[np.where(only_visited_occupancy < drop_below_threshold)] = np.nan
        
        if max_normalized:
            only_visited_occupancy = only_visited_occupancy / np.nanmax(only_visited_occupancy)
            
        if not plot_pos_bin_axes:
            x_centers = active_epoch_placefields1D.ratemap.xbin_centers
            x_edges = active_epoch_placefields1D.ratemap.xbin
        else:
            from neuropy.utils.mixins.binning_helpers import get_bin_edges
            
            n_xbins: int = len(active_epoch_placefields1D.ratemap.xbin_centers)
            xbin_size: float = 1.0
            half_xbin_size: float = xbin_size/2.0
            
            x_centers = np.arange(n_xbins) + half_xbin_size            
            x_edges = get_bin_edges(x_centers)

        if should_fill:
            occupancy_ax.plot(x_centers, only_visited_occupancy)
            occupancy_ax.scatter(x_centers, only_visited_occupancy, color='r')
        
        occupancy_ax.stairs(only_visited_occupancy, x_edges, fill=False, label='1D Placefield Occupancy', hatch='//') # can also use: , orientation='horizontal'
        occupancy_ax.set_ylim([0, np.nanmax(only_visited_occupancy)])
        
        # specify bin_size, etc
        occupancy_ax.set_title('Occupancy 1D')
        return occupancy_fig, occupancy_ax

    def _subfn_plot_occupancy_custom(occupancy, xbin, ybin, max_normalized: bool, drop_below_threshold: float=None, fig=None, ax=None):
        """ Plots a 2D Heatmap of the animal's occupancy (the amount of time the animal spent in each posiution bin)

        Args:
            occupancy ([type]): [description]
            xbin ([type]): [description]
            ybin ([type]): [description]
            max_normalized (bool): [description]
            drop_below_threshold (float, optional): [description]. Defaults to None.
            fig ([type], optional): [description]. Defaults to None.
            ax ([type], optional): [description]. Defaults to None.
        """
        if fig is None:
            occupancy_fig = plt.figure()
        else:
            print(f'using specified fig')
            occupancy_fig = fig
        
        if ax is None:
            occupancy_ax = occupancy_fig.gca()
        else:
            print(f'using specified ax')
            occupancy_ax = ax
            
        only_visited_occupancy = occupancy.copy()
        # print('only_visited_occupancy: {}'.format(only_visited_occupancy))
        if drop_below_threshold is not None:
            only_visited_occupancy[np.where(only_visited_occupancy < drop_below_threshold)] = np.nan
        if max_normalized:
            only_visited_occupancy = only_visited_occupancy / np.nanmax(only_visited_occupancy)
        im = occupancy_ax.pcolorfast(xbin, ybin, np.rot90(np.fliplr(only_visited_occupancy)), cmap="jet", vmin=0.0)  # rot90(flipud... is necessary to match plotRaw configuration.
        occupancy_ax.set_title('Custom Occupancy')
        occupancy_ax.set_aspect('equal', adjustable=None)
        occupancy_cbar = occupancy_fig.colorbar(im, ax=occupancy_ax, location='right')
        occupancy_cbar.minorticks_on()
        return occupancy_fig, occupancy_ax


    # BEGIN MAIN FUNCTION ________________________________________________________________________________________________ #
    if active_epoch_placefields.ndim > 1:
        return _subfn_plot_occupancy_custom(active_epoch_placefields.occupancy, active_epoch_placefields.ratemap.xbin_centers, active_epoch_placefields.ratemap.ybin_centers, fig=fig, ax=ax, **overriding_dict_with(lhs_dict={'max_normalized':True, 'drop_below_threshold':1E-16}, **kwargs))
    else:
        return _subfn_plot_occupancy_1D(active_epoch_placefields, fig=fig, ax=ax, **overriding_dict_with(lhs_dict={'max_normalized':True, 'drop_below_threshold':1E-16}, **kwargs)) # handle 1D case
                                 
