from neuropy.core.datawriter import DataWriter
from pathlib import Path
import neuropy.io.openephysio as oeio
from neuropy.core.session import ProcessData
from neuropy.io.binarysignalio import BinarysignalIO
from neuropy.io.neuroscopeio import NeuroscopeIO


class OESync(DataWriter):
    """Class to synchronize different systems with ephys data collected with OpenEphys. Most useful when multiple dat
    files have been concatenated into one, but also useful for individual recordings."""

    def __init__(
        self,
        basepath,
        metadata=None,
    ) -> None:
        super().__init__(metadata=metadata)

        self.basepath = Path(basepath)
        self.sess = ProcessData(basepath)
        self.oe_sync_df = oeio.create_sync_df(self.basepath)

    def check_external_ts_vs_ttl(self, external_ts, ttl_num, plot_check=True):
        """Checks external timestamps vs corresponding TTLs in OpenEphys to make sure there is a consistent lag between
        expected TTL time and actual TTL time.  Raise a flag/warning if that is not the case or if there is a dropped
        TTL somewhere
        :param external_ts: timezone aware array of timestamps (OR array of elapsed times from start of recording?),
        calculated by an external (un-synced) system
        :param ttl_num: int for OE TTL channel corresponding to all timestamps in external_ts
        :param plot_check: bool, plots lags between TTLs versus time and identifies dropped frames

        :return lags: np.ndarray of lags to correct external_ts to combined time
        :return good_bool: boolean np.ndarray of frames with corresponding TTLs in OpenEphys files
        """
        pass

    def sync_external_ts_to_oe(self, external_ts, ttl_num, plot_check):
        """Matches up external timestamps to TTL times in OE. See check_external_ts_vs_ttl
        :return: np.ndarray of external timestamps which correspond to OE TTLs in combined OE time
        :return: good_bool: boolean np.ndarray of frames with corresponding TTLs in OpenEphys files.
                            external_ts[good_bool] will return times of all frames recorded in OE."""
