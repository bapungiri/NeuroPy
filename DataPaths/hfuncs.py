from base64 import b64encode
from io import BytesIO

import bokeh.models as bmodels
import bokeh.plotting as bplot
import numpy as np
import pandas as pd
from matplotlib import cm
from neuropy.utils import signal_process
from PIL import Image
from scipy import stats
from skimage import img_as_ubyte
from neuropy import core


def wavelet_gamma_theta_phase(signal, theta_phase, binsize=9, frgamma=None, fs=1250):

    if frgamma is None:
        frgamma = np.arange(25, 150)

    # ----- wavelet power for gamma oscillations----------
    wavdec = signal_process.wavelet_decomp(signal, freqs=frgamma, sampfreq=fs)
    wav = wavdec.colgin2009()
    wav = stats.zscore(wav, axis=1)

    # ----segmenting gamma wavelet at theta phases ----------
    bin_angle = np.linspace(0, 360, int(360 / binsize) + 1)
    phase_centers = bin_angle[:-1] + np.diff(bin_angle).mean() / 2

    bin_ind = np.digitize(theta_phase, bin_angle)

    gamma_at_theta = pd.DataFrame()
    for i in np.unique(bin_ind):
        find_where = np.where(bin_ind == i)[0]
        gamma_at_theta[phase_centers[i - 1]] = np.mean(wav[:, find_where], axis=1)
    gamma_at_theta.insert(0, column="freq", value=frgamma)
    gamma_at_theta.set_index("freq", inplace=True)

    return gamma_at_theta


def to_png(arr):
    out = BytesIO()
    im = Image.fromarray(arr)
    im.save(out, format="png")
    return out.getvalue()


def b64_image_files(images, colormap="hot"):
    cmap = cm.get_cmap(colormap)
    urls = []
    for im in images:
        im_ = np.flipud((im - np.min(im)) / np.max(im))
        png = to_png(img_as_ubyte(cmap(im_)))
        url = "data:image/png;base64," + b64encode(png).decode("utf-8")
        urls.append(url)
    return urls


def plot_in_bokeh(
    x, y, img_arr, color_by=None, palette="jet", size=5, width=1200, height=800
):

    # info to show on images
    tooltips = """
        <div>
            <div>
                <img
                    src="@imgs" height="100" alt="@imgs" width="100"
                    style="float: left; margin: 0px 15px 15px 0px;image-rendering: pixelated"
                    border="2"
                ></img>
            </div>
        </div>
    """

    arr_images = b64_image_files(img_arr)

    if color_by is not None:
        cmap = cm.get_cmap(palette)
        color_by = color_by - np.min(color_by)
        color_by /= np.max(color_by)
        colors = cmap(color_by)
    else:
        colors = ["red"] * len(x)

    source = bplot.ColumnDataSource(
        data=dict(
            x=x,
            y=y,
            imgs=arr_images,
            colors=colors,
        )
    )

    p = bplot.figure(
        width=width,
        height=height,
        tooltips=tooltips,
        x_axis_label="Time",
        y_axis_label="Replay score",
        title="Replay score over time, Hover to see posterior",
    )
    p.circle("x", "y", size=size, source=source, color="colors")
    return p


def linearize_using_shapely(position: core.Position):
    from matplotlib.backend_bases import MouseButton
    from shapely.geometry import LineString, Point
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(position.x, position.y)

    global coord
    coord = []
    global flag
    flag = 0

    def on_click(event):
        if (event.button is MouseButton.LEFT) and (event.inaxes):
            global coord
            coord.append((event.xdata, event.ydata))
            x = event.xdata
            y = event.ydata

            ax.plot(x, y, "o", color="r")
            fig.canvas.draw()  # redraw the figure

        if event.button is MouseButton.RIGHT:
            flag = 1
            fig.disconnent()

    fig.canvas.mpl_connect("button_press_event", on_click)

    while flag != 0:
        line = LineString(coord)
        lin_pos = []
        for x, y in zip(position.x, position.y):
            lin_pos.append(line.project(Point(x, y)))

        return lin_pos


def whiten_signal(signal: core.Signal):
    from statsmodels.tsa.ar_model import AutoReg

    trace = np.pad(signal.traces[0], (2, 0), "constant", constant_values=(0,))
    model = AutoReg(trace, 2, old_names=False)
    res = model.fit().predict(0, len(trace) - 1)[2:]

    return core.Signal(
        traces=(signal.traces[0] - res).reshape(1, -1),
        sampling_rate=signal.sampling_rate,
        t_start=signal.t_start,
    )
