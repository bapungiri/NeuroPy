import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class pf:

    nChans = 16
    sRate = 30000
    binSize = 0.250  # in seconds
    timeWindow = 3600  # in seconds

    def __init__(self, obj, **kwargs):
        self._obj = obj
        # self.pf1d = pf1d(obj)
        self.pf2d = pf2d(obj)


class pf1d:
    def compute(self):
        spkAll = np.load(str(self.filePrefix) + "_spikes.npy", allow_pickle=True)
        position = np.load(str(self.filePrefix) + "_position.npy", allow_pickle=True)

        xcoord = self._obj.position.x
        ycoord = position.item().get("Y")
        time = position.item().get("time")

        epochs = np.load(str(self.filePrefix) + "_epochs.npy", allow_pickle=True)

        maze = epochs.item().get("MAZE")  # in seconds

        ind_maze = np.where((time > maze[0]) & (time < maze[1]))
        self.x = xcoord[ind_maze] / 10
        self.y = ycoord[ind_maze] / 10
        self.t = time[ind_maze]

        diff_posx = np.diff(self.x)
        diff_posy = np.diff(self.y)

        dt = self.t[1] - self.t[0]

        # location = np.sqrt((xcoord) ** 2 + (ycoord) ** 2)
        self.speed = np.abs(diff_posy) / dt

        spk_pfx, spk_pfy, spk_pft = [], [], []
        for cell in spkAll:

            spk_maze = cell[np.where((cell > maze[0]) & (cell < maze[1]))]
            spk_spd = np.interp(spk_maze, self.t[:-1], self.speed)
            spk_y = np.interp(spk_maze, self.t, self.y)
            spk_x = np.interp(spk_maze, self.t, self.x)

            # speed threshold
            spd_ind = np.where(spk_spd > 2)
            spk_spd = spk_spd[spd_ind]
            spk_x = spk_x[spd_ind]
            spk_y = spk_y[spd_ind]
            spk_t = spk_maze[spd_ind]
            spk_pfx.append(spk_x)
            spk_pfy.append(spk_y)
            spk_pft.append(spk_t)

        self.spkx = spk_pfx
        self.spky = spk_pfy
        self.spkt = spk_pft

        # spdt = timepos

        # spktime = spikes

        # xmesh = np.arange(min(xcoord), max(xcoord) + 1, 2)
        # ymesh = np.arange(min(ycoord), max(ycoord) + 1, 2)
        # xx, yy = np.meshgrid(xmesh, ymesh)
        # pf2, xe1, ye1 = np.histogram2d(xcoord, ycoord, bins=[xmesh, ymesh])

        # plt.clf()
        # nPyr = 10
        # for cell in range(10):
        #     spkt = spktime[cell]
        #     spd_spk = np.interp(spkt, spdt[:-1], speed)

        #     spkt = spkt[
        #         spd_spk > 50
        #     ]  # only selecting spikes where rat's speed is  > 5 cm/s

        #     spktx = np.interp(spkt, timepos, xcoord)
        #     spkty = np.interp(spkt, timepos, ycoord)

        #     spktx = spktx.reshape((len(spktx)))
        #     spkty = spkty.reshape((len(spkty)))

        #     pf, xe, ye = np.histogram2d(spktx, spkty, bins=[xmesh, ymesh])

        #     pft = pf2 * (1 / 30)

        #     eps = np.spacing(1)
        #     pfRate = pf / (pft + eps)

        #     pfRate_smooth = gaussian_filter(pfRate, sigma=3)

        #     nRows = np.ceil(np.sqrt(nPyr))
        #     nCols = np.ceil(np.sqrt(nPyr))

        #     #    plt.plot(posx_mz,posy_mz,'.')
        #     plt.subplot(nRows, nCols, cell + 1)
        #     plt.imshow(pfRate_smooth)


class pf2d:
    def __init__(self, obj, **kwargs):
        self._obj = obj

    def compute(self):
        spkAll = self._obj.spikes.times
        xcoord = self._obj.position.x
        ycoord = self._obj.position.y
        time = self._obj.position.t
        maze = self._obj.epochs.maze  # in seconds

        ind_maze = np.where((time > maze[0]) & (time < maze[1]))
        x = xcoord[ind_maze]
        y = ycoord[ind_maze]
        t = time[ind_maze]

        x_grid = np.arange(min(x), max(x), 70)
        y_grid = np.arange(min(y), max(y), 70)
        x_, y_ = np.meshgrid(x_grid, y_grid)

        diff_posx = np.diff(x)
        diff_posy = np.diff(y)

        speed = np.sqrt(diff_posx ** 2 + diff_posy ** 2)
        dt = t[1] - t[0]
        speed_thresh = np.where(speed / dt > 0)[0]

        x_thresh = x[speed_thresh]
        y_thresh = y[speed_thresh]
        t_thresh = t[speed_thresh]

        occupancy = np.histogram2d(x, y, bins=(x_grid, y_grid))[0]
        occupancy = occupancy + np.spacing(1)
        occupancy = occupancy / 120  # converting to seconds

        # spk_pfx, spk_pfy, spk_pft = [], [], []
        pf, spk_pos = [], []
        for cell in spkAll:

            spk_maze = cell[np.where((cell > maze[0]) & (cell < maze[1]))]
            spk_speed = np.interp(spk_maze, t[1:], speed)
            spk_y = np.interp(spk_maze, t, y)
            spk_x = np.interp(spk_maze, t, x)

            # speed threshold
            # spd_ind = np.where(spk_speed > 2)
            # spk_spd = spk_speed[spd_ind]
            # spk_x = spk_x[spd_ind]
            # spk_y = spk_y[spd_ind]
            # spk_t = spk_maze[spd_ind]

            spk_map = np.histogram2d(spk_x, spk_y, bins=(x_grid, y_grid))[0]
            pf.append(spk_map / occupancy)
            spk_pos.append([spk_x, spk_y])

            # spk_pfx.append(spk_x)
            # spk_pfy.append(spk_y)
            # spk_pft.append(spk_t)

        self.spk_pos = spk_pos
        self.maps = pf
        self.speed = speed
        self.x = x
        self.y = y
        # self.spkx = spk_pfx
        # self.spky = spk_pfy
        # self.spkt = spk_pft

    def plotMap(self):
        fig = plt.figure(1, figsize=(6, 10))
        gs = GridSpec(7, 6, figure=fig)
        fig.subplots_adjust(hspace=0.4)

        for cell, pfmap in enumerate(self.maps):
            ax1 = fig.add_subplot(gs[cell])
            ax1.imshow(pfmap, cmap="jet", interpolation="gaussian")

    def plotRaw(self):
        fig = plt.figure(2, figsize=(6, 10))
        gs = GridSpec(7, 6, figure=fig)
        # fig.subplots_adjust(hspace=0.4)

        for cell, (spk_x, spk_y) in enumerate(self.spk_pos):
            ax1 = fig.add_subplot(gs[cell])
            ax1.plot(self.x, self.y, color="#d3c5c5")
            ax1.plot(spk_x, spk_y, ".r", markersize=1.2)
            ax1.axis("off")
