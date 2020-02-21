import numpy as np
import matplotlib.pyplot as plt
from parsePath import name2path


class pf(name2path):

    nChans = 16
    sRate = 30000
    binSize = 0.250  # in seconds
    timeWindow = 3600  # in seconds

    def pf1d(self):
        spkAll = np.load(
            self.basePath + self.subname + "_spikes.npy", allow_pickle=True
        )
        position = np.load(self.filePrefix + "_position.npy", allow_pickle=True)

        xcoord = position.item().get("X")
        ycoord = position.item().get("Y")
        time = position.item().get("time")

        epochs = np.load(self.filePrefix + "_epochs.npy", allow_pickle=True)

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
            spd_ind = np.where(spk_spd > 5)
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
