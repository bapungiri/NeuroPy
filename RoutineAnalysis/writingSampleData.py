basePath2 = ("/data/Clustering/SleepDeprivation/RatK/Day4/RatK__2019-08-16_04-42-36/experiment1/recording1/continuous/Rhythm_FPGA-100.0/"
             )


subname = os.path.basename(os.path.normpath(basePath))
fileName = basePath2 + 'continuous' + '.dat'
reqChan = 33
b1 = np.memmap(fileName, dtype='int16', mode='r', shape=(1, 30000*134*20))
ThetaExtract = b1[reqChan::nChans]
data = b1

# np.save(basePath+subname+'_BestThetaChan.npy', ThetaExtract)

file = open(basePath+subname+'_Example.dat', "wb")
file.write(data)
file.close()
