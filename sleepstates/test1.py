import numpy as np
from unidip import UniDip, dip

# create bi-modal distribution
dat = np.concatenate([np.random.randn(200) - 3, np.random.randn(200) + 3])

# sort data so returned indices are meaningful
dat = np.msort(dat)

# get start and stop indices of peaks
intervals = UniDip(dat).run()
test = dip.diptst(dat)
