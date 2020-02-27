import altair as alt
import numpy as np
import pandas as pd

a = pd.DataFrame(columns=["state", "time", "delta"])

for i in range(5):

    st_data = pd.DataFrame({"state": i, "time": [1, 2, 3], "delta": [5, 6, 7]})

    a = a.append(st_data)
