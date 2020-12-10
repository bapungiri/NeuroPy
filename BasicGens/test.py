import xml.etree.ElementTree as ET
import numpy as np

path = (
    "/data/Clustering/SleepDeprivation/RatS/Day3SD/RatS_Day3SD_2020-11-29_07-53-30.xml"
)


tree = ET.parse(path)
myroot = tree.getroot()
# for a in myroot[1][1][-1]:

#     if "Mapping" in a.attrib:
#         print(a.attrib["Mapping"])

for i, chan in enumerate(myroot[2][0][0].iter("channel")):
    print(i)
    chan.text = "1"
    # print(chan.text)
# tree.write(path)