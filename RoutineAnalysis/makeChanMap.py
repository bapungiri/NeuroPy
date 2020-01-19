import xml.etree.ElementTree as ET

basePath = "/data/Clustering/SleepDeprivation/RatJ/Day3/"
filename = basePath + "RatJ_Day3_2019-06-14_04-08-48.xml"
root = ET.parse(filename).getroot()

for i in root[3]:
    print(i.tag, i.text)


chan_session = []
for x in root.findall("anatomicalDescription"):
    for y in x.findall("channelGroups"):
        for z in y.findall("group"):
            for a in z.findall("channel"):
                chan_session.append(int(a.text))

