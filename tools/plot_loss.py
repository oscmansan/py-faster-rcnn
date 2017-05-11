#!/usr/bin/python
import re
import sys
import matplotlib.pyplot as plt

f = open(sys.argv[1])
points = []
for line in f.readlines():
    if re.search('Iteration.*loss', line):
        it, loss = re.findall('Iteration (.*?), loss = (.*?)\n', line)[0]
	points.append((int(it),float(loss)))

it, loss = zip(*points)
plt.plot(it,loss)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()
