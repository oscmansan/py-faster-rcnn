#!/usr/bin/python
import re
import sys
import matplotlib.pyplot as plt

log_file = sys.argv[1]
f = open(log_file)
points = []
for line in f.readlines():
    if re.search('Iteration.*loss', line):
        it, loss = re.findall('Iteration (.*?), loss = (.*?)\n', line)[0]
	points.append((int(it),float(loss)))

it, loss = zip(*points)
fig = plt.figure(log_file)
plt.plot(it,loss)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()
