#! /usr/bin/python
import caffe_pb2 as caffe
import sys
import numpy as np
import matplotlib.pyplot as plt

net_param = caffe.NetParameter()

print 'Reading weights from file...'
f = open(sys.argv[1], 'rb')
net_param.ParseFromString(f.read())
f.close()

layers = {}
for layer in net_param.layer:
	if layer.name in ['fc6','fc7','cls_score','bbox_pred']:
		blobs = layer.blobs
		data = np.array(blobs[0].data).reshape(blobs[0].shape.dim)
		if len(blobs) == 1:	
			layers[layer.name] = data
		elif len(blobs) == 2:
			bias = np.array(blobs[1].data).reshape(blobs[1].shape.dim)
			layers[layer.name] = data, bias


# pruning percentage
p = 0.8

print 'Pruning fc6 layer...'
inorm = np.absolute(layers['fc6'][0]).mean(axis=1)
th = inorm[np.argsort(inorm)[int(p*len(inorm))]]
keep1 = inorm>th
data = layers['fc6'][0]
bias = layers['fc6'][1]
layers['fc6'] = data[keep1,:], bias[keep1]

print 'Pruning fc7 layer...'
inorm = np.absolute(layers['fc7'][0]).mean(axis=1)
th = inorm[np.argsort(inorm)[int(p*len(inorm))]]
keep2 = inorm>th
data = layers['fc7'][0][:,keep1]
bias = layers['fc7'][1]
layers['fc7'] = data[keep2,:], bias[keep2]

print 'Adapting cls_score layer...'
data = layers['cls_score'][0]
bias = layers['cls_score'][1]
layers['cls_score'] = data[:,keep2], bias

print 'Adapting bbox_pred layer...'
data = layers['bbox_pred'][0]
bias = layers['bbox_pred'][1]
layers['bbox_pred'] = data[:,keep2], bias


print 'Writing weights to file...'
for layer in net_param.layer:
	if layer.name in ['fc6','fc7','cls_score','bbox_pred']:
		blobs = layer.blobs
		data = layers[layer.name][0]
		bias = layers[layer.name][1]
		blobs[0].data[:] = data.flatten()
		blobs[0].shape.dim[:] = data.shape
		blobs[1].data[:] = bias.flatten()
		blobs[1].shape.dim[:] = bias.shape

f = open(sys.argv[1]+'.pruned','wb')
f.write(net_param.SerializeToString())
f.close()
