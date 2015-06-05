import caffe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.io import imsave
from visualize_cls_salience import visualize

# load net model
caffe.set_mode_gpu()
net = caffe.Net('./data/deploy2.prototxt','./data/bvlc_reference_caffenet.caffemodel',caffe.TEST)

# net surgery
"""
net.params['score'][0].data[...] = np.zeros(net.params['score'][0].data.shape)
net.params['score'][1].data[...] = np.zeros(net.params['score'][1].data.shape)
net.params['score'][0].data[:,:,:,label] = 1
"""

# visual instance
vis_ml = visualize(iter=1000,lambda_=0.001,nu=0.05)

# read image
im = caffe.io.load_image('./data/cat.jpg')

transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_mean('data',np.load('./data/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_channel_swap('data',[2,1,0])
transformer.set_raw_scale('data',255)
transformer.set_transpose('data',[2,0,1])

#transform image to net input
input_img = transformer.preprocess('data',im)
input_img = input_img.reshape([1] + [input_img.shape[i] for i in range(3)])

#get feature
net.forward(data=input_img,feat=np.zeros(net.blobs['feat'].data.shape))
feat = net.blobs['conv4'].data

#begin reconstruct
init_img = np.random.random(net.blobs['data'].data.shape)
re_img,s = vis_ml.feat_reconstruct(net,feat,init_img,verbose=True)


#save and show reconstruction result
re_img_name = './result/conv4_reconstruct_cat.png'
imsave(re_img_name,re_img)
plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(re_img)
plt.subplot(1,2,2)
plt.imshow(transformer.deprocess('data',transformer.preprocess('data',im)))
plt.show()
