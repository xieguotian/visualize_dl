import caffe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.io import imsave
from visualize_cls_salience import visualize

#class label=99 is goose
label = 99

# load net model
caffe.set_mode_gpu()
net = caffe.Net('./data/deploy.prototxt','./data/bvlc_reference_caffenet.caffemodel',caffe.TEST)

# net surgery
net.params['score'][0].data[...] = np.zeros(net.params['score'][0].data.shape)
net.params['score'][1].data[...] = np.zeros(net.params['score'][1].data.shape)
net.params['score'][0].data[:,:,:,label] = 1

# visual instance
vis_ml = visualize(iter=1000,lambda_=0.0005,nu=2)
#initial image
I = np.random.random(net.blobs['data'].data.shape)*10
#I = np.ones(net.blobs['data'].data.shape)*10

#calculate class visual image
cls_img,s = vis_ml.calc_cls_model(net,I,verbose=True)

#save cls_img
cls_img_name = 'goose.png'
print 'save classification image of goose to {}'.format(cls_img_name)
imsave(cls_img_name,cls_img)

# show result
plt.figure(1)
plt.imshow(cls_img)
plt.show()

