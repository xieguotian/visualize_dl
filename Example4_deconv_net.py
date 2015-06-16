import caffe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.io import imsave
from visualize_cls_salience import visualize
from deconv_net import deconv_net

# load net model
caffe.set_mode_gpu()
net = caffe.Net('./data/deploy3.prototxt','./data/bvlc_reference_caffenet.caffemodel',caffe.TEST)

# show net details
print '\n blobs of caffe  model:'
for k,value in net.blobs.items():
    print '{}:{}'.format(k,value.data.shape)

print '\n params of caffe model:'
for k,value in net.params.items():
    print '{}:{},{}'.format(k,value[0].data.shape,value[1].data.shape)


# set deconv_net
deconv5 = caffe.Net('./data/deconv5.prototxt',caffe.TEST)
deconv4 = caffe.Net('./data/deconv4.prototxt',caffe.TEST)
deconv3 = caffe.Net('./data/deconv3.prototxt',caffe.TEST)
deconv2 = caffe.Net('./data/deconv2.prototxt',caffe.TEST)
deconv1 = caffe.Net('./data/deconv1.prototxt',caffe.TEST)
deconv5.params['deconv'][0].data[...] = net.params['conv5'][0].data
deconv4.params['deconv'][0].data[...] = net.params['conv4'][0].data
deconv3.params['deconv'][0].data[...] = net.params['conv3'][0].data
deconv2.params['deconv'][0].data[...] = net.params['conv2'][0].data
deconv1.params['deconv'][0].data[...] = net.params['conv1'][0].data

print '\n blobs of caffe model:'
for k,value in deconv5.blobs.items():
    print '{}:{}'.format(k,value.data.shape)

for k,value in deconv4.blobs.items():
    print '{}:{}'.format(k,value.data.shape)

for k,value in deconv3.blobs.items():
    print '{}:{}'.format(k,value.data.shape)

for k,value in deconv2.blobs.items():
    print '{}:{}'.format(k,value.data.shape)

for k,value in deconv1.blobs.items():
    print '{}:{}'.format(k,value.data.shape)


print '\n params of caffe model:'
for k,value in deconv5.params.items():
    print '{}:{}'.format(k,value[0].data.shape,value[1].data.shape)

for k,value in deconv4.params.items():
    print '{}:{},{}'.format(k,value[0].data.shape,value[1].data.shape)

for k,value in deconv3.params.items():
    print '{}:{},{}'.format(k,value[0].data.shape,value[1].data.shape)

for k,value in deconv2.params.items():
    print '{}:{},{}'.format(k,value[0].data.shape,value[1].data.shape)

for k,value in deconv1.params.items():
    print '{}:{},{}'.format(k,value[0].data.shape,value[1].data.shape)

deconv = deconv_net()
deconv.set_unpool_layer(net.blobs['pool5_mask'].data,2,3,'pool5')
deconv.set_relu_layer('relu5')
deconv.set_deconv_layer(deconv5,'conv5')
deconv.set_relu_layer('relu4')
deconv.set_deconv_layer(deconv4,'conv4')
deconv.set_relu_layer('relu3')
deconv.set_deconv_layer(deconv3,'conv3')
deconv.set_unpool_layer(net.blobs['pool2_mask'].data,2,3,'pool2')
deconv.set_relu_layer('relu2')
deconv.set_deconv_layer(deconv2,'conv2')
deconv.set_unpool_layer(net.blobs['pool1_mask'].data,2,3,'pool1')
deconv.set_relu_layer('relu1')
deconv.set_deconv_layer(deconv1,'conv1')

# read image
im = caffe.io.load_image('./data/person.JPEG')

transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_mean('data',np.load('./data/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_channel_swap('data',[2,1,0])
transformer.set_raw_scale('data',255)
transformer.set_transpose('data',[2,0,1])

#transform image to net input
input_img = transformer.preprocess('data',im)
input_img = input_img.reshape([1] + [input_img.shape[i] for i in range(3)])

#get feature
out = net.forward(data=input_img)

# set switch of pool
deconv.set_unpool_layer(net.blobs['pool5_mask'].data,2,3,'pool5')
deconv.set_unpool_layer(net.blobs['pool2_mask'].data,2,3,'pool2')
deconv.set_unpool_layer(net.blobs['pool1_mask'].data,2,3,'pool1')

# find top activation and reconstruction
top_act = np.zeros(net.blobs['pool5'].data.shape)
layer_feat_map = net.blobs['pool5'].data

top_act[layer_feat_map==layer_feat_map.max()] = layer_feat_map.max()
recon_feat = deconv.recon_down(top_act,'pool5')

# show reconstruction image
image = recon_feat[-1][0]
image -= image.min()
image /= image.max()
image = image.transpose([1,2,0])
re_image_name = './result/deconv_net_5.png'
imsave(re_image_name,image)

plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(image)
plt.subplot(1,2,2)
plt.imshow(transformer.deprocess('data',input_img[0]))
plt.show()
