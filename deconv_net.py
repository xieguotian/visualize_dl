import numpy as np

class deconv_net(object):
    def __init__(self):
        self.layerNum = 0
        self.params = {}    # params for each layer of reconstruction
        self.layers = []    # layer names in order from bottom to top

    def set_deconv_layer(self,deconv_net,layer_name):
        if not self.params.has_key(layer_name):
            self.layers.append(layer_name)
        self.params[layer_name] = ['deconv',deconv_net]

    def set_relu_layer(self,layer_name):
        if not self.params.has_key(layer_name):
            self.layers.append(layer_name)
        self.params[layer_name]=['relu']

    def set_unpool_layer(self,switch,stride,kernel_size,layer_name):
        if not self.params.has_key(layer_name):
            self.layers.append(layer_name)
        param = [switch,stride,kernel_size]
        self.params[layer_name] = ['unpool',param]

    def recon_down(self,feat,begin):

        begin_flag = False
        idx=0
        recon_feat=[]
        recon_feat.append(feat)

        for key in self.layers:
            if key==begin:
                begin_flag = True

            if begin_flag:
                print key
                val = self.params[key]
                # reconstruct from convolutional layer
                if val[0]=='deconv':
                    idx+=1
                    deconv_net = val[1]
                    input_data = recon_feat[idx-1]
                    deconv_net.forward(data=input_data)
                    tmp_out_feat = deconv_net.blobs['deconv'].data
                    recon_feat.append(tmp_out_feat)

                # reconstruct from max pool layer
                elif val[0]=='unpool':
                    idx+=1
                    switch,stride,kernel_size = val[1]
                    sw_shape = switch.shape

                    up_shape = stride*(sw_shape[2]-1)+kernel_size
                    feat_shape = recon_feat[idx-1].shape
                    poolout = np.zeros((feat_shape[0],feat_shape[1],up_shape,up_shape))
                    last_feat = recon_feat[idx-1]
                    print up_shape
                    for ry in range(0,sw_shape[2]):
                        for rx in range(0,sw_shape[3]):
                            offset = switch[:,:,ry,rx]
                            for nch in range(sw_shape[1]):
                                ix = int(offset[0,nch] % up_shape)
                                iy = int(offset[0,nch] /up_shape)
                                poolout[:,nch,iy,ix] = last_feat[:,nch,ry,rx]

                    recon_feat.append(poolout)

                # reconstruct from relu layer
                elif val[0]=='relu':
                    idx+=1
                    relu_val = recon_feat[idx-1]
                    relu_val[relu_val<0] = 0
                    recon_feat.append(relu_val)
                else:
                    print 'wrong type of layer:{}'.format(key)


        return recon_feat