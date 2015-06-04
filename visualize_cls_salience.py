import numpy as np
import matplotlib.pyplot as plt

class visualize(object):
    """
    an implementation of paper:
        Simonyan, Karen, Andrea Vedaldi, and Andrew Zisserman. "Deep inside convolutional networks: Visualising image classification models and saliency maps." arXiv preprint arXiv:1312.6034 (2013).
    """

    def __init__(self, iter=2000, lambda_=0.0005, nu=0.0005):
        self.iter = iter    #number of iteration
        self.lambda_ = lambda_ #ratio between score item and regularization item
        self.nu = nu          #learn rate


    def calc_cls_model(self, net, muImg=None, verbose=False):
        """
        calculate an image to show what a class of a deep network looks like
        """
        if not muImg==None:
            Img = muImg
        else:
            I_shape = net.blobs['data'].data.shape
            Img = np.random.random(I_shape)

        print 'calculating class model...'
        if verbose:
            plt.figure(0)
        for i in range(self.iter):
            # net forward and backward
            fw = net.forward(data=Img)
            bw = net.backward(score=net.blobs['score'].data)

            diff = bw['data']
            diff = -diff

            # calculate total gradient of image
            diff = diff + 2*self.lambda_*Img

            #update image
            Img = Img - self.nu * diff

            #show changes of image
            if i % 100==0 and verbose:
                print 'max-min Img:({},{}),max diff:{}'.format(Img.max(),Img.min(),diff.max())
                print 'iter %d for class model,score %f' % (i,fw['score'][0])
                tmpImg = Img[0].transpose([1,2,0]).copy()
                tmpImg -= tmpImg.min()
                tmpImg /= tmpImg.max()
                plt.imshow(tmpImg)
                plt.show(block=False)
                plt.draw()

        if verbose:
            plt.close()
        fw = net.forward(data=Img)
        Img = Img[0].transpose([1,2,0])
        Img -= Img.min()
        Img /= Img.max()
        return (Img,fw)

    def calc_cls_saliency(self, net, Img):
        """
        calculate an salience map of a given image for a specific class of a network
        """
        fw = net.forward(data=Img)
        bw = net.backward(score=net.blobs['score'].data)

        diff = bw['data']
        diff = diff[0].transpose(1,2,0)
        diff = diff.max(2)
        diff -= diff.min()
        diff /= diff.max()
        return [diff,fw]