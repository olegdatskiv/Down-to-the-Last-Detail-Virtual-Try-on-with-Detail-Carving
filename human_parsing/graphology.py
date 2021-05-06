import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
from gluoncv.data.transforms.presets.segmentation import test_transform
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg
import numpy as np
from PIL import Image, ImageOps
import cv2

from matplotlib import pyplot as plt


def mxnet_test():
    url = 'https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/segmentation/mhpv1_examples/1.jpg'
    filename = 'mhp_v1_example.jpg'
    gluoncv.utils.download(url, filename, True)

    img = image.imread(filename)

    plt.imshow(img.asnumpy())
    plt.show()

    img = test_transform(img, ctx)

    model = gluoncv.model_zoo.get_model('icnet_resnet50_mhpv1', pretrained=True)

    output = model.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

    mask = get_color_pallete(predict, 'mhpv1')
    mask.save('output.png')
    mmask = mpimg.imread('output.png')
    plt.imshow(mmask)
    plt.show()


def graphology(path_to_img, ctx):
    img = image.imread(path_to_img)
    img = test_transform(img, ctx)
    model = gluoncv.model_zoo.get_model('icnet_resnet50_mhpv1', pretrained=True)

    output = model.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

    mask = get_color_pallete(predict, 'pascal_voc')
    greyscale_img = ImageOps.grayscale(mask)
    print(np.shape(greyscale_img))
    greyscale_img.save('{}.png'.format(path_to_img.split('.')[0]))
    rgb_im = mask.convert('RGB')
    # rgb_im = cv2.cvtColor(np.asarray(rgb_im), cv2.COLOR_RGB2BGR)
    print(np.shape(rgb_im))
    # cv2.imwrite('{}_vis.png'.format(path_to_img.split('.')[0]), rgb_im)
    rgb_im.save('{}_vis_1.png'.format(path_to_img.split('.')[0]))


if __name__ == "__main__":
    # using cpu
    ctx = mx.cpu(0)
    graphology('test3.jpg', ctx)
