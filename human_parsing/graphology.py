import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
from gluoncv.data.transforms.presets.segmentation import test_transform
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg

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

    mask = get_color_pallete(predict, 'mhpv1')
    mask.save('{}.png'.format(path_to_img.split('.')[0]))


if __name__ == "__main__":
    # using cpu
    ctx = mx.cpu(0)
    graphology('JZ20-R-TRU5400-012-1-03.jpg', ctx)
