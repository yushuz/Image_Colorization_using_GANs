from torchvision import datasets
import os
import cv2
import numpy as np

if not os.path.exists("cifar_image"):
    os.mkdir("cifar_image")
    os.mkdir("cifar_image/train")
    os.mkdir("cifar_image/test")

if not os.path.exists("image256"):
    os.mkdir("image256")
    os.mkdir("image256/train")

if not os.path.exists("loss_plot"):
    os.mkdir("loss_plot")
if not os.path.exists("val"):
    os.mkdir("val")

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding="latin1")
    return dict


# datasets.CIFAR10('.', download=True)
# batch1 = unpickle("cifar-10-batches-py/data_batch_1")
# batch2 = unpickle("cifar-10-batches-py/data_batch_2")
# batch3 = unpickle("cifar-10-batches-py/data_batch_3")
# batch4 = unpickle("cifar-10-batches-py/data_batch_4")
# batch5 = unpickle("cifar-10-batches-py/data_batch_5")
# test_batch = unpickle("cifar-10-batches-py/test_batch")
# X_train = np.vstack((batch1['data'], batch2['data'], batch3['data'],\
#                      batch4['data'], batch5['data']))
# X_test = test_batch['data']


# datasets.CIFAR100('.', download=True)
# X_train = unpickle("cifar-100-python/train")['data']
# X_test = unpickle("cifar-100-python/test")['data']
#
# for i in range(X_train.shape[0]):
#     img = np.transpose(X_train[i,:].reshape([3,32,32]), (1,2,0))
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     cv2.imwrite('cifar_image/train/{:04d}.png'.format(i), img)
#     cv2.imwrite('cifar_image/train/{:04d}.jpg'.format(i), img)
#
# for i in range(X_test.shape[0]):
#     img_test = np.transpose(X_test[i,:].reshape([3,32,32]), (1,2,0))
#     img_test = cv2.cvtColor(img_test, cv2.COLOR_RGB2BGR)
#     cv2.imwrite('cifar_image/test/{:04d}.png'.format(i), img_test)
#     cv2.imwrite('cifar_image/test/{:04d}.jpg'.format(i), img_test)