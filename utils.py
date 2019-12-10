import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import numpy as np


def val_accuracy(fake, real):
    dist = (fake - real).pow(2).sqrt()
    acc_2 = torch.zeros_like(dist)
    acc_5 = torch.zeros_like(dist)
    acc_2[dist <= 0.04] = 1
    acc_2[dist > 0.04] = 0
    acc_5[dist <= 0.10] = 1
    acc_5[dist > 0.10] = 0
    acc_2 = acc_2.prod(dim=1).mean()
    acc_5 = acc_5.prod(dim=1).mean()

    return acc_2, acc_5


def visual_result(gray, reals, G_model, U_model, epoch=None, mode='val'):
    fakes_G = G_model(Variable(gray.cuda()))
    fakes_U = U_model(Variable(gray.cuda()))

    fakes_G = torch.cat([Variable(gray.cuda()), fakes_G], dim=1)
    fakes_U = torch.cat([Variable(gray.cuda()), fakes_U], dim=1)
    reals = torch.cat([Variable(gray.cuda()), Variable(reals.cuda())], dim=1)

    plt.figure(figsize=(25,10))
    num_plot = 9
    for i in range(num_plot):
        fake_G = fakes_G[i].cpu().detach().numpy()
        fake_U = fakes_U[i].cpu().detach().numpy()
        real = reals[i].cpu().numpy()

        real = ((real + 0.0) * 128.0).astype('uint8')
        fake_G = ((fake_G + 0.0) * 128.0).astype('uint8')
        fake_U = ((fake_U + 0.0) * 128.0).astype('uint8')

        real = cv2.cvtColor(np.transpose(real, (1,2,0)), cv2.COLOR_LAB2RGB)
        fake_G = cv2.cvtColor(np.transpose(fake_G, (1,2,0)), cv2.COLOR_LAB2RGB)
        fake_U = cv2.cvtColor(np.transpose(fake_U, (1, 2, 0)), cv2.COLOR_LAB2RGB)

        plt.subplot(3, num_plot / 3, i+1)
        plt.imshow(np.hstack((real, fake_G, fake_U)))
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('./val/' + 'epoch%d_%s.png' % (epoch, mode))
    plt.clf()


def visual_result_singleModel(gray, reals, model, epoch):
    fakes = model(Variable(gray.cuda()))
    plt.figure(figsize=(25,10))
    num_plot = 9
    for i in range(num_plot):
        fake = fakes[i].cpu().detach().numpy()
        real = reals[i].cpu().numpy()

        real = ((real + 1.0) * 128.0).astype('uint8')
        fake = ((fake + 1.0) * 128.0).astype('uint8')

        real = cv2.cvtColor(np.transpose(real, (1,2,0)), cv2.COLOR_LAB2RGB)
        fake = cv2.cvtColor(np.transpose(fake, (1,2,0)), cv2.COLOR_LAB2RGB)

        plt.subplot(3, num_plot / 3, i+1)
        plt.imshow(np.hstack((real, fake)))
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('./val/' + 'epoch%d_val.png' % epoch)
    plt.clf()