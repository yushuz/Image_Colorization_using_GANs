import torch
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


def visual_result(fake_batch, base_batch, real_batch, epoch):
    fake_batch = fake_batch[:9,::]
    plt.figure(figsize=(25,10))
    num_plot = 9
    for i in range(num_plot):
        real = real_batch[i].cpu().numpy()
        pred = fake_batch[i].cpu().detach().numpy()
        pred_2 = base_batch[i].cpu().detach().numpy()

        real = ((real + 1.0) * 128.0).astype('uint8')
        pred = ((pred + 1.0) * 128.0).astype('uint8')
        pred_2 = ((pred_2 + 1.0) * 128.0).astype('uint8')

        real = cv2.cvtColor(np.transpose(real, (1,2,0)), cv2.COLOR_LAB2RGB)
        pred = cv2.cvtColor(np.transpose(pred, (1,2,0)), cv2.COLOR_LAB2RGB)
        pred_2 = cv2.cvtColor(np.transpose(pred_2, (1, 2, 0)), cv2.COLOR_LAB2RGB)

        plt.subplot(3, num_plot / 3, i+1)
        plt.imshow(np.hstack((real, pred, pred_2)))
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('./val/' + 'epoch%d_val.png' % epoch)
    plt.clf()