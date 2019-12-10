from torch.utils.data import DataLoader, sampler

from models import Cifar10_GAN, U_Net32
from dataset import FacadeDataset
from utils import visual_result


def train(num_epoch, batch_size, learning_rate, l1_weight):

    train_data = FacadeDataset(flag='train', data_range=(0, 49000))
    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_data = FacadeDataset(flag='train', data_range=(49000, 50000))
    val_loader = DataLoader(val_data, batch_size=batch_size)
    # visual_data = next(iter(DataLoader(FacadeDataset(flag='train', data_range=(4900, 4909)), batch_size=9)))
    visual_data = next(iter(DataLoader(val_data, batch_size=9)))

    GAN_cifar = Cifar10_GAN(learning_rate=learning_rate, l1_weight=l1_weight)
    Unet = U_Net32(learning_rate=learning_rate)

    assert GAN_cifar.trained_epoch + 1 < num_epoch
    start_epoch = 0
    if GAN_cifar.trained_epoch > 0:
        start_epoch = GAN_cifar.trained_epoch + 1

    for epoch in range(start_epoch, num_epoch):
        print('----------------------------- epoch {:d} -----------------------------'.format(epoch + 1))
        GAN_cifar.train_one_epoch(train_loader, val_loader, epoch)
        Unet.train_one_epoch(train_loader, val_loader, epoch)

        visual_result(visual_data[0], visual_data[1], GAN_cifar.G_model, Unet.model, epoch+1)

    GAN_cifar.plot_loss()
    Unet.plot_loss()

    GAN_cifar.save()
    Unet.save()


if __name__ == '__main__':
    train(num_epoch=15, batch_size=64, learning_rate=1e-4, l1_weight=5)
