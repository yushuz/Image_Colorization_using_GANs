from torch.utils.data import DataLoader

from models import GAN_256, Unet_256
from dataset import FacadeDataset_256
from utils import visual_result


def train(num_epoch, batch_size, learning_rate, l1_weight):

    train_data = FacadeDataset_256(flag='train', data_range=(0, 500))
    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_data = FacadeDataset_256(flag='train', data_range=(500, 600))
    val_loader = DataLoader(val_data, batch_size=batch_size)
    visual_data = next(iter(DataLoader(val_data, batch_size=9)))

    GAN_cifar = GAN_256(learning_rate=learning_rate, l1_weight=l1_weight)
    Unet = Unet_256(learning_rate=learning_rate)

    assert GAN_cifar.trained_epoch + 1 < num_epoch
    start_epoch = 0
    if GAN_cifar.trained_epoch > 0:
        start_epoch = GAN_cifar.trained_epoch + 1

    for epoch in range(start_epoch, num_epoch):
        print('----------------------------- epoch {:d} -----------------------------'.format(epoch + 1))
        GAN_cifar.train_one_epoch(train_loader, val_loader, epoch)
        Unet.train_one_epoch(train_loader, val_loader, epoch)

        visual_result(visual_data[0], visual_data[1], GAN_cifar.G_model, Unet.model, GAN_cifar.trained_epoch+1)

    GAN_cifar.plot_loss()
    Unet.plot_loss()

    GAN_cifar.save()
    Unet.save()


if __name__ == '__main__':
    train(num_epoch=20, batch_size=16, learning_rate=1e-4, l1_weight=100)