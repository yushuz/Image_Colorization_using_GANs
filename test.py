from torch.utils.data import DataLoader, sampler

from models import Cifar10_GAN, U_Net32, GAN_256, Unet_256
from dataset import FacadeDataset, FacadeDataset_256
from utils import visual_result, visual_result_four


def test(batch_size, test_range, size=32, only_visual=False):
    if size == 32:
        test_date = FacadeDataset(flag='test', data_range=(0, test_range))
        # GAN_model = Cifar10_GAN()
        # Unet = U_Net32()
        GAN_model = Cifar10_GAN(save_path="trained_model_best.pth.tar")
        Unet = U_Net32(save_path="trained_UNet_model_best.pth.tar")
    else:
        test_date = FacadeDataset_256(flag='train', data_range=(0, test_range))
        GAN_model = GAN_256()
        Unet = Unet_256()

    test_loader = DataLoader(test_date, batch_size=batch_size)
    visual_data = next(iter(DataLoader(test_date, batch_size=9,
                                       sampler=sampler.SubsetRandomSampler(range(test_range)))))
    if not only_visual:
        GAN_model.test(test_loader)
        Unet.test(test_loader)

    # visual_result(visual_data[0], visual_data[1], GAN_model.G_model, Unet.model, mode='test')
    visual_result_four(visual_data[0], visual_data[1], GAN_model.G_model, Unet.model, mode='test')


if __name__ == '__main__':
    test(batch_size=64, test_range=1000, size=32, only_visual=True)