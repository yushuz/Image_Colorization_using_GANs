import torch
from torch import optim, nn
from torch.autograd import Variable
import os
import time
from tqdm import tqdm

from network import Generator_cifar, Discriminator_cifar, U_Net_network_cifar, Generator_256, Discriminator_256, U_Net_network_256
from utils import *


class Cifar10_GAN():
    def __init__(self, learning_rate=1e-4, l1_weight=5, save_path="trained_model.pth.tar"):
        self.trained_epoch = 0
        self.learning_rate = learning_rate

        try:
            self.G_model = Generator_cifar().cuda()
            self.D_model = Discriminator_cifar().cuda()
        except TypeError:
            print("cuda is not available!")

        self.G_optimizer = optim.Adam(self.G_model.parameters(), lr=learning_rate)
        self.D_optimizer = optim.Adam(self.D_model.parameters(), lr=learning_rate)

        self.criterion = nn.BCELoss(reduction='mean')
        self.L1 = nn.L1Loss()
        self.l1_weight = l1_weight

        self.loss_history = []
        self.acc_history = []

        # read trained model
        self.save_path = save_path
        if os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path)
            self.G_model.load_state_dict(checkpoint['G_state_dict'])
            self.D_model.load_state_dict(checkpoint['D_state_dict'])

            self.G_optimizer.load_state_dict(checkpoint['G_optimizer_state_dict'])
            self.D_optimizer.load_state_dict(checkpoint['D_optimizer_state_dict'])

            self.trained_epoch = checkpoint['trained_epoch']
            self.loss_history = checkpoint['loss_history']
            self.acc_history = checkpoint['acc_history']

    def train_one_epoch(self, train_loader, val_loader, epoch):
        self.trained_epoch = epoch
        if epoch % 10 == 0:
          self.l1_weight = self.l1_weight * 0.95
        start = time.time()
        lossesD, lossesD_real, lossesD_fake, lossesG, lossesG_GAN, \
        lossesG_L1, Dreals, Dfakes = [], [], [], [], [], [], [], []

        self.G_model.train()
        self.D_model.train()

        # for gray, color in tqdm(self.train_loader):
        for gray, img_ab in tqdm(train_loader):
            gray = Variable(gray.cuda())
            img_ab = Variable(img_ab.cuda())

            # train D with real image
            self.D_model.zero_grad()
            label = torch.FloatTensor(img_ab.size(0)).cuda()

            D_output = self.D_model(img_ab)
            label_real = Variable(label.fill_(1))
            D_loss_real = self.criterion(torch.squeeze(D_output), label_real)
            D_loss_real.backward()
            Dreal = D_output.data.mean()

            # train D with Generator
            fake_img = self.G_model(gray)
            D_output = self.D_model(fake_img.detach())
            label_fake = Variable(label.fill_(0))

            D_loss_fake = self.criterion(torch.squeeze(D_output), label_fake)
            D_loss_fake.backward()

            lossD = D_loss_real + D_loss_fake
            self.D_optimizer.step()

            # train G
            self.G_model.zero_grad()

            fake_img = self.G_model(gray)
            D_output = self.D_model(fake_img.detach())
            label_real = Variable(label.fill_(1))
            lossG_GAN = self.criterion(torch.squeeze(D_output), label_real)
            lossG_L1 = self.L1(fake_img.view(fake_img.size(0), -1), img_ab.view(img_ab.size(0), -1))

            lossG = lossG_GAN + self.l1_weight * lossG_L1
            lossG.backward()
            Dfake = D_output.data.mean()
            self.G_optimizer.step()

            lossesD.append(lossD)
            lossesD_real.append(D_loss_real)
            lossesD_fake.append(D_loss_fake)
            lossesG.append(lossG)
            lossesG_GAN.append(lossG_GAN)
            lossesG_L1.append(lossG_L1)
            Dreals.append(Dreal)
            Dfakes.append(Dfake)

        end = time.time()
        lossD = torch.stack(lossesD).mean().item()
        D_loss_real = torch.stack(lossesD_real).mean().item()
        D_loss_fake = torch.stack(lossesD_fake).mean().item()
        lossG = torch.stack(lossesG).mean().item()
        lossG_GAN = torch.stack(lossesG_GAN).mean().item()
        lossG_L1 = torch.stack(lossesG_L1).mean().item()
        Dreal = torch.stack(Dreals).mean().item()
        Dfake = torch.stack(Dfakes).mean().item()
        print('loss_D: %.3f (real: %.3f fake: %.3f)  loss_G: %.3f (GAN: %.3f L1: %.3f) D(real): %.3f  '
              'D(fake): %3f  elapsed time %.3f' %
              (lossD, D_loss_real, D_loss_fake, lossG, lossG_GAN, lossG_L1, Dreal, Dfake, end-start))

        lossD_val, lossG_val, acc_2, acc_5 = self.validate(val_loader)

        # update history
        loss_all = [lossD, D_loss_real, D_loss_fake, lossG, lossG_GAN, lossG_L1, lossD_val,
                    lossG_val, Dreal, Dfake]

        if len(self.loss_history):
            self.loss_history = np.hstack((self.loss_history, np.vstack(loss_all)))
            self.acc_history = np.hstack((self.acc_history, np.vstack([acc_2, acc_5])))
        else:
            self.loss_history = np.vstack(loss_all)
            self.acc_history = np.vstack([acc_2, acc_5])

    def validate(self, val_loader):
        lossesD, lossesG, cnt = 0.0, 0.0, 0
        acc_2_list, acc_5_list = [], []

        with torch.no_grad():
            self.G_model.eval()
            self.D_model.eval()

            for gray, color in tqdm(val_loader):
                gray = Variable(gray.cuda())
                color = Variable(color.cuda())

                # for D_model
                label = torch.FloatTensor(color.size(0)).cuda()
                label_real = Variable(label.fill_(1))
                lossD_real = self.criterion(torch.squeeze(self.D_model(color)), label_real)

                fake_img = self.G_model(gray)
                label_fake = Variable(label.fill_(0))
                pred_D_fake = self.D_model(fake_img.detach())
                lossD_fake = self.criterion(torch.squeeze(pred_D_fake), label_fake)

                lossD = lossD_real.item() + lossD_fake.item()

                # for G_model
                lossG_GAN = self.criterion(torch.squeeze(pred_D_fake), label_real)
                lossG_L1 = self.L1(fake_img.view(fake_img.size(0), -1), color.view(color.size(0), -1))
                lossG = lossG_GAN.item() + 100 * lossG_L1.item()

                lossesD += lossD
                lossesG += lossG
                cnt += 1

                acc_2, acc_5 = val_accuracy(fake_img, color)
                acc_2_list.append(acc_2)
                acc_5_list.append(acc_5)

            print('loss_D: %.3f loss_G: %.3f' % (lossesD/cnt, lossesG/cnt))

            acc_2 = torch.stack(acc_2_list).mean().item()
            acc_5 = torch.stack(acc_5_list).mean().item()
            print('GAN: 2%% Validation Accuracy = %.4f' % acc_2)
            print('GAN: 5%% Validation Accuracy = %.4f' % acc_5)

            return lossesD/cnt, lossesG/cnt, acc_2, acc_5

    def test(self, test_loader):
        lossesD, lossesG, cnt = 0.0, 0.0, 0
        acc_2_list, acc_5_list = [], []

        with torch.no_grad():
            self.G_model.eval()
            self.D_model.eval()

            for gray, color in tqdm(test_loader):
                gray = Variable(gray.cuda())
                color = Variable(color.cuda())

                # for D_model
                label = torch.FloatTensor(color.size(0)).cuda()
                label_real = Variable(label.fill_(1))
                lossD_real = self.criterion(torch.squeeze(self.D_model(color)), label_real)

                fake_img = self.G_model(gray)
                label_fake = Variable(label.fill_(0))
                pred_D_fake = self.D_model(fake_img.detach())
                lossD_fake = self.criterion(torch.squeeze(pred_D_fake), label_fake)

                lossD = lossD_real.item() + lossD_fake.item()

                # for G_model
                lossG_GAN = self.criterion(torch.squeeze(pred_D_fake), label_real)
                lossG_L1 = self.L1(fake_img.view(fake_img.size(0), -1), color.view(color.size(0), -1))
                lossG = lossG_GAN.item() + 100 * lossG_L1.item()

                lossesD += lossD
                lossesG += lossG
                cnt += 1

                acc_2, acc_5 = val_accuracy(fake_img, color)
                acc_2_list.append(acc_2)
                acc_5_list.append(acc_5)

            print('loss_D: %.3f loss_G: %.3f' % (lossesD / cnt, lossesG / cnt))

            acc_2 = torch.stack(acc_2_list).mean().item()
            acc_5 = torch.stack(acc_5_list).mean().item()
            print('GAN: 2%% test Accuracy = %.4f' % acc_2)
            print('GAN: 5%% test Accuracy = %.4f' % acc_5)

            return lossesD / cnt, lossesG / cnt, acc_2, acc_5

    def save(self):
        torch.save({'G_state_dict': self.G_model.state_dict(),
                    'D_state_dict': self.D_model.state_dict(),
                    'G_optimizer_state_dict': self.G_optimizer.state_dict(),
                    'D_optimizer_state_dict': self.D_optimizer.state_dict(),
                    'loss_history': self.loss_history,
                    'acc_history': self.acc_history,
                    'trained_epoch': self.trained_epoch}, self.save_path)

    def plot_loss(self):
        loss_name = ['lossD_train', 'lossD_fake', 'lossD_real', 'lossG_train', 'lossG_GAN',
                     'lossG_L1', 'lossD_val', 'lossG_val', 'Dreal', 'Dfake']

        if not os.path.exists("loss_plot/"):
            print('creating directory loss_plot/')
            os.mkdir("loss_plot/")

        for i in range(self.loss_history.shape[0]-2):
            plt.figure()
            plt.plot(self.loss_history[i, :])
            plt.xlabel('epoch'), plt.title('{}'.format(loss_name[i]))
            plt.savefig('./loss_plot/{}_history.png'.format(loss_name[i]))
            plt.clf()
        for i in range(self.loss_history.shape[0]-2, self.loss_history.shape[0]):
            plt.figure()
            plt.plot(self.loss_history[i, :])
            plt.xlabel('epoch'), plt.title('{}'.format(loss_name[i]))
            plt.ylim([-0.1, 1.1])
            plt.savefig('./loss_plot/{}_history.png'.format(loss_name[i]))
            plt.clf()

        plt.figure()
        plt.plot(self.acc_history[0], label='2%% accuracy')
        plt.plot(self.acc_history[1], label='5%% accuracy')
        plt.title('Accuracy History GAN'), plt.legend(), plt.ylim([0,1])
        plt.savefig('./loss_plot/accuracy_history_GAN.png')


class U_Net32():
    def __init__(self, learning_rate=1e-4, save_path="trained_UNet_model.pth.tar"):
        self.trained_epoch = 0
        self.learning_rate = learning_rate

        try:
            self.model = U_Net_network_cifar().cuda()
        except TypeError:
            print("cuda is not available!")

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.loss_train_history = []
        self.loss_val_history = []
        self.acc_history = []

        # read trained model
        self.save_path = save_path
        if os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.trained_epoch = checkpoint['trained_epoch']
            self.loss_train_history = checkpoint['loss_train_history']
            self.loss_val_history = checkpoint['loss_val_history']
            self.acc_history = checkpoint['acc_history']

    def train_one_epoch(self, train_loader, val_loader, epoch):
        self.trained_epoch = epoch
        if epoch % 10 == 0:
            self.learning_rate *= 0.95
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        losses, cnt = 0.0, 0

        self.model.train()

        for gray, color in tqdm(train_loader):
            gray = Variable(gray.cuda())
            color = Variable(color.cuda())

            self.model.zero_grad()
            output = self.model(gray)
            loss = self.criterion(output.view(output.size(0), -1), color.view(color.size(0), -1))
            loss.backward()
            self.optimizer.step()

            losses += loss.item()
            cnt += 1

        print('U-Net training loss: %.3f ' % (losses/cnt))

        loss_val, acc_2, acc_5 = self.validate(val_loader)

        # update history
        self.loss_train_history.append(losses/cnt)
        self.loss_val_history.append(loss_val)
        self.acc_history.append(np.vstack((acc_2, acc_5)))

    def validate(self, val_loader):
        losses, cnt = 0.0, 0
        acc_2_list, acc_5_list = [], []

        with torch.no_grad():
            self.model.eval()

            for gray, color in tqdm(val_loader):
                gray = Variable(gray.cuda())
                color = Variable(color.cuda())

                output = self.model(gray)
                loss = self.criterion(output.view(output.size(0), -1), color.view(color.size(0), -1))
                # F.mse_loss(output, img_LAB).item()
                losses += loss.item()
                cnt += 1

                acc_2, acc_5 = val_accuracy(output, color)
                acc_2_list.append(acc_2)
                acc_5_list.append(acc_5)

            print('U-Net validation loss: %.3f' % (losses/cnt))

            acc_2 = torch.stack(acc_2_list).mean().item()
            acc_5 = torch.stack(acc_5_list).mean().item()
            print('U-Net: 2%% Validation Accuracy = %.4f' % acc_2)
            print('U-Net: 5%% Validation Accuracy = %.4f' % acc_5)

            return losses/cnt, acc_2, acc_5

    def test(self, test_loader):
        losses, cnt = 0.0, 0
        acc_2_list, acc_5_list = [], []

        with torch.no_grad():
            self.model.eval()

            for gray, color in tqdm(test_loader):
                gray = Variable(gray.cuda())
                color = Variable(color.cuda())

                output = self.model(gray)
                loss = self.criterion(output.view(output.size(0), -1), color.view(color.size(0), -1))
                losses += loss.item()
                cnt += 1

                acc_2, acc_5 = val_accuracy(output, color)
                acc_2_list.append(acc_2)
                acc_5_list.append(acc_5)

            print('U-Net test loss: %.3f' % (losses / cnt))

            acc_2 = torch.stack(acc_2_list).mean().item()
            acc_5 = torch.stack(acc_5_list).mean().item()
            print('U-Net: 2%% test Accuracy = %.4f' % acc_2)
            print('U-Net: 5%% test Accuracy = %.4f' % acc_5)

            return losses / cnt, acc_2, acc_5

    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss_train_history': self.loss_train_history,
                    'loss_val_history': self.loss_val_history,
                    'acc_history': self.acc_history,
                    'trained_epoch': self.trained_epoch}, self.save_path)

    def plot_loss(self):
        if not os.path.exists("loss_plot/"):
            print('creating directory loss_plot/')
            os.mkdir("loss_plot/")

        plt.figure()
        plt.plot(self.loss_train_history)
        plt.xlabel('epoch'), plt.title('{}'.format('loss_unet_train'))
        plt.savefig('./loss_plot/{}_history.png'.format('loss_unet_train'))
        plt.clf()

        plt.figure()
        plt.plot(self.loss_val_history)
        plt.xlabel('epoch'), plt.title('{}'.format('loss_unet_val'))
        plt.savefig('./loss_plot/{}_history.png'.format('loss_unet_val'))
        plt.clf()

        plt.figure()
        acc_history = np.hstack(self.acc_history)
        plt.plot(acc_history[0], label='2%% accuracy')
        plt.plot(acc_history[1], label='5%% accuracy')
        plt.title('Accuracy History U-Net'), plt.legend()
        plt.savefig('./loss_plot/accuracy_history_unet.png')


class GAN_256():
    def __init__(self, learning_rate=1e-4, l1_weight=5, save_path="trained_256_model.pth.tar"):
        self.trained_epoch = 0
        self.learning_rate = learning_rate

        try:
            self.G_model = Generator_256().cuda()
            self.D_model = Discriminator_256().cuda()
        except TypeError:
            print("cuda is not available!")

        self.G_optimizer = optim.Adam(self.G_model.parameters(), lr=learning_rate)
        self.D_optimizer = optim.Adam(self.D_model.parameters(), lr=learning_rate)

        self.criterion = nn.BCELoss(reduction='mean')
        self.L1 = nn.L1Loss()
        self.l1_weight = l1_weight

        self.loss_history = []
        self.acc_history = []

        # read trained model
        self.save_path = save_path
        if os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path)
            self.G_model.load_state_dict(checkpoint['G_state_dict'])
            self.D_model.load_state_dict(checkpoint['D_state_dict'])

            self.G_optimizer.load_state_dict(checkpoint['G_optimizer_state_dict'])
            self.D_optimizer.load_state_dict(checkpoint['D_optimizer_state_dict'])

            self.trained_epoch = checkpoint['trained_epoch']
            self.loss_history = checkpoint['loss_history']
            self.acc_history = checkpoint['acc_history']

    def train_one_epoch(self, train_loader, val_loader, epoch):
        self.trained_epoch = epoch
        if epoch % 10 == 0:
          self.l1_weight = self.l1_weight * 0.95
        start = time.time()
        lossesD, lossesD_real, lossesD_fake, lossesG, lossesG_GAN, \
        lossesG_L1, Dreals, Dfakes = [], [], [], [], [], [], [], []

        self.G_model.train()
        self.D_model.train()

        # for gray, color in tqdm(self.train_loader):
        for gray, img_ab in tqdm(train_loader):
            gray = Variable(gray.cuda())
            img_ab = Variable(img_ab.cuda())

            # train D with real image
            self.D_model.zero_grad()
            label = torch.FloatTensor(img_ab.size(0)).cuda()

            D_output = self.D_model(img_ab)
            label_real = Variable(label.fill_(1))
            D_loss_real = self.criterion(torch.squeeze(D_output), label_real)
            D_loss_real.backward()
            Dreal = D_output.data.mean()

            # train D with Generator
            fake_img = self.G_model(gray)
            D_output = self.D_model(fake_img.detach())
            label_fake = Variable(label.fill_(0))

            D_loss_fake = self.criterion(torch.squeeze(D_output), label_fake)
            D_loss_fake.backward()

            lossD = D_loss_real + D_loss_fake
            self.D_optimizer.step()

            # train G
            self.G_model.zero_grad()

            fake_img = self.G_model(gray)
            D_output = self.D_model(fake_img.detach())
            label_real = Variable(label.fill_(1))
            lossG_GAN = self.criterion(torch.squeeze(D_output), label_real)
            lossG_L1 = self.L1(fake_img.view(fake_img.size(0), -1), img_ab.view(img_ab.size(0), -1))

            lossG = lossG_GAN + self.l1_weight * lossG_L1
            lossG.backward()
            Dfake = D_output.data.mean()
            self.G_optimizer.step()

            lossesD.append(lossD)
            lossesD_real.append(D_loss_real)
            lossesD_fake.append(D_loss_fake)
            lossesG.append(lossG)
            lossesG_GAN.append(lossG_GAN)
            lossesG_L1.append(lossG_L1)
            Dreals.append(Dreal)
            Dfakes.append(Dfake)

        end = time.time()
        lossD = torch.stack(lossesD).mean().item()
        D_loss_real = torch.stack(lossesD_real).mean().item()
        D_loss_fake = torch.stack(lossesD_fake).mean().item()
        lossG = torch.stack(lossesG).mean().item()
        lossG_GAN = torch.stack(lossesG_GAN).mean().item()
        lossG_L1 = torch.stack(lossesG_L1).mean().item()
        Dreal = torch.stack(Dreals).mean().item()
        Dfake = torch.stack(Dfakes).mean().item()
        print('loss_D: %.3f (real: %.3f fake: %.3f)  loss_G: %.3f (GAN: %.3f L1: %.3f) D(real): %.3f  '
              'D(fake): %3f  elapsed time %.3f' %
              (lossD, D_loss_real, D_loss_fake, lossG, lossG_GAN, lossG_L1, Dreal, Dfake, end-start))

        lossD_val, lossG_val, acc_2, acc_5 = self.validate(val_loader)

        # update history
        loss_all = [lossD, D_loss_real, D_loss_fake, lossG, lossG_GAN, lossG_L1, lossD_val,
                    lossG_val, Dreal, Dfake]

        if len(self.loss_history):
            self.loss_history = np.hstack((self.loss_history, np.vstack(loss_all)))
            self.acc_history = np.hstack((self.acc_history, np.vstack([acc_2, acc_5])))
        else:
            self.loss_history = np.vstack(loss_all)
            self.acc_history = np.vstack([acc_2, acc_5])

    def validate(self, val_loader):
        lossesD, lossesG, cnt = 0.0, 0.0, 0
        acc_2_list, acc_5_list = [], []

        with torch.no_grad():
            self.G_model.eval()
            self.D_model.eval()

            for gray, color in tqdm(val_loader):
                gray = Variable(gray.cuda())
                color = Variable(color.cuda())

                # for D_model
                label = torch.FloatTensor(color.size(0)).cuda()
                label_real = Variable(label.fill_(1))
                lossD_real = self.criterion(torch.squeeze(self.D_model(color)), label_real)

                fake_img = self.G_model(gray)
                label_fake = Variable(label.fill_(0))
                pred_D_fake = self.D_model(fake_img.detach())
                lossD_fake = self.criterion(torch.squeeze(pred_D_fake), label_fake)

                lossD = lossD_real.item() + lossD_fake.item()

                # for G_model
                lossG_GAN = self.criterion(torch.squeeze(pred_D_fake), label_real)
                lossG_L1 = self.L1(fake_img.view(fake_img.size(0), -1), color.view(color.size(0), -1))
                lossG = lossG_GAN.item() + 100 * lossG_L1.item()

                lossesD += lossD
                lossesG += lossG
                cnt += 1

                acc_2, acc_5 = val_accuracy(fake_img, color)
                acc_2_list.append(acc_2)
                acc_5_list.append(acc_5)

            print('loss_D: %.3f loss_G: %.3f' % (lossesD/cnt, lossesG/cnt))

            acc_2 = torch.stack(acc_2_list).mean().item()
            acc_5 = torch.stack(acc_5_list).mean().item()
            print('GAN: 2%% Validation Accuracy = %.4f' % acc_2)
            print('GAN: 5%% Validation Accuracy = %.4f' % acc_5)

            return lossesD/cnt, lossesG/cnt, acc_2, acc_5

    def test(self, test_loader):
        lossesD, lossesG, cnt = 0.0, 0.0, 0
        acc_2_list, acc_5_list = [], []

        with torch.no_grad():
            self.G_model.eval()
            self.D_model.eval()

            for gray, color in tqdm(test_loader):
                gray = Variable(gray.cuda())
                color = Variable(color.cuda())

                # for D_model
                label = torch.FloatTensor(color.size(0)).cuda()
                label_real = Variable(label.fill_(1))
                lossD_real = self.criterion(torch.squeeze(self.D_model(color)), label_real)

                fake_img = self.G_model(gray)
                label_fake = Variable(label.fill_(0))
                pred_D_fake = self.D_model(fake_img.detach())
                lossD_fake = self.criterion(torch.squeeze(pred_D_fake), label_fake)

                lossD = lossD_real.item() + lossD_fake.item()

                # for G_model
                lossG_GAN = self.criterion(torch.squeeze(pred_D_fake), label_real)
                lossG_L1 = self.L1(fake_img.view(fake_img.size(0), -1), color.view(color.size(0), -1))
                lossG = lossG_GAN.item() + 100 * lossG_L1.item()

                lossesD += lossD
                lossesG += lossG
                cnt += 1

                acc_2, acc_5 = val_accuracy(fake_img, color)
                acc_2_list.append(acc_2)
                acc_5_list.append(acc_5)

            print('loss_D: %.3f loss_G: %.3f' % (lossesD / cnt, lossesG / cnt))

            acc_2 = torch.stack(acc_2_list).mean().item()
            acc_5 = torch.stack(acc_5_list).mean().item()
            print('GAN: 2%% test Accuracy = %.4f' % acc_2)
            print('GAN: 5%% test Accuracy = %.4f' % acc_5)

            return lossesD / cnt, lossesG / cnt, acc_2, acc_5

    def save(self):
        torch.save({'G_state_dict': self.G_model.state_dict(),
                    'D_state_dict': self.D_model.state_dict(),
                    'G_optimizer_state_dict': self.G_optimizer.state_dict(),
                    'D_optimizer_state_dict': self.D_optimizer.state_dict(),
                    'loss_history': self.loss_history,
                    'acc_history': self.acc_history,
                    'trained_epoch': self.trained_epoch}, self.save_path)

    def plot_loss(self):
        loss_name = ['lossD_train', 'lossD_fake', 'lossD_real', 'lossG_train', 'lossG_GAN',
                     'lossG_L1', 'lossD_val', 'lossG_val', 'Dreal', 'Dfake']

        if not os.path.exists("loss_plot/"):
            print('creating directory loss_plot/')
            os.mkdir("loss_plot/")

        for i in range(self.loss_history.shape[0]-2):
            plt.figure()
            plt.plot(self.loss_history[i, :])
            plt.xlabel('epoch'), plt.title('{}'.format(loss_name[i]))
            plt.savefig('./loss_plot/{}_history.png'.format(loss_name[i]))
            plt.clf()
        for i in range(self.loss_history.shape[0]-2, self.loss_history.shape[0]):
            plt.figure()
            plt.plot(self.loss_history[i, :])
            plt.xlabel('epoch'), plt.title('{}'.format(loss_name[i]))
            plt.ylim([-0.1, 1.1])
            plt.savefig('./loss_plot/{}_history.png'.format(loss_name[i]))
            plt.clf()

        plt.figure()
        plt.plot(self.acc_history[0], label='2%% accuracy')
        plt.plot(self.acc_history[1], label='5%% accuracy')
        plt.title('Accuracy History GAN'), plt.legend(), plt.ylim([0,1])
        plt.savefig('./loss_plot/accuracy_history_GAN.png')


class Unet_256():
    def __init__(self, learning_rate=1e-4, save_path="trained_256_UNet_model.pth.tar"):
        self.trained_epoch = 0
        self.learning_rate = learning_rate

        try:
            self.model = U_Net_network_256().cuda()
        except TypeError:
            print("cuda is not available!")

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.loss_train_history = []
        self.loss_val_history = []
        self.acc_history = []

        # read trained model
        self.save_path = save_path
        if os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.trained_epoch = checkpoint['trained_epoch']
            self.loss_train_history = checkpoint['loss_train_history']
            self.loss_val_history = checkpoint['loss_val_history']
            self.acc_history = checkpoint['acc_history']

    def train_one_epoch(self, train_loader, val_loader, epoch):
        self.trained_epoch = epoch
        if epoch % 10 == 0:
            self.learning_rate *= 0.95
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        losses, cnt = 0.0, 0

        self.model.train()

        for gray, color in tqdm(train_loader):
            gray = Variable(gray.cuda())
            color = Variable(color.cuda())

            self.model.zero_grad()
            output = self.model(gray)
            loss = self.criterion(output.view(output.size(0), -1), color.view(color.size(0), -1))
            loss.backward()
            self.optimizer.step()

            losses += loss.item()
            cnt += 1

        print('U-Net training loss: %.3f ' % (losses/cnt))

        loss_val, acc_2, acc_5 = self.validate(val_loader)

        # update history
        self.loss_train_history.append(losses/cnt)
        self.loss_val_history.append(loss_val)
        self.acc_history.append(np.vstack((acc_2, acc_5)))

    def validate(self, val_loader):
        losses, cnt = 0.0, 0
        acc_2_list, acc_5_list = [], []

        with torch.no_grad():
            self.model.eval()

            for gray, color in tqdm(val_loader):
                gray = Variable(gray.cuda())
                color = Variable(color.cuda())

                output = self.model(gray)
                loss = self.criterion(output.view(output.size(0), -1), color.view(color.size(0), -1))
                # F.mse_loss(output, img_LAB).item()
                losses += loss.item()
                cnt += 1

                acc_2, acc_5 = val_accuracy(output, color)
                acc_2_list.append(acc_2)
                acc_5_list.append(acc_5)

            print('U-Net validation loss: %.3f' % (losses/cnt))

            acc_2 = torch.stack(acc_2_list).mean().item()
            acc_5 = torch.stack(acc_5_list).mean().item()
            print('U-Net: 2%% Validation Accuracy = %.4f' % acc_2)
            print('U-Net: 5%% Validation Accuracy = %.4f' % acc_5)

            return losses/cnt, acc_2, acc_5

    def test(self, test_loader):
        losses, cnt = 0.0, 0
        acc_2_list, acc_5_list = [], []

        with torch.no_grad():
            self.model.eval()

            for gray, color in tqdm(test_loader):
                gray = Variable(gray.cuda())
                color = Variable(color.cuda())

                output = self.model(gray)
                loss = self.criterion(output.view(output.size(0), -1), color.view(color.size(0), -1))
                losses += loss.item()
                cnt += 1

                acc_2, acc_5 = val_accuracy(output, color)
                acc_2_list.append(acc_2)
                acc_5_list.append(acc_5)

            print('U-Net test loss: %.3f' % (losses / cnt))

            acc_2 = torch.stack(acc_2_list).mean().item()
            acc_5 = torch.stack(acc_5_list).mean().item()
            print('U-Net: 2%% test Accuracy = %.4f' % acc_2)
            print('U-Net: 5%% test Accuracy = %.4f' % acc_5)

            return losses / cnt, acc_2, acc_5

    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss_train_history': self.loss_train_history,
                    'loss_val_history': self.loss_val_history,
                    'acc_history': self.acc_history,
                    'trained_epoch': self.trained_epoch}, self.save_path)

    def plot_loss(self):
        if not os.path.exists("loss_plot/"):
            print('creating directory loss_plot/')
            os.mkdir("loss_plot/")

        plt.figure()
        plt.plot(self.loss_train_history)
        plt.xlabel('epoch'), plt.title('{}'.format('loss_unet_train'))
        plt.savefig('./loss_plot/{}_history.png'.format('loss_unet_train'))
        plt.clf()

        plt.figure()
        plt.plot(self.loss_val_history)
        plt.xlabel('epoch'), plt.title('{}'.format('loss_unet_val'))
        plt.savefig('./loss_plot/{}_history.png'.format('loss_unet_val'))
        plt.clf()

        plt.figure()
        acc_history = np.hstack(self.acc_history)
        plt.plot(acc_history[0], label='2%% accuracy')
        plt.plot(acc_history[1], label='5%% accuracy')
        plt.title('Accuracy History U-Net'), plt.legend()
        plt.savefig('./loss_plot/accuracy_history_unet.png')