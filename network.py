import torch
import torch.nn as nn
import math


class Generator_cifar(nn.Module):
    def __init__(self):
        super().__init__()

        # networks layers here
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.traver6 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        self.traver7 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(256)

        self.traver8 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(128)

        self.traver9 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.bn9 = nn.BatchNorm2d(64)

        self.traver10 = nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0)

        self.drop = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, input):
        x = self.conv1(input)
        x = self.leakyrelu(x)
        x = self.bn1(x)
        feature1 = x

        x = self.conv2(x)
        x = self.leakyrelu(x)
        x = self.bn2(x)
        feature2 = x

        x = self.conv3(x)
        x = self.leakyrelu(x)
        x = self.bn3(x)
        feature3 = x

        x = self.conv4(x)
        x = self.leakyrelu(x)
        x = self.bn4(x)
        feature4 = x

        x = self.conv5(x)
        x = self.leakyrelu(x)
        x = self.bn5(x)

        x = self.traver6(x)
        x = self.relu(x)
        x = self.bn6(x)
        x = self.drop(x)
        x = torch.cat((x, feature4), dim=1)

        x = self.traver7(x)
        x = self.relu(x)
        x = self.bn7(x)
        x = self.drop(x)
        x = torch.cat((x, feature3), dim=1)

        x = self.traver8(x)
        x = self.relu(x)
        x = self.bn8(x)
        x = torch.cat((x, feature2), dim=1)

        x = self.traver9(x)
        x = self.relu(x)
        x = self.bn9(x)
        x = torch.cat((x, feature1), dim=1)

        x = self.traver10(x)
        x = self.relu(x)

        return x


class Discriminator_cifar(nn.Module):
    def __init__(self):
        super().__init__()

        # networks layers here
        self.conv1 = nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, input):
        x = self.conv1(input)
        x = self.leakyrelu(x)

        x = self.conv2(x)
        x = self.leakyrelu(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.leakyrelu(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.leakyrelu(x)
        x = self.bn4(x)

        x = self.conv5(x)
        x = torch.sigmoid(x)

        return x


class U_Net_network_cifar(nn.Module):
    def __init__(self):
        super().__init__()

        # networks layers here
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.traver6 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        self.traver7 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(256)

        self.traver8 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(128)

        self.traver9 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.bn9 = nn.BatchNorm2d(64)

        self.traver10 = nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0)

        self.drop = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, input):
        x = self.conv1(input)
        x = self.leakyrelu(x)
        x = self.bn1(x)
        feature1 = x

        x = self.conv2(x)
        x = self.leakyrelu(x)
        x = self.bn2(x)
        feature2 = x

        x = self.conv3(x)
        x = self.leakyrelu(x)
        x = self.bn3(x)
        feature3 = x

        x = self.conv4(x)
        x = self.leakyrelu(x)
        x = self.bn4(x)
        feature4 = x

        x = self.conv5(x)
        x = self.leakyrelu(x)
        x = self.bn5(x)

        x = self.traver6(x)
        x = self.relu(x)
        x = self.bn6(x)
        x = self.drop(x)
        x = torch.cat((x, feature4), dim=1)

        x = self.traver7(x)
        x = self.relu(x)
        x = self.bn7(x)
        x = self.drop(x)
        x = torch.cat((x, feature3), dim=1)

        x = self.traver8(x)
        x = self.relu(x)
        x = self.bn8(x)
        x = torch.cat((x, feature2), dim=1)

        x = self.traver9(x)
        x = self.relu(x)
        x = self.bn9(x)
        x = torch.cat((x, feature1), dim=1)

        x = self.traver10(x)
        x = self.relu(x)

        return x


class Generator_256(nn.Module):
    def __init__(self):
        super().__init__()

        # networks layers here
        # [batch, 256, 256, ch] => [batch, 256, 256, 64]
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.leakyrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()

        # [batch, 256, 256, 64] => [batch, 128, 128, 64]
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # [batch, 128, 128, 64] => [batch, 64, 64, 128]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # [batch, 64, 64, 128] => [batch, 32, 32, 256]
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # [batch, 32, 32, 256] => [batch, 16, 16, 512]
        self.conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        # [batch, 16, 16, 512] => [batch, 8, 8, 512]
        self.conv6 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        # [batch, 8, 8, 512] => [batch, 4, 4, 512]
        self.conv7 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(512)

        # [batch, 4, 4, 512] => [batch, 2, 2, 512]
        self.conv8 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        # [batch, 2, 2, 512] => [batch, 4, 4, 512]
        self.traver9 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn9 = nn.BatchNorm2d(512)

        # [batch, 4, 4, 512] => [batch, 8, 8, 512]
        self.traver10 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.bn10 = nn.BatchNorm2d(512)

        # [batch, 8, 8, 512] => [batch, 16, 16, 512]
        self.traver11 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.bn11 = nn.BatchNorm2d(512)

        # [batch, 16, 16, 512] => [batch, 32, 32, 256]
        self.traver12 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1)
        self.bn12 = nn.BatchNorm2d(256)

        # [batch, 32, 32, 256] => [batch, 64, 64, 128]
        self.traver13 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.bn13 = nn.BatchNorm2d(128)

        # [batch, 64, 64, 128] => [batch, 128, 128, 64]
        self.traver14 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.bn14 = nn.BatchNorm2d(64)

        # [batch, 128, 128, 64] => [batch, 256, 256, 64]
        self.traver15 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn15 = nn.BatchNorm2d(64)

        # [batch, 256, 256, 64] => [batch, 256, 256, 3]
        self.traver16 = nn.ConvTranspose2d(128, 2, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        feature1 = x

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leakyrelu(x)
        feature2 = x

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leakyrelu(x)
        feature3 = x

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leakyrelu(x)
        feature4 = x

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.leakyrelu(x)
        feature5 = x

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.leakyrelu(x)
        feature6 = x

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.leakyrelu(x)
        feature7 = x

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.leakyrelu(x)

        x = self.traver9(x)
        x = self.bn9(x)
        x = self.relu(x)
        x = torch.cat([x, feature7], dim=1)

        x = self.traver10(x)
        x = self.bn10(x)
        x = self.relu(x)
        x = torch.cat([x, feature6], dim=1)

        x = self.traver11(x)
        x = self.bn11(x)
        x = self.relu(x)
        x = torch.cat([x, feature5], dim=1)

        x = self.traver12(x)
        x = self.bn12(x)
        x = self.relu(x)
        x = torch.cat([x, feature4], dim=1)

        x = self.traver13(x)
        x = self.bn13(x)
        x = self.relu(x)
        x = torch.cat([x, feature3], dim=1)

        x = self.traver14(x)
        x = self.bn14(x)
        x = self.relu(x)
        x = torch.cat([x, feature2], dim=1)

        x = self.traver15(x)
        x = self.bn15(x)
        x = self.relu(x)
        x = torch.cat([x, feature1], dim=1)

        x = self.traver16(x)
        x = self.relu(x)

        return x


class Discriminator_256(nn.Module):
    def __init__(self):
        super().__init__()

        # networks layers here
        self.conv1 = nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.leakyrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 1, kernel_size=32, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.leakyrelu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leakyrelu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leakyrelu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leakyrelu(x)

        x = self.conv5(x)
        x = torch.sigmoid(x)

        return x


class U_Net_network_256(nn.Module):
    def __init__(self):
        super().__init__()

        # networks layers here
        # [batch, 256, 256, ch] => [batch, 256, 256, 64]
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.leakyrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()

        # [batch, 256, 256, 64] => [batch, 128, 128, 64]
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # [batch, 128, 128, 64] => [batch, 64, 64, 128]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # [batch, 64, 64, 128] => [batch, 32, 32, 256]
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # [batch, 32, 32, 256] => [batch, 16, 16, 512]
        self.conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        # [batch, 16, 16, 512] => [batch, 8, 8, 512]
        self.conv6 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        # [batch, 8, 8, 512] => [batch, 4, 4, 512]
        self.conv7 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(512)

        # [batch, 4, 4, 512] => [batch, 2, 2, 512]
        self.conv8 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        # [batch, 2, 2, 512] => [batch, 4, 4, 512]
        self.traver9 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn9 = nn.BatchNorm2d(512)

        # [batch, 4, 4, 512] => [batch, 8, 8, 512]
        self.traver10 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.bn10 = nn.BatchNorm2d(512)

        # [batch, 8, 8, 512] => [batch, 16, 16, 512]
        self.traver11 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.bn11 = nn.BatchNorm2d(512)

        # [batch, 16, 16, 512] => [batch, 32, 32, 256]
        self.traver12 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1)
        self.bn12 = nn.BatchNorm2d(256)

        # [batch, 32, 32, 256] => [batch, 64, 64, 128]
        self.traver13 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.bn13 = nn.BatchNorm2d(128)

        # [batch, 64, 64, 128] => [batch, 128, 128, 64]
        self.traver14 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.bn14 = nn.BatchNorm2d(64)

        # [batch, 128, 128, 64] => [batch, 256, 256, 64]
        self.traver15 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn15 = nn.BatchNorm2d(64)

        # [batch, 256, 256, 64] => [batch, 256, 256, 3]
        self.traver16 = nn.ConvTranspose2d(128, 2, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        feature1 = x

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leakyrelu(x)
        feature2 = x

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leakyrelu(x)
        feature3 = x

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leakyrelu(x)
        feature4 = x

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.leakyrelu(x)
        feature5 = x

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.leakyrelu(x)
        feature6 = x

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.leakyrelu(x)
        feature7 = x

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.leakyrelu(x)

        x = self.traver9(x)
        x = self.bn9(x)
        x = self.relu(x)
        x = torch.cat([x, feature7], dim=1)

        x = self.traver10(x)
        x = self.bn10(x)
        x = self.relu(x)
        x = torch.cat([x, feature6], dim=1)

        x = self.traver11(x)
        x = self.bn11(x)
        x = self.relu(x)
        x = torch.cat([x, feature5], dim=1)

        x = self.traver12(x)
        x = self.bn12(x)
        x = self.relu(x)
        x = torch.cat([x, feature4], dim=1)

        x = self.traver13(x)
        x = self.bn13(x)
        x = self.relu(x)
        x = torch.cat([x, feature3], dim=1)

        x = self.traver14(x)
        x = self.bn14(x)
        x = self.relu(x)
        x = torch.cat([x, feature2], dim=1)

        x = self.traver15(x)
        x = self.bn15(x)
        x = self.relu(x)
        x = torch.cat([x, feature1], dim=1)

        x = self.traver16(x)
        x = self.relu(x)

        return x