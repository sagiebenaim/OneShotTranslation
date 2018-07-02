import torch.nn as nn
import torch.nn.functional as F


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


class G11(nn.Module):
    def __init__(self, conv_dim=64):
        super(G11, self).__init__()

        # encoding blocks
        self.conv1 = conv(1, conv_dim, 4)
        self.conv1_svhn = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)

        # residual blocks
        res_dim = conv_dim * 2
        self.conv3 = conv(res_dim, res_dim, 3, 1, 1)
        self.conv4 = conv(res_dim, res_dim, 3, 1, 1)

        # decoding blocks
        self.deconv1 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 1, 4, bn=False)
        self.deconv2_svhn = deconv(conv_dim, 3, 4, bn=False)

    def forward(self, x, svhn=False):
        if svhn:
            out = F.leaky_relu(self.conv1_svhn(x), 0.05)  # (?, 64, 16, 16)
        else:
            out = F.leaky_relu(self.conv1(x), 0.05)  # (?, 64, 16, 16)

        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # ( " )
        out = F.leaky_relu(self.conv4(out), 0.05)  # ( " )
        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 64, 16, 16)

        if svhn:
            out = F.tanh(self.deconv2_svhn(out))  # (?, 3, 32, 32)
        else:
            out = F.tanh(self.deconv2(out))  # (?, 3, 32, 32)

        return out

    def encode(self, x, svhn=False):

        if svhn:
            out = F.leaky_relu(self.conv1_svhn(x), 0.05)  # (?, 64, 16, 16)
        else:
            out = F.leaky_relu(self.conv1(x), 0.05)  # (?, 64, 16, 16)

        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # ( " )

        return out

    def decode(self, out, svhn=False):

        out = F.leaky_relu(self.conv4(out), 0.05)
        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 64, 16, 16)

        if svhn:
            out = F.tanh(self.deconv2_svhn(out))  # (?, 3, 32, 32)
        else:
            out = F.tanh(self.deconv2(out))  # (?, 3, 32, 32)

        return out

    def encode_params(self):
        layers_basic = list(self.conv1_svhn.parameters()) + \
                       list(self.conv1.parameters())
        layers_basic += list(self.conv2.parameters())
        layers_basic += list(self.conv3.parameters())

        return layers_basic

    def decode_params(self):
        layers_basic = list(self.deconv2_svhn.parameters()) + \
                       list(self.deconv2.parameters())
        layers_basic += list(self.deconv1.parameters())
        layers_basic += list(self.conv4.parameters())

        return layers_basic

    def unshared_parameters(self):
        return list(self.deconv2_svhn.parameters()) + list(self.conv1_svhn.parameters()) + \
               list(self.deconv2.parameters()) + list(self.conv1.parameters())


class G22(nn.Module):
    def __init__(self, conv_dim=64):
        super(G22, self).__init__()

        # encoding blocks
        self.conv1 = conv(3, conv_dim, 4)
        self.conv1_mnist = conv(1, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)

        # residual blocks
        res_dim = conv_dim * 2
        self.conv3 = conv(res_dim, res_dim, 3, 1, 1)
        self.conv4 = conv(res_dim, res_dim, 3, 1, 1)

        # decoding blocks
        self.deconv1 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 3, 4, bn=False)
        self.deconv2_mnist = deconv(conv_dim, 1, 4, bn=False)

    def forward(self, x, mnist=False):
        if mnist:
            out = F.leaky_relu(self.conv1_mnist(x), 0.05)  # (?, 64, 16, 16)
        else:
            out = F.leaky_relu(self.conv1(x), 0.05)  # (?, 64, 16, 16)

        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # ( " )
        out = F.leaky_relu(self.conv4(out), 0.05)  # ( " )
        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 64, 16, 16)

        if mnist:
            out = F.tanh(self.deconv2_mnist(out))  # (?, 3, 32, 32)
        else:
            out = F.tanh(self.deconv2(out))  # (?, 3, 32, 32)

        return out

    def encode(self, x, mnist=False):

        if mnist:
            out = F.leaky_relu(self.conv1_mnist(x), 0.05)  # (?, 64, 16, 16)
        else:
            out = F.leaky_relu(self.conv1(x), 0.05)  # (?, 64, 16, 16)

        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # ( " )

        return out

    def decode(self, out, mnist=False):

        out = F.leaky_relu(self.conv4(out), 0.05)
        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 64, 16, 16)

        if mnist:
            out = F.tanh(self.deconv2_mnist(out))  # (?, 3, 32, 32)
        else:
            out = F.tanh(self.deconv2(out))  # (?, 3, 32, 32)

        return out

    def encode_params(self):
        layers_basic = list(self.conv1_mnist.parameters()) + \
                       list(self.conv1.parameters())
        layers_basic += list(self.conv2.parameters())
        layers_basic += list(self.conv3.parameters())

        return layers_basic

    def decode_params(self):
        layers_basic = list(self.deconv2_mnist.parameters()) + \
                       list(self.deconv2.parameters())
        layers_basic += list(self.deconv1.parameters())
        layers_basic += list(self.conv4.parameters())

        return layers_basic

    def unshared_parameters(self):
        return list(self.deconv2_mnist.parameters()) + list(self.conv1_mnist.parameters()) + \
               list(self.deconv2.parameters()) + list(self.conv1.parameters())


class D1(nn.Module):
    """Discriminator for mnist."""

    def __init__(self, conv_dim=64, use_labels=False):
        super(D1, self).__init__()
        self.conv1 = conv(1, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim * 4, n_out, 4, 1, 0, False)

    def forward(self, x_0):
        out = F.leaky_relu(self.conv1(x_0), 0.05)  # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out_0 = self.fc(out).squeeze()

        return out_0


class D2(nn.Module):
    """Discriminator for svhn."""

    def __init__(self, conv_dim=64, use_labels=False):
        super(D2, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim * 4, n_out, 4, 1, 0, False)

    def forward(self, x_0):
        out = F.leaky_relu(self.conv1(x_0), 0.05)  # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out_0 = self.fc(out).squeeze()

        return out_0
