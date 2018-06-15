import os

import numpy as np
import scipy.io
import torch
from torch import optim
from torch.autograd import Variable

from mnist_to_svhn.model import D1, D2
from mnist_to_svhn.model import G11


class Solver(object):
    def __init__(self, config, svhn_loader, mnist_loader):
        self.config = config
        self.svhn_loader = svhn_loader
        self.mnist_loader = mnist_loader
        self.g11 = None
        self.g22 = None
        self.d1 = None
        self.d2 = None
        self.g_optimizer = None
        self.num_classes = config.num_classes
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.train_iters = config.train_iters
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.kl_lambda = config.kl_lambda
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.sample_path = config.sample_path
        self.model_path = config.model_path
        self.g11_load_path = os.path.join(config.load_path, "g11-" + str(config.load_iter) + ".pkl")
        self.d1_load_path = os.path.join(config.load_path, "d1-" + str(config.load_iter) + ".pkl")
        self.g22_load_path = os.path.join(config.load_path, "g22-" + str(config.load_iter) + ".pkl")
        self.d2_load_path = os.path.join(config.load_path, "d2-" + str(config.load_iter) + ".pkl")
        self.build_model()


    def build_model(self):
        """Builds a generator and a discriminator."""
        self.g11 = G11(conv_dim=self.g_conv_dim)
        self.g_optimizer = optim.Adam(list(self.g11.encode_params()) + list(self.g11.decode_params()), self.lr,
                                      [self.beta1, self.beta2])
        self.unshared_optimizer = optim.Adam(list(self.g11.unshared_parameters()), self.lr,
                                             [self.beta1, self.beta2])

        self.d1 = D1(conv_dim=self.d_conv_dim, use_labels=False)
        self.d2 = D2(conv_dim=self.d_conv_dim, use_labels=False)

        self.d_optimizer = optim.Adam(list(self.d1.parameters()) + list(self.d2.parameters()), self.lr, [self.beta1, self.beta2])
        
        if torch.cuda.is_available():
            self.g11.cuda()
            self.d1.cuda()
            self.d2.cuda()
    
    def merge_images(self, sources, targets, k=10):
        _, _, h, w = sources.shape
        row = int(np.sqrt(self.batch_size)) + 1
        merged = np.zeros([3, row*h, row*w*2])
        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[:, i*h:(i+1)*h, (j*2)*h:(j*2+1)*h] = s
            merged[:, i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h] = t
        return merged.transpose(1, 2, 0)
    
    def to_var(self, x, volatile=False):
        """Converts numpy to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        if volatile:
            return Variable(x, volatile=True)
        return Variable(x)

    def to_no_grad_var(self, x):
        x = self.to_data(x, no_numpy=True)
        return self.to_var(x, volatile=True)
    
    def to_data(self, x, no_numpy=False):
        """Converts variable to numpy."""
        if torch.cuda.is_available():
            x = x.cpu()
        if no_numpy:
            return x.data
        return x.data.numpy()
    
    def reset_grad(self):
        """Zeros the gradient buffers."""
        self.unshared_optimizer.zero_grad()
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def _compute_kl(self, mu):

        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def train(self):
        self.build_model()
        if self.config.pretrained_g:
            self.g11.load_state_dict(torch.load(self.g11_load_path))

        svhn_iter = iter(self.svhn_loader)
        mnist_iter = iter(self.mnist_loader)
        iter_per_epoch = min(len(svhn_iter), len(mnist_iter))
        
        # fixed mnist and svhn for sampling
        svhn_fixed_data, svhn_fixed_labels = svhn_iter.next()
        mnist_fixed_data, mnist_fixed_labels = mnist_iter.next()
        fixed_mnist = self.to_var(mnist_fixed_data)
        counter = 0
        
        for step in range(self.train_iters+1):

            # reset data_iter for each epoch
            if (step+1) % iter_per_epoch == 0:
                mnist_iter = iter(self.mnist_loader)
                svhn_iter = iter(self.svhn_loader)
            
            # load svhn and mnist dataset
            svhn_data, s_labels_data = svhn_iter.next()
            mnist_data, m_labels_data = mnist_iter.next()
            svhn, s_labels = self.to_var(svhn_data), self.to_var(s_labels_data).long().squeeze()
            mnist, m_labels = self.to_var(mnist_data), self.to_var(m_labels_data)

            # This sets the maximum number of items for A domain
            # We assume max_items is a multiple of batch_size
            # And reset mnist loader when we pass the number of allowed items.
            if self.batch_size > self.config.max_items:
                exit(-1)
            elif self.batch_size == self.config.max_items:
                mnist = fixed_mnist
            elif self.batch_size < self.config.max_items:
                counter += 1
                if counter * self.batch_size >= self.config.max_items:
                    mnist_iter = iter(self.mnist_loader)
                    counter = 0

            #============ train D ============#
            # train with real images
            self.reset_grad()
            out = self.d1(mnist)
            d1_loss = torch.mean((out - 1) ** 2)

            out = self.d2(svhn)
            d2_loss = torch.mean((out - 1) ** 2)

            d_mnist_loss = d1_loss
            d_svhn_loss = d2_loss
            # Only optimizing d1
            d_real_loss = d1_loss + d2_loss
            d_real_loss.backward()
            self.d_optimizer.step()

            # train with fake images
            self.reset_grad()
            es = self.g11.encode(svhn, svhn=True)
            fake_mnist = self.g11.decode(es)
            out = self.d1(fake_mnist)
            d2_loss = torch.mean(out ** 2)

            em = self.g11.encode(mnist)
            fake_svhn = self.g11.decode(em, svhn=True)
            out = self.d2(fake_svhn)
            d1_loss = torch.mean(out ** 2)

            d_fake_loss = d2_loss + d1_loss
            d_fake_loss.backward()
            self.d_optimizer.step()

            #============ train G ============#

            # train mnist-svhn-mnist cycle
            self.reset_grad()
            es = self.g11.encode(svhn, svhn=True)
            fake_mnist = self.g11.decode(es)
            out = self.d1(fake_mnist)
            g_loss = torch.mean((out - 1) ** 2)

            em = self.g11.encode(mnist)
            fake_svhn = self.g11.decode(em, svhn=True)
            out = self.d2(fake_svhn)
            g_loss += torch.mean((out - 1) ** 2)

            self.reset_grad()
            em = self.g11.encode(mnist)
            fake_mnist = self.g11.decode(em)
            g_loss += torch.mean((mnist - fake_mnist) ** 2)

            if self.config.one_way_cycle:
                em = self.g11.encode(mnist)
                fake_svhn = self.g11.decode(em, svhn=True)
                es = self.g11.encode(fake_svhn, svhn=True)
                fake_mnist = self.g11.decode(es)
                g_loss += torch.mean((mnist - fake_mnist) ** 2)

            g_loss.backward()
            self.unshared_optimizer.step()

            if not self.config.freeze_shared:
                self.reset_grad()
                es = self.g11.encode(svhn, svhn=True)
                fake_es = self.g11.decode(es, svhn=True)
                g_loss = torch.mean((svhn - fake_es) ** 2)
                g_loss += self.kl_lambda*self._compute_kl(es)

                g_loss.backward()
                self.g_optimizer.step()

            # print the log info
            if (step+1) % self.log_step == 0:
                print('Step [%d/%d], d_real_loss: %.4f, d_mnist_loss: %.4f, d_svhn_loss: %.4f, '
                      'd_fake_loss: %.4f, g_loss: %.4f'
                      %(step+1, self.train_iters, d_real_loss.data[0], d_mnist_loss.data[0],
                        d_svhn_loss.data[0], d_fake_loss.data[0], g_loss.data[0]))

            # save the sampled images
            if (step+1) % self.sample_step == 0:
                em = self.g11.encode(fixed_mnist)
                fake_svhn_var = self.g11.decode(em, svhn=True)
                fake_svhn = self.to_data(fake_svhn_var)
                if self.config.save_models_and_samples:
                    merged = self.merge_images(mnist_fixed_data, fake_svhn)
                    path = os.path.join(self.sample_path, 'sample-%d-s-m.png' % (step + 1))
                    scipy.misc.imsave(path, merged)
                    print('saved %s' % path)

            if (step+1) % self.config.num_iters_save_model_and_return == 0:
                # save the model parameters for each epoch
                if self.config.save_models_and_samples:
                    g11_path = os.path.join(self.model_path, 'g11-%d.pkl' %(step+1))
                    d1_path = os.path.join(self.model_path, 'd1-%d.pkl' %(step+1))
                    d2_path = os.path.join(self.model_path, 'd2-%d.pkl' % (step + 1))
                    torch.save(self.g11.state_dict(), g11_path)
                    torch.save(self.d1.state_dict(), d1_path)
                    torch.save(self.d2.state_dict(), d2_path)

                return
