import torch
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class AutoEncoderModel(BaseModel):
    def name(self):
        return 'AutoEncoderModel'

    def set_encoders_and_decoders(self, opt):
        n_downsampling = opt.n_downsampling
        start_unshared = 0
        num_unshared = opt.num_unshared
        start_shared = num_unshared
        end_shared = n_downsampling
        start_dec_shared = start_unshared
        end_dec_shared = start_unshared + (end_shared - start_shared)
        start_dec_unshared = end_dec_shared
        end_dec_unshared = n_downsampling

        num_res_blocks_unshared = opt.num_res_blocks_unshared
        n_res_blocks_shared = opt.num_res_blocks_shared

        self.netEnc_b, self.netDec_b = networks.define_ED(opt.input_nc, opt.output_nc,
                                                          opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout,
                                                          opt.init_type, self.gpu_ids,
                                                          n_blocks_encoder=num_res_blocks_unshared,
                                                          n_blocks_decoder=num_res_blocks_unshared,
                                                          start=start_unshared,
                                                          end=num_unshared, n_downsampling=n_downsampling,
                                                          input_layer=True,
                                                          output_layer=True, start_dec=start_dec_unshared,
                                                          end_dec=end_dec_unshared)

        self.netEnc_shared, self.netDec_shared = networks.define_ED(opt.input_nc, opt.output_nc,
                                                                    opt.ngf, opt.which_model_netG, opt.norm,
                                                                    not opt.no_dropout,
                                                                    opt.init_type, self.gpu_ids,
                                                                    n_blocks_encoder=n_res_blocks_shared,
                                                                    n_blocks_decoder=n_res_blocks_shared,
                                                                    start=start_shared, n_downsampling=n_downsampling,
                                                                    end=end_shared,
                                                                    input_layer=False,
                                                                    output_layer=False, start_dec=start_dec_shared,
                                                                    end_dec=end_dec_shared)

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.set_encoders_and_decoders(opt)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netEnc_b, 'Enc_b', which_epoch)
            self.load_network(self.netDec_b, 'Dec_b', which_epoch)
            self.load_network(self.netEnc_shared, 'Enc_shared', which_epoch)
            self.load_network(self.netDec_shared, 'Dec_shared', which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', which_epoch)

        if self.isTrain:
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionIdt = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizer_Enc = torch.optim.Adam(
                itertools.chain(self.netEnc_b.parameters(), self.netEnc_shared.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Dec = torch.optim.Adam(
                itertools.chain(self.netDec_b.parameters(), self.netDec_shared.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_Enc)
            self.optimizers.append(self.optimizer_Dec)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netEnc_b)
        networks.print_network(self.netDec_b)
        networks.print_network(self.netEnc_shared)
        networks.print_network(self.netDec_shared)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        # 'A' is given as single_dataset
        input_B = input['A']
        if len(self.gpu_ids) > 0:
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
        self.input_B = input_B
        # 'A' is given as single_dataset
        self.image_paths = input['A_paths']

    def forward(self):
        self.real_B = Variable(self.input_B)

    def netEnc(self, x):
        return self.netEnc_shared(self.netEnc_b(x))

    def netDec(self, x):
        return self.netDec_b(self.netDec_shared(x))

    def test(self):
        real_B = Variable(self.input_B, volatile=True)
        fake_B = self.netDec(self.netEnc(real_B))
        self.fake_B = fake_B.data

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        loss_D = self.backward_D_basic(self.netD, self.real_B, fake_B)
        self.loss_D = loss_D.data[0]

    def _compute_kl(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def backward_G(self):
        lambda_B = self.opt.lambda_B

        # GAN loss D_B(G_B(B))
        enc_b = self.netEnc(self.real_B)
        fake_B = self.netDec(enc_b)
        pred_fake = self.netD(fake_B)
        loss_Gan = self.criterionGAN(pred_fake, True)
        loss_idt_B = self.criterionIdt(fake_B, self.real_B) * lambda_B
        loss_kl_B = self.opt.kl_lambda * self._compute_kl(enc_b)

        # combined loss
        loss_G = loss_Gan + loss_idt_B + loss_kl_B
        loss_G.backward()

        self.fake_B = fake_B.data
        self.loss_Gan = loss_Gan.data[0]
        self.loss_idt_B = loss_idt_B.data[0]

    def optimize_parameters(self):
        # forward
        self.forward()

        # G
        self.optimizer_Enc.zero_grad()
        self.optimizer_Dec.zero_grad()
        self.backward_G()
        self.optimizer_Enc.step()
        self.optimizer_Dec.step()

        # D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('D', self.loss_D), ('G_B', self.loss_Gan), ('Idt_B', self.loss_idt_B)])
        return ret_errors

    def get_current_visuals(self):
        real_B = util.tensor2im(self.input_B)
        fake_B = util.tensor2im(self.fake_B)
        ret_visuals = OrderedDict([('real_B', real_B), ('fake_B', fake_B), ])
        return ret_visuals

    def save(self, label):
        self.save_network(self.netEnc_b, 'Enc_b', label, self.gpu_ids)
        self.save_network(self.netDec_b, 'Dec_b', label, self.gpu_ids)
        self.save_network(self.netEnc_shared, 'Enc_shared', label, self.gpu_ids)
        self.save_network(self.netDec_shared, 'Dec_shared', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
