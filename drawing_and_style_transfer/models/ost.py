import torch
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class OSTModel(BaseModel):
    def name(self):
        return 'OSTModel'

    def _compute_kl(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

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

        self.netEnc_a, self.netDec_a = networks.define_ED(opt.input_nc, opt.output_nc,
                                                          opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout,
                                                          opt.init_type, self.gpu_ids,
                                                          n_blocks_encoder=num_res_blocks_unshared,
                                                          n_blocks_decoder=num_res_blocks_unshared,
                                                          start=start_unshared,
                                                          end=num_unshared, n_downsampling=n_downsampling,
                                                          input_layer=True, output_layer=True,
                                                          start_dec=start_dec_unshared, end_dec=end_dec_unshared)

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
            self.netD_a = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_b = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

            if not opt.dont_load_pretrained_autoencoder:
                which_epoch = opt.which_epoch
                self.load_network(self.netEnc_b, 'Enc_b', which_epoch)
                self.load_network(self.netDec_b, 'Dec_b', which_epoch)
                self.load_network(self.netEnc_shared, 'Enc_shared', which_epoch)
                self.load_network(self.netDec_shared, 'Dec_shared', which_epoch)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netEnc_a, 'Enc_a', which_epoch)
            self.load_network(self.netDec_a, 'Dec_a', which_epoch)
            self.load_network(self.netEnc_b, 'Enc_b', which_epoch)
            self.load_network(self.netDec_b, 'Dec_b', which_epoch)
            self.load_network(self.netEnc_shared, 'Enc_shared', which_epoch)
            self.load_network(self.netDec_shared, 'Dec_shared', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_a, 'D_a', which_epoch)
                self.load_network(self.netD_b, 'D_b', which_epoch)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionIdt = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizer_Enc_a = torch.optim.Adam(self.netEnc_a.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Dec_a = torch.optim.Adam(self.netDec_a.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Enc_b = torch.optim.Adam(
                itertools.chain(self.netEnc_b.parameters(), self.netEnc_shared.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Dec_b = torch.optim.Adam(
                itertools.chain(self.netDec_b.parameters(), self.netDec_shared.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_a = torch.optim.Adam(self.netD_a.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_b = torch.optim.Adam(self.netD_b.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_Enc_a)
            self.optimizers.append(self.optimizer_Dec_a)
            self.optimizers.append(self.optimizer_Enc_b)
            self.optimizers.append(self.optimizer_Dec_b)
            self.optimizers.append(self.optimizer_D_a)
            self.optimizers.append(self.optimizer_D_b)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netEnc_a)
        networks.print_network(self.netDec_a)
        networks.print_network(self.netEnc_b)
        networks.print_network(self.netDec_b)
        networks.print_network(self.netEnc_shared)
        networks.print_network(self.netDec_shared)
        if self.isTrain:
            networks.print_network(self.netD_a)
            networks.print_network(self.netD_b)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        real_A = Variable(self.input_A, volatile=True)
        real_B = Variable(self.input_B, volatile=True)
        enc_a = self.netEnc_shared(self.netEnc_a(real_A))
        enc_b = self.netEnc_shared(self.netEnc_b(real_B))

        fake_AA = self.netDec_a(self.netDec_shared(enc_a))
        fake_AB = self.netDec_b(self.netDec_shared(enc_a))
        fake_BB = self.netDec_b(self.netDec_shared(enc_b))

        enc_ab = self.netEnc_shared(self.netEnc_b(fake_AB))
        fake_ABA = self.netDec_a(self.netDec_shared(enc_ab))

        self.fake_AA = fake_AA.data
        self.fake_AB = fake_AB.data
        self.fake_BB = fake_BB.data
        self.fake_ABA = fake_ABA.data

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
        fake_AB = self.fake_B_pool.query(self.fake_AB)
        loss_D_ab = self.backward_D_basic(self.netD_b, self.real_B, fake_AB)
        self.loss_D_ab = loss_D_ab.data[0]

        fake_BB = self.fake_B_pool.query(self.fake_BB)
        loss_D_bb = self.backward_D_basic(self.netD_b, self.real_B, fake_BB)
        self.loss_D_bb = loss_D_bb.data[0]

    def backward_G(self):
        # GAN loss D_A(G_A(A))
        enc_a = self.netEnc_shared(self.netEnc_a(self.real_A))
        enc_b = self.netEnc_shared(self.netEnc_b(self.real_B))
        fake_AA = self.netDec_a(self.netDec_shared(enc_a))
        fake_AB = self.netDec_b(self.netDec_shared(enc_a))
        fake_BB = self.netDec_b(self.netDec_shared(enc_b))
        enc_ab = self.netEnc_shared(self.netEnc_b(fake_AB))
        fake_ABA = self.netDec_a(self.netDec_shared(enc_ab))

        pred_fake_AB = self.netD_b(fake_AB)
        loss_Gan_AB = self.criterionGAN(pred_fake_AB, True)
        loss_idt_A = self.criterionIdt(fake_AA, self.real_A)
        loss_cycle_A = self.opt.lambda_A * self.criterionIdt(fake_ABA, self.real_A)
        loss_idt_B = self.criterionIdt(fake_BB, self.real_B)
        pred_fake_BB = self.netD_b(fake_BB)
        loss_Gan_BB = self.criterionGAN(pred_fake_BB, True)
        loss_kl_B = self.opt.kl_lambda * self._compute_kl(enc_b)

        # combined losses
        loss_G_B = loss_idt_B + loss_kl_B + loss_Gan_BB
        loss_G_A = loss_Gan_AB + loss_cycle_A + loss_idt_A

        self.fake_AA = fake_AA.data
        self.fake_BB = fake_BB.data
        self.fake_AB = fake_AB.data
        self.fake_ABA = fake_ABA.data
        self.loss_cycle_A = loss_cycle_A.data[0]
        self.loss_Gan_AB = loss_Gan_AB.data[0]
        self.loss_Gan_BB = loss_Gan_AB.data[0]
        self.loss_idt_A = loss_idt_A.data[0]
        self.loss_idt_B = loss_idt_B.data[0]
        self.loss_kl_B = loss_kl_B.data[0]

        return loss_G_A, loss_G_B

    def optimize_parameters(self):
        # forward
        self.forward()
        loss_G_A, loss_G_B = self.backward_G()

        # x loss updates
        self.optimizer_Enc_a.zero_grad()
        self.optimizer_Dec_a.zero_grad()
        loss_G_A.backward(retain_graph=True)
        self.optimizer_Enc_a.step()
        self.optimizer_Dec_a.step()

        # B loss updates
        self.optimizer_Enc_b.zero_grad()
        self.optimizer_Dec_b.zero_grad()
        loss_G_B.backward()
        self.optimizer_Enc_b.step()
        self.optimizer_Dec_b.step()

        # D
        self.optimizer_D_a.zero_grad()
        self.optimizer_D_b.zero_grad()
        self.backward_D()
        self.optimizer_D_a.step()
        self.optimizer_D_b.step()

    def get_current_errors(self):
        ret_errors = OrderedDict(
            [('D_ab', self.loss_D_ab), ('D_bb', self.loss_D_bb),
             ('G_AB', self.loss_Gan_AB), ('G_BB', self.loss_Gan_BB),
             ('Idt_B', self.loss_idt_B), ('Idt_A', self.loss_idt_A),
             ('Cycle_A', self.loss_cycle_A), ('Kl_B', self.loss_kl_B), ])
        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_A)
        real_B = util.tensor2im(self.input_B)
        fake_BB = util.tensor2im(self.fake_BB)
        fake_AB = util.tensor2im(self.fake_AB)
        fake_AA = util.tensor2im(self.fake_AA)
        fake_ABA = util.tensor2im(self.fake_ABA)

        ret_visuals = OrderedDict(
            [('real_B', real_B), ('fake_BB', fake_BB),
             ('real_A', real_A), ('fake_AA', fake_AA), ('fake_AB', fake_AB), ('fake_ABA', fake_ABA), ])
        return ret_visuals

    def save(self, label):
        self.save_network(self.netEnc_a, 'Enc_a', label, self.gpu_ids)
        self.save_network(self.netDec_a, 'Dec_a', label, self.gpu_ids)
        self.save_network(self.netD_a, 'D_a', label, self.gpu_ids)
        self.save_network(self.netEnc_b, 'Enc_b', label, self.gpu_ids)
        self.save_network(self.netDec_b, 'Dec_b', label, self.gpu_ids)
        self.save_network(self.netD_b, 'D_b', label, self.gpu_ids)
        self.save_network(self.netEnc_shared, 'Enc_shared', label, self.gpu_ids)
        self.save_network(self.netDec_shared, 'Dec_shared', label, self.gpu_ids)
