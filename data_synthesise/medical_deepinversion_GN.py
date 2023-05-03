# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Official PyTorch implementation of CVPR2020 paper
# Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion
# Hongxu Yin, Pavlo Molchanov, Zhizhong Li, Jose M. Alvarez, Arun Mallya, Derek
# Hoiem, Niraj K. Jha, and Jan Kautz
# --------------------------------------------------------

from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import torch.optim as optim
import collections
import torch.cuda.amp as amp
import torchvision.utils as vutils
from PIL import Image

from utils.utils import lr_cosine_policy, clip, create_folder
import wandb
import matplotlib.pyplot as plt

from models.layers.continual_normalization.cn import *
import random
from torchvision import transforms


def check_training(module):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == CN4 or type(target_attr) == CN8 or type(target_attr) == CN16:
            print(target_attr.training)
    #             address = './sweep_checkpoint_new/best_models/new_saved_values/'

    #             np.save(
    #                 address+"mean_"+str(target_attr.layer_number)+"_"+str(target_attr.num_epoch),
    #                 target_attr.group_running_mean.clone().detach().cpu().numpy())
    #             np.save(
    #                 address+"varience_"+str(target_attr.layer_number)+"_"+str(target_attr.num_epoch),
    #                 target_attr.group_running_var.clone().detach().cpu().numpy())
    for name, icm in module.named_children():
        if type(icm) == CN4 or type(icm) == CN8 or type(icm) == CN16:
            print(icm.training)
        check_training(icm)
    return


class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        # print("check_input_shape_2",len(input),input[0].shape)
        nch = module.out_gn.shape[1]
        mean = module.out_gn.mean([0, 2, 3])
        # print("mean",mean.shape,module.running_mean.data.shape)
        var = module.out_gn.permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.group_running_var.data - module.total_var.data, 2) + torch.norm(
            module.group_running_mean.data - module.total_mean.data, 2) + torch.norm(module.running_var.data - var,
                                                                                     2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2


class DeepInversionClass(object):
    def __init__(self, bs=84,
                 use_fp16=True, net_teacher=None, path="./gen_images/",
                 final_data_path="/gen_images_final/",
                 parameters=dict(),
                 setting_id=0,
                 jitter=30,
                 criterion=None,
                 coefficients=dict(),
                 network_output_function=lambda x: x,
                 hook_for_display=None,
                 hook_for_self_eval=None,
                 device=None,
                 target_classes_min=0,
                 target_classes_max=0,
                 mean_image_dir="./saved_Sample",
                 order_mine=None,
                 cm=None,
                 alpha=None,
                 gamma=None,
                 skin=False,
                 medmnist='None',
                 look_back = False,
                 synthesis= True):
        '''
        :param bs: batch size per GPU for image generation
        :param use_fp16: use FP16 (or APEX AMP) for model inversion, uses less memory and is faster for GPUs with Tensor Cores
        :parameter net_teacher: Pytorch model to be inverted
        :param path: path where to write temporal images and data
        :param final_data_path: path to write final images into
        :param parameters: a dictionary of control parameters:
            "resolution": input image resolution, single value, assumed to be a square, 224
            "random_label" : for classification initialize target to be random values
            "start_noise" : start from noise, def True, other options are not supported at this time
            "detach_student": if computing Adaptive DI, should we detach student?
        :param setting_id: predefined settings for optimization:
            0 - will run low resolution optimization for 1k and then full resolution for 1k;
            1 - will run optimization on high resolution for 2k
            2 - will run optimization on high resolution for 20k

        :param jitter: amount of random shift applied to image at every iteration
        :param coefficients: dictionary with parameters and coefficients for optimization.
            keys:
            "r_feature" - coefficient for feature distribution regularization
            "tv_l1" - coefficient for total variation L1 loss
            "tv_l2" - coefficient for total variation L2 loss
            "l2" - l2 penalization weight
            "lr" - learning rate for optimization
            "main_loss_multiplier" - coefficient for the main loss optimization
            "adi_scale" - coefficient for Adaptive DeepInversion, competition, def =0 means no competition
        network_output_function: function to be applied to the output of the network to get the output
        hook_for_display: function to be executed at every print/save call, useful to check accuracy of verifier
        '''

        print("Deep inversion class generation")
        # for reproducibility
        torch.manual_seed(torch.cuda.current_device())

        self.net_teacher = net_teacher

        if "resolution" in parameters.keys():
            self.image_resolution = parameters["resolution"]
            self.random_label = parameters["random_label"]
            self.start_noise = parameters["start_noise"]
            self.detach_student = parameters["detach_student"]
            self.do_flip = parameters["do_flip"]
            self.store_best_images = parameters["store_best_images"]
        else:
            self.image_resolution = 224
            self.random_label = False
            self.start_noise = True
            self.detach_student = False
            self.do_flip = True
            self.store_best_images = False

        self.setting_id = setting_id
        self.bs = bs  # batch size
        self.use_fp16 = use_fp16
        self.save_every = 4000
        self.jitter = jitter
        self.criterion = criterion
        self.network_output_function = network_output_function
        do_clip = True

        if "r_feature" in coefficients:
            self.bn_reg_scale = coefficients["r_feature"]
            self.first_bn_multiplier = coefficients["first_bn_multiplier"]
            self.var_scale_l1 = coefficients["tv_l1"]
            self.var_scale_l2 = coefficients["tv_l2"]
            self.l2_scale = coefficients["l2"]
            self.lr = coefficients["lr"]
            self.main_loss_multiplier = coefficients["main_loss_multiplier"]
            self.adi_scale = coefficients["adi_scale"]
        else:
            print("Provide a dictionary with ")

        self.num_generations = 0
        self.final_data_path = final_data_path

        ## Create folders for images and logs
        prefix = path
        self.prefix = prefix

        local_rank = torch.cuda.current_device()
        if local_rank == 0:
            create_folder(prefix)
            create_folder(prefix + "/best_images/")
            if self.store_best_images:
                create_folder(self.final_data_path)
            # save images to folders
            # for m in range(1000):
            #     create_folder(self.final_data_path + "/s{:03d}".format(m))
        self.log_file = open(prefix + "/log_file.csv", "w+")
        self.log_file.write(
            'iteration,\
            total loss,\
            loss batch normalization,\
            loss variation_l2,\
            loss variation_l1,\
            loss l2 on images,\
            Cross Entropy,\
            Verifier Acc,\
            learning rate\n')
        self.base_iteration = 0
        ## Create hooks for feature statistics
        self.loss_r_feature_layers = []

        for module in self.net_teacher.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, CN4) or isinstance(module, CN8) or isinstance(
                    module, CN16):
                self.loss_r_feature_layers.append(DeepInversionFeatureHook(module))

        self.hook_for_display = None
        if hook_for_display is not None:
            self.hook_for_display = hook_for_display

        self.hook_for_self_eval = None
        # if hook_for_self_eval is not None:
        #     self.hook_for_self_eval = hook_for_self_eval

        self.device = device
        self.target_classes_min = target_classes_min
        self.target_classes_max = target_classes_max
        self.mean_image_dir = mean_image_dir

        self.order_mine = order_mine

        self.cm = cm
        self.alpha = alpha
        self.gamma = gamma
        self.skin = skin
        self.medmnist = medmnist
        self.look_back = look_back
        self.synthesis = synthesis

    def get_images(self, net_student=None, targets=None, use_mean_initialization=False, beta_2=0.9):
        print("get_images call")

        net_teacher = self.net_teacher

        # for module in net_teacher.modules():
        #     print("module.training:",module.training)

        use_fp16 = self.use_fp16
        save_every = self.save_every

        kl_loss = nn.KLDivLoss(reduction='batchmean').to(self.device)
        local_rank = torch.cuda.current_device()
        best_cost = 1e4
        criterion = self.criterion

        # setup target labels
        if targets is None:
            # only works for classification now, for other tasks need to provide target vector
            targets = torch.LongTensor(
                [random.randint(self.target_classes_min, self.target_classes_max) for _ in range(self.bs)]).to(
                self.device)
            if not self.random_label:
                # preselected classes, good for ResNet50v1.5
                # print("pppp",self.target_classes_min,self.target_classes_max+1)
                targets = [i for i in np.arange(self.target_classes_min, self.target_classes_max + 1)]

                targets = torch.LongTensor(targets * (int(self.bs / len(targets)) + 1))[0:self.bs].to(self.device)
        print(targets)
        if self.look_back:
            print(self.cm)
            # print(targets,self.cm[0]/np.sum(self.cm[0]),self.cm[1]/np.sum(self.cm[1]),self.cm[2]/np.sum(self.cm[2]))
            targets_prob = torch.zeros((self.bs, len(self.cm))).to(self.device)
            for i, t in enumerate(targets):
                dirichlet = np.random.dirichlet(self.alpha, size=None)
                while np.sum(np.abs(dirichlet - self.cm[t] / np.sum(self.cm[t]))) > self.gamma:
                    dirichlet = np.random.dirichlet(self.alpha, size=None)
                # print(t, dirichlet,np.sum(np.abs(dirichlet - self.cm[t]/np.sum(self.cm[t]))))
                # print("dirichlet: ",dirichlet,self.cm[t]/np.sum(self.cm[t]),np.sum(np.power(dirichlet - self.cm[t]/np.sum(self.cm[t]),2)))
                targets_prob[i] = torch.tensor(dirichlet, device=self.device).float()

        img_original = self.image_resolution

        data_type = torch.half if use_fp16 else torch.float
        inputs_list = []
        # variance = random.randint(1,12)
        variance = 1
        if self.skin:
            inputs_layer = torch.from_numpy(np.random.normal(0, variance
                                                             , (self.bs, 3, img_original, img_original))).type(
                torch.FloatTensor).to(self.device)
        elif self.medmnist == 'BloodMnist' or self.medmnist == 'PathMnist':
            inputs_layer = torch.from_numpy(np.random.normal(0, variance
                                                             , (self.bs, 3, img_original, img_original))).type(
                torch.FloatTensor).to(self.device)
        elif self.medmnist == 'TissueMnist' or self.medmnist == 'OrganAMnist':
            inputs_layer = torch.from_numpy(np.random.normal(0, variance
                                                             , (self.bs, 1, img_original, img_original))).type(
                torch.FloatTensor).to(self.device)
        else:
            inputs_layer = torch.from_numpy(np.random.normal(0, variance
                                                             , (self.bs, 1, img_original, img_original))).type(
                torch.FloatTensor).to(self.device)
        inputs_layer.requires_grad = False
        if self.skin:
            mean = np.array([147.41485128, 113.04089631, 104.00467844]) / 255.0
            std = np.sqrt([5431.84436186, 3904.17606529, 3638.48287211]) / 255.0

        elif self.medmnist != 'None':
            mean = [0, 0, 0]
            std = [1, 1, 1]
        else:
            mean = [0.122, 0.122, 0.122]
            std = [0.184, 0.184, 0.184]
        if use_mean_initialization:
            for t in range(len(targets)):
                initialized_image_dir = self.mean_image_dir + "/label_" + str(
                    self.order_mine[targets[t].item()]) + "_integrated.png"
                image = Image.open(initialized_image_dir)
                # print("readed image max and min",
                #       np.array(image).max(),
                #       np.array(image).min(),
                #       np.array(image).shape)
                convert_tensor = transforms.ToTensor()
                image_array = convert_tensor(np.divide(((np.array(image) / 255.0) - mean), std)).to(self.device)
                # print("normalized image max and min",
                #       image_array.max(),
                #       image_array.min(),
                #       image_array.shape,
                #       torch.reshape(image_array,(3,224,224)).shape)
                # inputs_layer[t] = inputs_layer[t]/10 + torch.reshape(image_array[:,:,0],(1,224,224))
                if self.synthesis:
                    if self.skin:
                        inputs_layer[t] = inputs_layer[t] / 10 + torch.reshape(image_array, (
                        3, self.image_resolution, self.image_resolution))
                    elif self.medmnist == 'BloodMnist' or self.medmnist == 'PathMnist':
                        # print(image_array.shape)
                        # p2d = (1, 1, 1, 1) # pad last dim by (1, 1) and 2nd to last by (2, 2)
                        # image_array = F.pad(torch.reshape(image_array,(3,self.image_resolution,self.image_resolution)), p2d, "constant", 0)
                        # print(image_array.shape)
                        inputs_layer[t] = inputs_layer[t] / 10 + torch.reshape(image_array, (
                        3, self.image_resolution, self.image_resolution))
                    elif self.medmnist == 'TissueMnist' or self.medmnist == 'OrganAMnist':
                        inputs_layer[t] = inputs_layer[t] / 10 + torch.reshape(image_array[0, :, :], (
                        1, self.image_resolution, self.image_resolution))
                    else:

                        inputs_layer[t] = inputs_layer[t] / 10 + torch.reshape(image_array[:, :, 0], (
                        1, self.image_resolution, self.image_resolution))
                else:
                    if self.skin:
                        inputs_layer[t] = torch.reshape(image_array, (
                            3, self.image_resolution, self.image_resolution))
                    elif self.medmnist == 'BloodMnist' or self.medmnist == 'PathMnist':
                        # print(image_array.shape)
                        # p2d = (1, 1, 1, 1) # pad last dim by (1, 1) and 2nd to last by (2, 2)
                        # image_array = F.pad(torch.reshape(image_array,(3,self.image_resolution,self.image_resolution)), p2d, "constant", 0)
                        # print(image_array.shape)
                        inputs_layer[t] = torch.reshape(image_array, (
                            3, self.image_resolution, self.image_resolution))
                    elif self.medmnist == 'TissueMnist' or self.medmnist == 'OrganAMnist':
                        inputs_layer[t] =  torch.reshape(image_array[0, :, :], (
                            1, self.image_resolution, self.image_resolution))
                    else:

                        inputs_layer[t] = torch.reshape(image_array[:, :, 0], (
                            1, self.image_resolution, self.image_resolution))
        inputs_layer.requires_grad = True
        # print("min(inputs_layer),max(inputs_layer)",torch.min(inputs_layer),torch.max(inputs_layer))
        print("saving image dir", self.prefix)
        vutils.save_image(inputs_layer, '{}/best_images/output_{:05d}_gpu_{}_first.png'.format(self.prefix,
                                                                                               (
                                                                                                   self.base_iteration) // save_every,
                                                                                               local_rank),
                          normalize=True, scale_each=True, nrow=int(10))
        plt.style.use('dark_background')
        image = plt.imread('{}/best_images/output_{:05d}_gpu_{}_first.png'.format(self.prefix,
                                                                                  (self.base_iteration) // save_every,
                                                                                  local_rank))
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis('off')
        fig.set_size_inches(10 * 3, int((len(inputs_layer) + 1) / 10) * 3 + 2)
        plt.title("variance = " + str(variance) + "\n" + str(targets), fontweight="bold")
        plt.savefig('{}/best_images/output_{:05d}_gpu_{}_first.png'.format(self.prefix,
                                                                           (self.base_iteration) // save_every,
                                                                           local_rank))

        # for i in range(3):
        #     inputs_list.append(inputs_layer)
        # inputs = torch.cat(inputs_list, dim=1).to('cuda')
        # inputs.requires_grad = True
        inputs = inputs_layer
        pooling_function = nn.modules.pooling.AvgPool2d(kernel_size=2)

        if self.setting_id == 0:
            skipfirst = False
        else:
            skipfirst = True
        print(self.setting_id)
        iteration = 0
        if self.synthesis:
            for lr_it, lower_res in enumerate([2, 1]):
                print(lr_it, lower_res)
                if lr_it == 0:
                    iterations_per_layer = 3000
                else:
                    iterations_per_layer = 1000 if not skipfirst else 5000
                    if self.setting_id == 2:
                        iterations_per_layer = 20000

                if lr_it == 0 and skipfirst:
                    # print("I'm here")
                    continue

                # print(self.jitter)
                lim_0, lim_1 = self.jitter // lower_res, self.jitter // lower_res
                # print("lim_0, lim_1",lim_0, lim_1)

                if self.setting_id == 0:
                    # multi resolution, 2k iterations with low resolution, 1k at normal, ResNet50v1.5 works the best, ResNet50 is ok
                    optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, beta_2], eps=1e-8)
                    do_clip = True
                elif self.setting_id == 1:
                    # 2k normal resolultion, for ResNet50v1.5; Resnet50 works as well
                    optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, beta_2], eps=1e-8)
                    do_clip = True
                    # print("self.lr",self.lr)
                elif self.setting_id == 2:
                    # 20k normal resolution the closes to the paper experiments for ResNet50
                    optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, beta_2], eps=1e-8)
                    do_clip = True

                if use_fp16:
                    static_loss_scale = 256
                    static_loss_scale = "dynamic"
                    _, optimizer = amp.initialize([], optimizer, opt_level="O2", loss_scale=static_loss_scale)

                lr_scheduler = lr_cosine_policy(self.lr, 100, iterations_per_layer)
                # print(iterations_per_layer)
                print("hey I'm checking training")
                # check_training(net_teacher)
                for iteration_loc in range(iterations_per_layer):

                    iteration += 1
                    # learning rate scheduling
                    lr = lr_scheduler(optimizer, iteration_loc, iteration_loc)

                    # perform downsampling if needed
                    if lower_res != 1:
                        # inputs_jit = pooling_function(inputs)
                        inputs_jit = inputs
                        # print(inputs_jit.shape)
                    else:
                        inputs_jit = inputs

                    # apply random jitter offsets
                    off1 = random.randint(-lim_0, lim_0)
                    off2 = random.randint(-lim_1, lim_1)
                    inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

                    # Flipping
                    flip = random.random() > 0.5
                    if flip and self.do_flip:
                        inputs_jit = torch.flip(inputs_jit, dims=(3,))

                    # forward pass
                    optimizer.zero_grad()
                    net_teacher.zero_grad()
                    # print(net_teacher)
                    outputs = net_teacher(inputs_jit)
                    # if iteration_loc == 0:
                    #     print("predicted",outputs)
                    outputs = self.network_output_function(outputs)

                    # R_cross classification loss
                    if self.look_back:
                        loss = kl_loss(outputs, targets_prob)
                    else:
                        loss = criterion(outputs, targets)
                    # if iteration_loc == 0:
                    #     print("cross entropy loss",loss.item())

                    # R_prior losses
                    loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)
                    # if iteration_loc == 0:
                    #     print("loss variation_l1,",loss_var_l1)
                    #     print("loss variation_l2",loss_var_l2)
                    # R_feature loss
                    rescale = [self.first_bn_multiplier] + [1. for _ in range(len(self.loss_r_feature_layers) - 1)]
                    # if iteration_loc == 0:
                    #     print("rescale",rescale)
                    loss_r_feature = sum(
                        [mod.r_feature * rescale[idx] for (idx, mod) in enumerate(self.loss_r_feature_layers)])
                    # if iteration_loc == 0:
                    #     print("loss batch normalization",loss_r_feature)
                    # R_ADI
                    loss_verifier_cig = torch.zeros(1)
                    if self.adi_scale != 0.0:
                        if self.detach_student:
                            outputs_student = net_student(inputs_jit).detach()
                        else:
                            outputs_student = net_student(inputs_jit)

                        T = 3.0
                        if 1:
                            T = 3.0
                            # Jensen Shanon divergence:
                            # another way to force KL between negative probabilities
                            P = nn.functional.softmax(outputs_student / T, dim=1)
                            Q = nn.functional.softmax(outputs / T, dim=1)
                            M = 0.5 * (P + Q)

                            P = torch.clamp(P, 0.01, 0.99)
                            Q = torch.clamp(Q, 0.01, 0.99)
                            M = torch.clamp(M, 0.01, 0.99)
                            eps = 0.0
                            loss_verifier_cig = 0.5 * kl_loss(torch.log(P + eps), M) + 0.5 * kl_loss(torch.log(Q + eps), M)
                            # JS criteria - 0 means full correlation, 1 - means completely different
                            loss_verifier_cig = 1.0 - torch.clamp(loss_verifier_cig, 0.0, 1.0)

                        if local_rank == 0:
                            if iteration % save_every == 0:
                                print('loss_verifier_cig', loss_verifier_cig.item())

                    # l2 loss on images
                    loss_l2 = torch.norm(inputs_jit.view(self.bs, -1), dim=1).mean()
                    if iteration_loc == 0:
                        print("loss l2 on images", loss_l2)
                    # combining losses
                    loss_aux = self.var_scale_l2 * loss_var_l2 + \
                               self.var_scale_l1 * loss_var_l1 + \
                               self.bn_reg_scale * loss_r_feature + \
                               self.l2_scale * loss_l2
                    # print("Scaled loss variation_l1,",self.var_scale_l1 * loss_var_l1.item(), self.var_scale_l1,loss_var_l1.item())
                    # print("Scaled loss variation_l2",self.var_scale_l2 * loss_var_l2.item(),self.var_scale_l2,loss_var_l2.item())
                    # print("Scaled loss batch normalization",self.bn_reg_scale * loss_r_feature.item(), self.bn_reg_scale,loss_r_feature.item())
                    # print("Scaled loss l2 on images",self.l2_scale * loss_l2.item(),self.l2_scale,loss_l2.item())

                    if self.adi_scale != 0.0:
                        loss_aux += self.adi_scale * loss_verifier_cig

                    loss = self.main_loss_multiplier * loss + loss_aux
                    # print("Scaled cross entropy",self.main_loss_multiplier * loss,self.main_loss_multiplier)

                    if local_rank == 0:
                        ce = criterion(outputs, targets).item()
                        if iteration % save_every == 0:
                            print("------------iteration {}----------".format(iteration))
                            print("total loss", loss.item())
                            print("loss_r_feature", loss_r_feature.item())
                            print("main criterion", ce)

                            if self.hook_for_display is not None:
                                acc = self.hook_for_display(inputs, targets)
                            else:
                                acc = 0

                            if self.hook_for_self_eval is not None:
                                acc_self,_ = self.hook_for_self_eval(inputs, targets)
                            else:
                                acc_self = 0
                            self.log_file.write('{},{},{},{},{},{},{},{},{}\n'.format(
                                iteration + self.base_iteration,
                                loss.item(),
                                self.bn_reg_scale * loss_r_feature.item(),
                                self.var_scale_l2 * loss_var_l2.item(),
                                self.var_scale_l1 * loss_var_l1.item(),
                                self.l2_scale * loss_l2.item(),
                                self.main_loss_multiplier * ce,
                                acc,
                                lr))
                            self.log_file.flush()

                            metrics = {"total loss": loss.item(),
                                       "loss batch normalization": self.bn_reg_scale * loss_r_feature.item(),
                                       "batch normalization value": loss_r_feature.item(),
                                       "loss variation_l2": self.var_scale_l2 * loss_var_l2.item(),
                                       "loss l2 on images": self.l2_scale * loss_l2.item(),
                                       "Cross Entropy": self.main_loss_multiplier * ce,
                                       "Verifier Acc": acc,
                                       "Self Acc": acc_self,
                                       "learning rate": lr}
                            wandb.log(metrics)

                    # do image update
                    if use_fp16:
                        # optimizer.backward(loss)
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    optimizer.step()

                    # clip color outlayers
                    if do_clip:
                        inputs.data = clip(inputs.data, use_fp16=use_fp16)

                    if best_cost > loss.item() or iteration == 1:
                        best_inputs = inputs.data.clone()
                        best_cost = loss.item()

                    if iteration % save_every == 0 and (save_every > 0):
                        if local_rank == 0:
                            vutils.save_image(inputs,
                                              '{}/best_images/output_{:05d}_gpu_{}.png'.format(self.prefix,
                                                                                               (
                                                                                                           self.base_iteration + iteration) // save_every,
                                                                                               local_rank),
                                              normalize=True, scale_each=True, nrow=int(10))
                            plt.style.use('dark_background')
                            image = plt.imread('{}/best_images/output_{:05d}_gpu_{}.png'.format(self.prefix,
                                                                                                (
                                                                                                            self.base_iteration + iteration) // save_every,
                                                                                                local_rank))
                            fig, ax = plt.subplots()
                            ax.imshow(image)
                            ax.axis('off')
                            fig.set_size_inches(10 * 3, int((len(inputs_layer) + 1) / 10) * 3 + 2)
                            plt.title(str(targets), fontweight="bold")
                            plt.savefig('{}/best_images/output_{:05d}_gpu_{}.png'.format(self.prefix,
                                                                                         (
                                                                                                     self.base_iteration + iteration) // save_every,
                                                                                         local_rank))
            
            optimizer.state = collections.defaultdict(dict)
            # acc_self,accepted_ones = self.hook_for_self_eval(inputs, targets)
            # print(accepted_ones)
            # best_inputs = best_inputs[accepted_ones]
            # targets = targets[accepted_ones]
            # print('len_targets',len(targets))

        else:
            best_inputs = inputs.data.clone()
        if self.store_best_images:
            vutils.save_image(best_inputs,
                              '{}/output_{:05d}.png'.format(self.final_data_path,
                                                            self.base_iteration),
                              normalize=True, scale_each=True, nrow=int(10))
            plt.style.use('dark_background')
            image = plt.imread('{}/output_{:05d}.png'.format(self.final_data_path,
                                                             (self.base_iteration),
                                                             local_rank))
            fig, ax = plt.subplots()
            ax.imshow(image)
            ax.axis('off')
            fig.set_size_inches(10 * 3, int((len(inputs_layer) + 1) / 10) * 3 + 2)
            plt.title(str(targets), fontweight="bold")
            plt.savefig('{}/output_{:05d}_gpu_{}.png'.format(self.final_data_path,
                                                             (self.base_iteration) // save_every,
                                                             local_rank))

        # to reduce memory consumption by states of the optimizer we deallocate memory
        self.base_iteration += iteration

        # come back
        # for hook in self.loss_r_feature_layers:
        #     hook.close()

        print("iteratiooooooooon ======================", iteration)
        return best_inputs, targets

    def save_images(self, images, targets):
        # method to store generated images locally
        local_rank = torch.cuda.current_device()
        for id in range(images.shape[0]):
            class_id = targets[id].item()
            if 0:
                # save into separate folders
                place_to_store = '{}/s{:03d}/img_{:05d}_id{:03d}_gpu_{}_2.jpg'.format(self.final_data_path, class_id,
                                                                                      self.num_generations, id,
                                                                                      local_rank)
            else:
                place_to_store = '{}/img_s{:03d}_{:05d}_id{:03d}_gpu_{}_2.jpg'.format(self.final_data_path, class_id,
                                                                                      self.num_generations, id,
                                                                                      local_rank)

            image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
            pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
            pil_image.save(place_to_store)

    def generate_batch(self, net_student=None, targets=None, use_mean_initialization=False, beta_2=0.9):
        # for ADI detach student and add put to eval mode
        # net_teacher = self.net_teacher

        use_fp16 = self.use_fp16

        # fix net_student
        if not (net_student is None):
            net_student = net_student.eval()

        if targets is not None:
            targets = torch.from_numpy(np.array(targets).squeeze()).to(self.device)
            if use_fp16:
                targets = targets.half()

        # check_training(self.net_teacher)
        self.net_teacher.eval()

        images, targets = self.get_images(net_student=net_student, targets=targets,
                                          use_mean_initialization=use_mean_initialization, beta_2=beta_2)

        self.num_generations += 1
        return images.cpu(), targets.cpu()
