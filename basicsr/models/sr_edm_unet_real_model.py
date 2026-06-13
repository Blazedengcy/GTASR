

import copy
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import random

import pyiqa
import torch.nn.functional as F
from basicsr.utils.img_process_util import filter2D
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.archs.edm_unet_arch import EDMUNet
from basicsr.archs import build_network
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.diffusion import improved_timesteps_schedule_decrease_linear, improved_timesteps_schedule_increase_linear, karras_schedule,karras_schedule_new, lognormal_timestep_distribution, q_sample
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils import generate_lq
from .base_model import BaseModel
import torch.distributed as dist
from torchvision.transforms.functional import normalize
import lpips
import numpy as np


@MODEL_REGISTRY.register()
class SREdmUNetRealModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SREdmUNetRealModel, self).__init__(opt)

        self.opt = opt
        self.train_phase = str(self.opt.get('train', {}).get('phase', 'secondphase')).lower()
        if self.train_phase not in ('firstphase', 'secondphase'):
            raise ValueError("train.phase must be 'firstphase' or 'secondphase'.")
        self.degradation = self.opt['degradation']
        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)

        self.lpips_dist = lpips.LPIPS(net='vgg').cuda()
        self.l1_loss = F.l1_loss
        self.l2_loss = F.mse_loss
        self.psnr = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr', device=self.device)
        self.ssim = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr', device=self.device)
        self.lpips = pyiqa.create_metric('lpips', device=self.device)
        self.clipiqa = pyiqa.create_metric('clipiqa', device=self.device)
        self.musiq = pyiqa.create_metric('musiq', device=self.device)
        self.niqe = pyiqa.create_metric('niqe', device=self.device)
        self.maniqa = pyiqa.create_metric('maniqa', device=self.device)
        self.dists = pyiqa.create_metric('dists', device=self.device)
        # self.fid = pyiqa.create_metric('fid', device=self.device)
        # self.ahiq = pyiqa.create_metric('ahiq', device=self.device)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()



        in_channels = 3 
        k_lap = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32)
        weight_lap = k_lap.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        
        # 3. 注册为 buffer (关键！使用 self.net_g, 基于上次的修复)
        # self.net_g 是 nn.Module, 而 self (EdmUNetRealModel) 不是
        try:
            self.net_g.register_buffer('weight_lap', weight_lap, persistent=True)
        except Exception:
            # 兜底：若 net_g 尚未是 nn.Module（异常情况），将其保存在实例上
            self.weight_lap = weight_lap

        # self.net_g = torch.compile(self.net_g) # torch2.0
    
    def prepare_data(self, data, dtype=torch.float32, realesrgan=True, phase='train'):
        # if realesrgan is None:
        #     realesrgan = self.configs.data.get(phase, dict).type == 'realesrgan'
        if realesrgan and phase == 'train':
            if not hasattr(self, 'jpeger'):
                self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
            if not hasattr(self, 'use_sharpener'):
                self.use_sharpener = USMSharp().cuda()

            im_gt = data['gt'].cuda()
            kernel1 = data['kernel1'].cuda()
            kernel2 = data['kernel2'].cuda()
            sinc_kernel = data['sinc_kernel'].cuda()

            ori_h, ori_w = im_gt.size()[2:4]
            if isinstance(self.degradation['sf'], int):
                sf = self.degradation['sf']
            else:
                assert len(self.degradation['sf']) == 2
                sf = random.uniform(*self.degradation['sf'])

            if self.degradation['use_sharp']:
                im_gt = self.use_sharpener(im_gt)

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(im_gt, kernel1)
            # random resize
            updown_type = random.choices(
                    ['up', 'down', 'keep'],
                    self.degradation['resize_prob'],
                    )[0]
            if updown_type == 'up':
                scale = random.uniform(1, self.degradation['resize_range'][1])
            elif updown_type == 'down':
                scale = random.uniform(self.degradation['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.degradation['gray_noise_prob']
            if random.random() < self.degradation['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.degradation['noise_range'],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                    )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.degradation['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.degradation['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            if random.random() < self.degradation['second_order_prob']:
                # blur
                if random.random() < self.degradation['second_blur_prob']:
                    out = filter2D(out, kernel2)
                # random resize
                updown_type = random.choices(
                        ['up', 'down', 'keep'],
                        self.degradation['resize_prob2'],
                        )[0]
                if updown_type == 'up':
                    scale = random.uniform(1, self.degradation['resize_range2'][1])
                elif updown_type == 'down':
                    scale = random.uniform(self.degradation['resize_range2'][0], 1)
                else:
                    scale = 1
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(int(ori_h / sf * scale), int(ori_w / sf * scale)),
                        mode=mode,
                        )
                # add noise
                gray_noise_prob = self.degradation['gray_noise_prob2']
                if random.random() < self.degradation['gaussian_noise_prob2']:
                    out = random_add_gaussian_noise_pt(
                        out,
                        sigma_range=self.degradation['noise_range2'],
                        clip=True,
                        rounds=False,
                        gray_prob=gray_noise_prob,
                        )
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.degradation['poisson_scale_range2'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False,
                        )

            
            if random.random() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(ori_h // sf, ori_w // sf),
                        mode=mode,
                        )
                out = filter2D(out, sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.degradation['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.degradation['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(ori_h // sf, ori_w // sf),
                        mode=mode,
                        )
                out = filter2D(out, sinc_kernel)
            
            # im_lq_ori = torch.clamp((out * 255.0).round(), 0, 255) / 255.
            
            # resize back
            if self.degradation['resize_back']:
                img_lq_ori = torch.clamp((out * 255.0).round(), 0, 255) / 255.
                out = F.interpolate(out, size=(ori_h, ori_w), mode='bicubic')
                img_lq_up = torch.clamp((out * 255.0).round(), 0, 255) / 255.
                # temp_sf = self.degradation['sf']
            else:
                img_lq_ori = torch.clamp((out * 255.0).round(), 0, 255) / 255.
                # temp_sf = self.degradation['sf']
                img_lq_up = None


            # random crop
            # gt_size = self.configs.degradation['gt_size']
            # im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, temp_sf)
            mean = data['mean'][0]
            std = data['std'][0]
            img_lq_ori = normalize(img_lq_ori, mean, std, inplace=True)
            img_lq_up = normalize(img_lq_up, mean, std, inplace=True)  # [0, 1] to [-1, 1]
            im_gt = normalize(im_gt, mean, std, inplace=True)  # [0, 1] to [-1, 1]

            return {'lq':img_lq_ori, 'gt':im_gt, 'lq_up': img_lq_up if img_lq_up is not None else None}
        else:
            lq = data['lq'].cuda()
            if 'gt' in data:
                return {'lq': lq, 'gt': data['gt']}
            else:
                return {'lq': lq}
            # if self.degradation['resize_back']:
            #     lq_up = F.interpolate(lq, scale_factor=self.degradation['sf'], mode='bicubic')
            #     return {'lq':lq, 'gt':data['gt'], 'lq_up':lq_up}
            # return {key:value.cuda().to(dtype=dtype) for key, value in data.items()}
    def init_training_settings(self):
        # self.net_g.train()
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()




        if train_opt.get('Stab_Rect_opt'):
            self.diff_opt = train_opt['Stab_Rect_opt']

            self.diff_model = build_network(self.opt['network_g'])
            self.diff_model = self.model_to_device(self.diff_model)
            self.diff_model.eval()

            for param in self.diff_model.parameters():
                param.requires_grad = False
            model_path = self.diff_opt.get('model_path', None)
            if model_path is not None:
                self.load_network(self.diff_model, model_path, True, 'params_ema')
            # for p_target, p_net in zip(self.diff_model.parameters(), self.net_g.parameters()):
            #     p_target.copy_(p_net.detach())      
        else:
            self.diff_opt = None
            self.diff_model = None


        if train_opt.get('consistency_opt'):
            self.consistency_opt = train_opt['consistency_opt']
            # loss_opt = {}
            # loss_opt['loss_weight'] = self.consistency_opt['loss_weight']

            self.net_target = build_network(self.opt['network_g']).to(self.device)
            self.net_target.eval()
            for param in self.net_target.parameters():
                param.requires_grad = False
            for p_target, p_net in zip(self.net_target.parameters(), self.net_g.parameters()):
                p_target.copy_(p_net.detach())        
        else:
            self.consistency_opt = None


        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)


        

    def feed_data(self, data):
        if 'phase' in data:
            phase = data['phase']
            data = self.prepare_data(data, phase=phase)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        else:
            self.gt = None
        self.lq = data['lq'].to(self.device)
        if 'lq_up' in data:
            self.lq_up = data['lq_up'].to(self.device)

    def optimize_parameters(self, current_iter,):        
        self.optimizer_g.zero_grad()

        if self.diff_opt:    
            # time_scale = int(self.diff_opt['time_scale'])
            pixel_weight = self.diff_opt['pixel_weight']
        else:
            # time_scale = 1
            pixel_weight = 1

        l_total = 0
        loss_dict = OrderedDict()
        # 用于后续 diff 分支（ours）依赖的一致性教师输出；若未启用一致性损失则保持为 None
        distiller = None
       
        # consistency loss
        if self.consistency_opt:
            for p_target, p_net in zip(self.net_target.parameters(), self.net_g.parameters()):
                p_target.copy_(p_net.detach())

            use_enc = self.opt['network_g']['use_enc']
            sigma_min = self.consistency_opt.get('sigma_min', 0)
            sigma_max = self.consistency_opt.get('sigma_max', 1)
            s0 = self.consistency_opt.get('s0')
            s1 = self.consistency_opt.get('s1')
            noise_rho = self.consistency_opt.get('noise_rho', 1)
            res_rho = self.consistency_opt.get('res_rho', 1)
            loss_opt = self.consistency_opt.get('loss_opt', None)
            power = self.consistency_opt.get('power', 1)
            res_min = 0
            res_max = 1
            total_iter = self.opt['train']['total_iter']
            if s0 > s1:
                num_steps = improved_timesteps_schedule_decrease_linear(current_iter, total_iter, s0, s1,)
            elif s0 < s1:
                num_steps = improved_timesteps_schedule_increase_linear(current_iter, total_iter, s0, s1,)
            else:
                num_steps = s0 + 1
            num_steps = int(num_steps)
            # 使用关键字参数，避免将 power 误传到 rho 位置，以及 noise_rho 误传到 device 位置
            sigmas = karras_schedule_new(
                num_steps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=noise_rho,
                time_divisor=num_steps-1,
                device=self.device,
                power=power,
            )
            alphas = karras_schedule_new(
                num_steps,
                sigma_min=res_min,
                sigma_max=res_max,
                rho=res_rho,
                time_divisor=num_steps-1,
                device=self.device,
                power=power,
            )
            

            timestep_high = num_steps - 1 if self.train_phase == 'firstphase' else min(3, num_steps - 1)
            timestep = torch.randint(0, timestep_high, size=(1,), device=self.device)

            
            
            
            index = timestep.repeat(self.gt.shape[0])
            next_index = index + 1
            

            noise = torch.randn_like(self.gt)

            

            x_cur = q_sample(self.lq_up, self.gt, sigmas, alphas, index, noise)
            x_next = q_sample(self.lq_up, self.gt, sigmas, alphas, next_index, noise)
            
            
            if use_enc:
                cond_lq = self.lq
            else:
                cond_lq = self.lq_up
            
            with torch.no_grad():                    
                distiller_target = self.net_target(x_cur, lq=cond_lq, sigma=sigmas[index]).detach()         
            distiller = self.net_g(x_next, lq=cond_lq, sigma=sigmas[next_index],)
            weight = pixel_weight
            self.output = distiller
            
            l_consistency = 0.0
            for name, l_opt in loss_opt.items():
                loss = self.get_loss_func(l_opt, distiller, distiller_target).mean()
                l_consistency += weight * loss

            loss_dict['l_consistency'] = l_consistency
            l_total += l_consistency



        if self.train_phase == 'firstphase':
            if distiller is None:
                raise RuntimeError('firstphase requires train.consistency_opt to produce distiller output.')

            l_TA = torch.zeros(1, device=self.device).squeeze()
            steps = torch.arange(1, num_steps - 1, device=self.device, dtype=torch.long)
            for t in steps:
                index_t = t.repeat(self.gt.shape[0])
                noise_t = torch.randn_like(self.gt)
                x_cur_t = q_sample(self.lq_up, self.gt, sigmas, alphas, index_t, noise_t)
                z_t = q_sample(self.lq_up, self.output, sigmas, alphas, index_t, noise_t)


                #TA Loss
                for name, l_opt in loss_opt.items():
                    loss = self.get_loss_func(l_opt, x_cur_t, z_t).mean()
                    l_TA += loss

            TA_weight = self.consistency_opt.get('TA_weight', 0.5)
            loss_dict['TA_weight'] = torch.tensor(TA_weight, device=self.device)
            l_TA =TA_weight * l_TA
            loss_dict['l_TA'] = l_TA
            l_total += l_TA

            l_total.backward()
            self.optimizer_g.step()
            self.log_dict = self.reduce_loss_dict(loss_dict)
            if self.ema_decay > 0:
                self.model_ema(decay=self.ema_decay)
            return

       
        if self.diff_model is not None:
            l_diff_1 = torch.zeros(1).to(self.device).squeeze()
            l_diff_2 = torch.zeros(1).to(self.device).squeeze()
            l_diff_3 = torch.zeros(1).to(self.device).squeeze()
            l_diff_4 = torch.zeros(1).to(self.device).squeeze()
            l_diff = torch.zeros(1).to(self.device).squeeze()
            time_scale = self.diff_opt['time_scale']
            diff_weight_1 = self.diff_opt['weight_1']
            diff_weight_2 = self.diff_opt['weight_2']
            num_steps = int(self.diff_opt['num_steps']) + 1
            cond_lq = self.diff_opt['cond_lq']
            use_enc = self.diff_opt['use_enc']
            weight_type = self.diff_opt['weight_type']
            t_min = self.diff_opt['t_min']
            t_max = self.diff_opt['t_max']
            self_iterative = self.diff_opt['self_iterative']
            loss_opt = self.diff_opt.get('loss_opt', None)
            loss_opt_sobel = self.diff_opt.get('loss_opt_sobel', None)
            loss_types = self.diff_opt.get('loss_type', 'dtm')
            if isinstance(loss_types, str):
                loss_types = [loss_types]

            if self_iterative and current_iter % time_scale == 0:
                self.diff_model = copy.deepcopy(self.net_g)
                for param in self.diff_model.parameters():
                    param.requires_grad = False
                self.diff_model.eval()

            step_min = max(1, int(t_min * num_steps))
            step_max = min(num_steps, int(t_max * num_steps))
            cond_lq = self.lq if use_enc else self.lq_up





            if 'Stab_Rect' in loss_types:
                timestep = torch.randint(step_min, step_max, (self.gt.shape[0],), device=self.device)
                index = timestep.reshape(self.gt.shape[0], 1, 1, 1)
                noise = torch.randn_like(self.output)
                x_t = q_sample(self.lq_up, self.gt, sigmas, alphas, index, noise)
                z_t = q_sample(self.lq_up, self.output.detach(), sigmas, alphas, index, noise)

                cond_lq = self.lq if use_enc else self.lq_up
                with torch.no_grad():
                    pred_x0_from_zt = self.diff_model(z_t, sigma=sigmas[index], lq=cond_lq).detach()
                    pred_x0_from_xt = self.diff_model(x_t, sigma=sigmas[index], lq=cond_lq).detach()

                if weight_type == 'simple':
                    weight = 1
                elif weight_type == 'improved':
                    weight = 1 / abs(self.output.detach() - self.gt).mean(dim=[1, 2, 3], keepdim=True)
                elif weight_type == 'lpips':
                    weight = 1 / (self.lpips(self.output.detach(), self.gt).view(-1, 1, 1, 1) + 1e-8)
                
                def sobel_edge_map(x):
                    B, C, H, W = x.shape
                    device = x.device
                    dtype = x.dtype
                    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=dtype, device=device)
                    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=dtype, device=device)
                    kx = kx.view(1, 1, 3, 3)
                    ky = ky.view(1, 1, 3, 3)
                    weight_x = kx.repeat(C, 1, 1, 1) 
                    weight_y = ky.repeat(C, 1, 1, 1)
                    grad_x = F.conv2d(x, weight_x, padding=1, groups=C)
                    grad_y = F.conv2d(x, weight_y, padding=1, groups=C)
                    edge = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-12)
                    return edge



                #以下的grad对应的是baseline CTMSR的DTM，grad_hf对应的是本文的Stab，l_hf_component对应的是本文的Rect

                pred_x0_from_zt_hf = sobel_edge_map(pred_x0_from_zt)
                pred_x0_from_xt_hf = sobel_edge_map(pred_x0_from_xt)
                
                grad = (pred_x0_from_zt - pred_x0_from_xt).detach() 
                grad_hf = (pred_x0_from_zt_hf - pred_x0_from_xt_hf).detach()
                grad_mix = grad + grad_hf*0.2  
                

                # (4. 同样, 替换 l_edge (sobel loss) 部分)
                with torch.no_grad(): # GT 的计算不需要梯度
                    hf_gt = sobel_edge_map(self.gt)
                    
                    
                
                
                hf_output = sobel_edge_map(self.output)

                l_high_freq_total_for_log = 0.0

                for name, l_opt in loss_opt_sobel.items():
                    l_hf_component = self.get_loss_func(l_opt, pred_x0_from_zt_hf, hf_gt.detach()) 
                    lambda_hf = 1.0  
                    l_hf_component_weighted = lambda_hf * l_hf_component
                    l_high_freq_total_for_log += l_hf_component_weighted.mean()
                
                # (重命名 loss_dict['l_edge'] -> 'l_high_freq')
                loss_dict['l_high_freq'] = l_high_freq_total_for_log
                l_total += l_high_freq_total_for_log # 加上高频损失


                
                # if timestep[0] == 1:
                for name, l_opt in loss_opt.items():
                    base = weight * self.get_loss_func(l_opt, self.output, (self.output.detach() - grad_mix))
                    loss = 0.5*1.6 * base
                    l_diff_2 += loss.mean()


                loss_dict['l_diff_dtm_Stab'] = l_diff_2

                l_total = l_diff_2 + l_total 






            l_total.backward()
            self.optimizer_g.step()
            self.log_dict = self.reduce_loss_dict(loss_dict)

            if self.ema_decay > 0:
                self.model_ema(decay=self.ema_decay)


        try:
            if not hasattr(self, '_ctmsr_weight_sum'):
                self._ctmsr_weight_sum = 0.0
                self._ctmsr_weight_count = 0
                self._ctmsr_weight_limit = 500   # 固定为 500
                self._ctmsr_weight_final_logged = False

            if self._ctmsr_weight_count < self._ctmsr_weight_limit:
                # 从归约后的日志中读取（仅当 ctmsr 分支写入了 weight 时才存在）
                w_mean_val = self.log_dict.get('weight', None)
                if w_mean_val is not None:
                    # 转成 float
                    try:
                        w_float = float(w_mean_val)
                    except Exception:
                        w_float = float(getattr(w_mean_val, 'item', lambda: w_mean_val)())

                    self._ctmsr_weight_sum += w_float
                    self._ctmsr_weight_count += 1
                    avg_ct = self._ctmsr_weight_sum / max(1, self._ctmsr_weight_count)

                    # 运行中的均值写入 self.log_dict，确保按 print_freq 打印
                    self.log_dict['weight_avg_run_ctmsr'] = avg_ct

                    # 第 500 次时记录最终均值，并在 rank0 打一条日志
                    if self._ctmsr_weight_count == self._ctmsr_weight_limit and not self._ctmsr_weight_final_logged:
                        self._ctmsr_weight_final_logged = True
                        self.log_dict['weight_avg_first_500_ctmsr'] = avg_ct
                        if self.opt.get('rank', 0) == 0:
                            logger = get_root_logger()
                            logger.info(f"CTMSR: Average of loss_dict['weight'] over first 500 iters: {avg_ct:.6f}")
        except Exception:
            pass
    def sum_gradients(self, model):
        total_grad = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_grad += torch.sum(torch.abs(param.grad)).item()
            else:
                print('none')
        return total_grad
    
    def sum_params(self, model):
        total_params = 0.0
        for param in model.parameters():
            total_params += param.sum().item()
        return total_params


    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')
        total_inference_time = 0.0
        num_imgs = 0
        for idx, val_data in enumerate(dataloader):
            num_imgs += 1
            if 'gt_path' in val_data:
                img_name = osp.splitext(osp.basename(val_data['gt_path'][0]))[0]
            else: 
                img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            val_data = self.prepare_data(val_data, phase='val')
            self.feed_data(val_data)
            self.test()
            total_inference_time += self.inference_time


            visuals = self.get_current_visuals()
            # print(torch.max(visuals['result']).item())
            sr_tensor = visuals['result']
            sr_img = tensor2img([visuals['result']])
            # print(torch.max(visuals['result']).item())

            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_tensor = visuals['gt']
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt
            

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                if 'gt' in visuals:
                    for name, opt_ in self.opt['val']['metrics'].items():
                        if name == 'lpips':
                            self.metric_results[name] += self.lpips(sr_tensor, gt_tensor).mean().item()
                        elif name == 'clipiqa':
                            self.metric_results[name] += self.clipiqa(sr_tensor).mean().item()
                        elif name == 'musiq':
                            self.metric_results[name] += self.musiq(sr_tensor).mean().item()
                        elif name == 'psnr':
                            self.metric_results[name] += self.psnr(sr_tensor, gt_tensor).mean().item()
                        elif name == 'ssim':
                            self.metric_results[name] += self.ssim(sr_tensor, gt_tensor).mean().item()
                        elif name == 'niqe':
                            self.metric_results[name] += self.niqe(sr_tensor).mean().item()
                        elif name == 'maniqa':
                            self.metric_results[name] += self.maniqa(sr_tensor).mean().item()
                        elif name == 'dists':
                            self.metric_results[name] += self.dists(sr_tensor, gt_tensor).mean().item()
                        elif name == 'fid':
                            self.metric_results[name] += self.fid(sr_tensor).mean().item()
                        elif name == 'ahiq':
                            self.metric_results[name] += self.ahiq(sr_tensor, gt_tensor).mean().item()
                    
                else:
                    for name, opt_ in self.opt['val']['metrics'].items():
                        if name == 'clipiqa':
                            self.metric_results[name] += self.clipiqa(sr_tensor).mean().item()
                        elif name == 'musiq':
                            self.metric_results[name] += self.musiq(sr_tensor).mean().item()
                        elif name == 'niqe':
                            self.metric_results[name] += self.niqe(sr_tensor).mean().item()
                        elif name == 'maniqa':
                            self.metric_results[name] += self.maniqa(sr_tensor).mean().item()
                    
                    # self.metric_results[name] += calculate_metric(metric_data, opt_)
                    # print(self.metric_results)    
                        
                    # self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        # print(num_imgs)
        avg_inference_time = total_inference_time / num_imgs
        log_str = f'inference time:{avg_inference_time:.6f}s\n'
        logger = get_root_logger()
        logger.info(log_str)
        # print(f'推理时间: {avg_inference_time:.6f} 秒')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        self.lq = self.totensor(self.lq)
        out_dict['lq'] = self.lq.detach().cpu()
        self.output = self.totensor(self.output)
        out_dict['result'] = self.output.detach().cpu()
        if self.gt is not None:
            self.gt = self.totensor(self.gt)
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    def totensor(self, input):
        # input = input.clamp(min=-1, max=1)
        input = (input + 1) / 2
        input = input.clamp(0, 1)
        return input
    
    def replace_nan_in_batch(im_lq, im_gt):
        '''
        Input:
            im_lq, im_gt: b x c x h x w
        '''
        if torch.isnan(im_lq).sum() > 0:
            valid_index = []
            im_lq = im_lq.contiguous()
            for ii in range(im_lq.shape[0]):
                if torch.isnan(im_lq[ii,]).sum() == 0:
                    valid_index.append(ii)
            assert len(valid_index) > 0
            im_lq, im_gt = im_lq[valid_index,], im_gt[valid_index,]
            flag = True
        else:
            flag = False
        return im_lq, im_gt, flag
    
    def charbonnier_loss(self, pred, target, eps=1e-12):
        return torch.sqrt((pred - target)**2 + eps)
    
    def get_loss_func(self, loss_opt, input1, input2):
        loss_type = loss_opt['type']
        weight = loss_opt['weight']
        if loss_type == 'l2':
            loss = self.l2_loss(input1, input2)
        elif loss_type == 'l1':
            loss = self.l1_loss(input1, input2)
        elif loss_type == 'lpips':
            loss = self.lpips_dist(input1, input2)
        elif loss_type == 'charbonnier':
            loss = self.charbonnier_loss(input1, input2)
        elif loss_type == 'dists':
            # pyiqa 的 DISTS 要求输入在 [0,1]；训练过程中的张量多为 [-1,1]，这里做转换
            # 注意：clamp 会在越界处截断梯度，若不希望截断可只做线性缩放 (x+1)/2 而不 clamp
            x1 = self.totensor(input1)
            x2 = self.totensor(input2)
            loss = self.dists(x1, x2)
        else:
            raise ValueError('Unsupported loss function.')

        return weight * loss
    
    def get_diff_loss(self, diff_loss_opt, ):
        loss_type = diff_loss_opt['loss_type']
        return

