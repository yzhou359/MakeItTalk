"""
 # Copyright 2020 Adobe
 # All Rights Reserved.
 
 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it.
 
"""

from src.models.model_image_translation import ResUnetGenerator, VGGLoss
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import time
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import glob
from src.dataset.image_translation.image_translation_dataset import vis_landmark_on_img, vis_landmark_on_img98, vis_landmark_on_img74


from thirdparty.AdaptiveWingLoss.core import models
from thirdparty.AdaptiveWingLoss.utils.utils import get_preds_fromhm

import face_alignment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Image_translation_block():

    def __init__(self, opt_parser, single_test=False):
        print('Run on device {}'.format(device))

        # for key in vars(opt_parser).keys():
        #     print(key, ':', vars(opt_parser)[key])
        self.opt_parser = opt_parser

        # model
        if(opt_parser.add_audio_in):
            self.G = ResUnetGenerator(
                input_nc=7, output_nc=3, num_downs=6, use_dropout=False)
        else:
            self.G = torch.load(opt_parser.load_G_name)
            #     self.G = ResUnetGenerator(
            #         input_nc=6, output_nc=3, num_downs=6, use_dropout=False)

            # if (opt_parser.load_G_name != ''):
            #     ckpt = torch.load(opt_parser.load_G_name)
            #     try:
            #         self.G.load_state_dict(ckpt['G'])
            #     except:
            #         tmp = nn.DataParallel(self.G)
            #         tmp.load_state_dict(ckpt['G'])
            #         self.G.load_state_dict(tmp.module.state_dict())
            #         del tmp

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs in G mode!")
            self.G = nn.DataParallel(self.G)

        self.G.to(device)

        if(not single_test):
            # dataset
            if(opt_parser.use_vox_dataset == 'raw'):
                if(opt_parser.comb_fan_awing):
                    from src.dataset.image_translation.image_translation_dataset import \
                        image_translation_raw74_dataset as image_translation_dataset
                elif(opt_parser.add_audio_in):
                    from src.dataset.image_translation.image_translation_dataset import image_translation_raw98_with_audio_dataset as \
                        image_translation_dataset
                else:
                    from src.dataset.image_translation.image_translation_dataset import image_translation_raw98_dataset as \
                        image_translation_dataset
            else:
                from src.dataset.image_translation.image_translation_dataset import image_translation_preprocessed98_dataset as \
                    image_translation_dataset

            self.dataset = image_translation_dataset(
                num_frames=opt_parser.num_frames)
            self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                          batch_size=opt_parser.batch_size,
                                                          shuffle=True,
                                                          num_workers=opt_parser.num_workers)

            # criterion
            self.criterionL1 = nn.L1Loss()
            self.criterionVGG = VGGLoss()
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(),
                      "GPUs in VGG model!")
                self.criterionVGG = nn.DataParallel(self.criterionVGG)
            self.criterionVGG.to(device)

            # optimizer
            self.optimizer = torch.optim.Adam(
                self.G.parameters(), lr=opt_parser.lr, betas=(0.5, 0.999))

            # writer
            if(opt_parser.write):
                self.writer = SummaryWriter(log_dir=os.path.join(
                    opt_parser.log_dir, opt_parser.name))
                self.count = 0

            # ===========================================================
            #       online landmark alignment : Awing
            # ===========================================================
            PRETRAINED_WEIGHTS = 'thirdparty/AdaptiveWingLoss/ckpt/WFLW_4HG.pth'
            GRAY_SCALE = False
            HG_BLOCKS = 4
            END_RELU = False
            NUM_LANDMARKS = 98

            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
            model_ft = models.FAN(HG_BLOCKS, END_RELU,
                                  GRAY_SCALE, NUM_LANDMARKS)

            checkpoint = torch.load(PRETRAINED_WEIGHTS)
            if 'state_dict' not in checkpoint:
                model_ft.load_state_dict(checkpoint)
            else:
                pretrained_weights = checkpoint['state_dict']
                model_weights = model_ft.state_dict()
                pretrained_weights = {k: v for k, v in pretrained_weights.items()
                                      if k in model_weights}
                model_weights.update(pretrained_weights)
                model_ft.load_state_dict(model_weights)
            print('Load AWing model sucessfully')
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs for AWing!")
                self.fa_model = nn.DataParallel(
                    model_ft).to(self.device).eval()
            else:
                self.fa_model = model_ft.to(self.device).eval()

            # ===========================================================
            #       online landmark alignment : FAN
            # ===========================================================
            if(opt_parser.comb_fan_awing):
                if(opt_parser.fan_2or3D == '2D'):
                    self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                                                  device='cuda' if torch.cuda.is_available() else "cpu",
                                                                  flip_input=True)
                else:
                    self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D,
                                                                  device='cuda' if torch.cuda.is_available() else "cpu",
                                                                  flip_input=True)

    def __train_pass__(self, epoch, is_training=True):
        st_epoch = time.time()
        if(is_training):
            self.G.train()
            status = 'TRAIN'
        else:
            self.G.eval()
            status = 'EVAL'

        g_time = 0.0
        for i, batch in enumerate(self.dataloader):
            if(i >= len(self.dataloader)-2):
                break
            st_batch = time.time()

            if(self.opt_parser.comb_fan_awing):
                image_in, image_out, fan_pred_landmarks = batch
                fan_pred_landmarks = fan_pred_landmarks.reshape(
                    -1, 68, 3).detach().cpu().numpy()
            elif(self.opt_parser.add_audio_in):
                image_in, image_out, audio_in = batch
                audio_in = audio_in.reshape(-1, 1, 256, 256).to(device)
            else:
                image_in, image_out = batch

            with torch.no_grad():
                # # online landmark (AwingNet)
                image_in, image_out = \
                    image_in.reshape(-1, 3, 256, 256).to(
                        device), image_out.reshape(-1, 3, 256, 256).to(device)
                inputs = image_out
                outputs, boundary_channels = self.fa_model(inputs)
                pred_heatmap = outputs[-1][:, :-1, :, :].detach().cpu()
                pred_landmarks, _ = get_preds_fromhm(pred_heatmap)
                pred_landmarks = pred_landmarks.numpy() * 4

                # online landmark (FAN) -> replace jaw + eye brow in AwingNet
                if(self.opt_parser.comb_fan_awing):
                    fl_jaw_eyebrow = fan_pred_landmarks[:, 0:27, 0:2]
                    fl_rest = pred_landmarks[:, 51:, :]
                    pred_landmarks = np.concatenate(
                        [fl_jaw_eyebrow, fl_rest], axis=1).astype(np.int)

            # draw landmark on while bg
            img_fls = []
            for pred_fl in pred_landmarks:
                img_fl = np.ones(shape=(256, 256, 3)) * 255.0
                if(self.opt_parser.comb_fan_awing):
                    img_fl = vis_landmark_on_img74(img_fl, pred_fl)  # 74x2
                else:
                    img_fl = vis_landmark_on_img98(img_fl, pred_fl)  # 98x2
                img_fls.append(img_fl.transpose((2, 0, 1)))
            img_fls = np.stack(img_fls, axis=0).astype(np.float32) / 255.0
            image_fls_in = torch.tensor(
                img_fls, requires_grad=False).to(device)
            if(self.opt_parser.add_audio_in):
                # print(image_fls_in.shape, image_in.shape, audio_in.shape)
                image_in = torch.cat([image_fls_in, image_in, audio_in], dim=1)
            else:
                image_in = torch.cat([image_fls_in, image_in], dim=1)

            # image_in, image_out = \
            #     image_in.reshape(-1, 6, 256, 256).to(device), image_out.reshape(-1, 3, 256, 256).to(device)

            # image2image net fp
            g_out = self.G(image_in)
            g_out = torch.tanh(g_out)

            loss_l1 = self.criterionL1(g_out, image_out)
            loss_vgg, loss_style = self.criterionVGG(
                g_out, image_out, style=True)

            loss_vgg, loss_style = torch.mean(loss_vgg), torch.mean(loss_style)

            loss = loss_l1 + loss_vgg + loss_style
            if(is_training):
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # log
            if(self.opt_parser.write):
                self.writer.add_scalar(
                    'loss', loss.cpu().detach().numpy(), self.count)
                self.writer.add_scalar(
                    'loss_l1', loss_l1.cpu().detach().numpy(), self.count)
                self.writer.add_scalar(
                    'loss_vgg', loss_vgg.cpu().detach().numpy(), self.count)
                self.count += 1

            # save image to track training process
            if (i % self.opt_parser.jpg_freq == 0):
                vis_in = np.concatenate([image_in[0, 3:6].cpu().detach().numpy().transpose((1, 2, 0)),
                                         image_in[0, 0:3].cpu().detach().numpy().transpose((1, 2, 0))], axis=1)
                vis_out = np.concatenate([image_out[0].cpu().detach().numpy().transpose((1, 2, 0)),
                                          g_out[0].cpu().detach().numpy().transpose((1, 2, 0))], axis=1)
                vis = np.concatenate([vis_in, vis_out], axis=0)
                try:
                    os.makedirs(os.path.join(
                        self.opt_parser.jpg_dir, self.opt_parser.name))
                except:
                    pass
                cv2.imwrite(os.path.join(self.opt_parser.jpg_dir, self.opt_parser.name,
                                         'e{:03d}_b{:04d}.jpg'.format(epoch, i)), vis * 255.0)
            # save ckpt
            if (i % self.opt_parser.ckpt_last_freq == 0):
                self.__save_model__('last', epoch)

            print("Epoch {}, Batch {}/{}, loss {:.4f}, l1 {:.4f}, vggloss {:.4f}, styleloss {:.4f} time {:.4f}".format(
                epoch, i, len(self.dataset) // self.opt_parser.batch_size,
                loss.cpu().detach().numpy(),
                loss_l1.cpu().detach().numpy(),
                loss_vgg.cpu().detach().numpy(),
                loss_style.cpu().detach().numpy(),
                time.time() - st_batch))

            g_time += time.time() - st_batch

            if(self.opt_parser.test_speed):
                if(i >= 100):
                    break

        print('Epoch time usage:', time.time() - st_epoch, 'I/O time usage:',
              time.time() - st_epoch - g_time, '\n=========================')
        if(self.opt_parser.test_speed):
            exit(0)
        if(epoch % self.opt_parser.ckpt_epoch_freq == 0):
            self.__save_model__('{:02d}'.format(epoch), epoch)

    def __save_model__(self, save_type, epoch):
        try:
            os.makedirs(os.path.join(
                self.opt_parser.ckpt_dir, self.opt_parser.name))
        except:
            pass
        if (self.opt_parser.write):
            torch.save(self.G, os.path.join(self.opt_parser.ckpt_dir,
                                            self.opt_parser.name, 'ckpt_{}.pth'.format(save_type)))
            # torch.save({
            #     'G': self.G.state_dict(),
            #     'opt': self.optimizer,
            #     'epoch': epoch
            # }, os.path.join(self.opt_parser.ckpt_dir, self.opt_parser.name, 'ckpt_{}.pth'.format(save_type)))

    def train(self):
        for epoch in range(self.opt_parser.nepoch):
            self.__train_pass__(epoch, is_training=True)

    def test(self):
        if (self.opt_parser.use_vox_dataset == 'raw'):
            if(self.opt_parser.add_audio_in):
                from src.dataset.image_translation.image_translation_dataset import \
                    image_translation_raw98_with_audio_test_dataset as image_translation_test_dataset
            else:
                from src.dataset.image_translation.image_translation_dataset import image_translation_raw98_test_dataset as image_translation_test_dataset
        else:
            from src.dataset.image_translation.image_translation_dataset import image_translation_preprocessed98_test_dataset as image_translation_test_dataset
        self.dataset = image_translation_test_dataset(
            num_frames=self.opt_parser.num_frames)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=1,
                                                      shuffle=True,
                                                      num_workers=self.opt_parser.num_workers)

        self.G.eval()
        for i, batch in enumerate(self.dataloader):
            print(i, 50)
            if (i > 50):
                break

            if (self.opt_parser.add_audio_in):
                image_in, image_out, audio_in = batch
                audio_in = audio_in.reshape(-1, 1, 256, 256).to(device)
            else:
                image_in, image_out = batch

            # # online landmark (AwingNet)
            with torch.no_grad():
                image_in, image_out = \
                    image_in.reshape(-1, 3, 256, 256).to(
                        device), image_out.reshape(-1, 3, 256, 256).to(device)

                pred_landmarks = []
                for j in range(image_in.shape[0] // 16):
                    inputs = image_out[j*16:j*16+16]
                    outputs, boundary_channels = self.fa_model(inputs)
                    pred_heatmap = outputs[-1][:, :-1, :, :].detach().cpu()
                    pred_landmark, _ = get_preds_fromhm(pred_heatmap)
                    pred_landmarks.append(pred_landmark.numpy() * 4)
                pred_landmarks = np.concatenate(pred_landmarks, axis=0)

            # draw landmark on while bg
            img_fls = []
            for pred_fl in pred_landmarks:
                img_fl = np.ones(shape=(256, 256, 3)) * 255.0
                img_fl = vis_landmark_on_img98(img_fl, pred_fl)  # 98x2
                img_fls.append(img_fl.transpose((2, 0, 1)))
            img_fls = np.stack(img_fls, axis=0).astype(np.float32) / 255.0
            image_fls_in = torch.tensor(
                img_fls, requires_grad=False).to(device)

            if (self.opt_parser.add_audio_in):
                # print(image_fls_in.shape, image_in.shape, audio_in.shape)
                image_in = torch.cat([image_fls_in,
                                      image_in[0:image_fls_in.shape[0]],
                                      audio_in[0:image_fls_in.shape[0]]], dim=1)
            else:
                image_in = torch.cat(
                    [image_fls_in, image_in[0:image_fls_in.shape[0]]], dim=1)

            # normal 68 test dataset
            # image_in, image_out = image_in.reshape(-1, 6, 256, 256), image_out.reshape(-1, 3, 256, 256)

            # random single frame
            # cv2.imwrite('random_img_{}.jpg'.format(i), np.swapaxes(image_out[5].numpy(),0, 2)*255.0)

            image_in, image_out = image_in.to(device), image_out.to(device)

            writer = cv2.VideoWriter('tmp_{:04d}.mp4'.format(
                i), cv2.VideoWriter_fourcc(*'mjpg'), 25, (256*4, 256))

            for j in range(image_in.shape[0] // 16):
                g_out = self.G(image_in[j*16:j*16+16])
                g_out = torch.tanh(g_out)

                # norm 68 pts
                # g_out = np.swapaxes(g_out.cpu().detach().numpy(), 1, 3)
                # ref_out = np.swapaxes(image_out[j*16:j*16+16].cpu().detach().numpy(), 1, 3)
                # ref_in = np.swapaxes(image_in[j*16:j*16+16, 3:6, :, :].cpu().detach().numpy(), 1, 3)
                # fls_in = np.swapaxes(image_in[j * 16:j * 16 + 16, 0:3, :, :].cpu().detach().numpy(), 1, 3)
                g_out = g_out.cpu().detach().numpy().transpose((0, 2, 3, 1))
                g_out[g_out < 0] = 0
                ref_out = image_out[j * 16:j * 16 +
                                    16].cpu().detach().numpy().transpose((0, 2, 3, 1))
                ref_in = image_in[j * 16:j * 16 + 16, 3:6, :,
                                  :].cpu().detach().numpy().transpose((0, 2, 3, 1))
                fls_in = image_in[j * 16:j * 16 + 16, 0:3, :,
                                  :].cpu().detach().numpy().transpose((0, 2, 3, 1))

                for k in range(g_out.shape[0]):
                    frame = np.concatenate(
                        (ref_in[k], g_out[k], fls_in[k], ref_out[k]), axis=1) * 255.0
                    writer.write(frame.astype(np.uint8))

            writer.release()

            os.system(
                'ffmpeg -y -i tmp_{:04d}.mp4 -pix_fmt yuv420p random_{:04d}.mp4'.format(i, i))
            os.system('rm tmp_{:04d}.mp4'.format(i))

    def single_test(self, jpg=None, fls=None, filename=None, prefix='', grey_only=False):
        import time
        st = time.time()
        self.G.eval()

        if(jpg is None):
            jpg = glob.glob1(self.opt_parser.single_test, '*.jpg')[0]
            jpg = cv2.imread(os.path.join(self.opt_parser.single_test, jpg))

        if(fls is None):
            fls = glob.glob1(self.opt_parser.single_test, '*.txt')[0]
            fls = np.loadtxt(os.path.join(self.opt_parser.single_test, fls))
            fls = fls * 95
            fls[:, 0::3] += 130
            fls[:, 1::3] += 80

        writer = cv2.VideoWriter(
            'out.mp4', cv2.VideoWriter_fourcc(*'mjpg'), 62.5, (256 * 3, 256))

        for i, frame in enumerate(fls):

            img_fl = np.ones(shape=(256, 256, 3)) * 255
            fl = frame.astype(int)
            img_fl = vis_landmark_on_img(img_fl, np.reshape(fl, (68, 3)))
            frame = np.concatenate(
                (img_fl, jpg), axis=2).astype(np.float32)/255.0

            image_in, image_out = frame.transpose(
                (2, 0, 1)), np.zeros(shape=(3, 256, 256))
            # image_in, image_out = frame.transpose((2, 1, 0)), np.zeros(shape=(3, 256, 256))
            image_in, image_out = torch.tensor(image_in, requires_grad=False), \
                torch.tensor(image_out, requires_grad=False)

            image_in, image_out = image_in.reshape(
                -1, 6, 256, 256), image_out.reshape(-1, 3, 256, 256)
            image_in, image_out = image_in.to(device), image_out.to(device)

            g_out = self.G(image_in)
            g_out = torch.tanh(g_out)

            g_out = g_out.cpu().detach().numpy().transpose((0, 2, 3, 1))
            g_out[g_out < 0] = 0
            ref_in = image_in[:, 3:6, :, :].cpu(
            ).detach().numpy().transpose((0, 2, 3, 1))
            fls_in = image_in[:, 0:3, :, :].cpu(
            ).detach().numpy().transpose((0, 2, 3, 1))
            # g_out = g_out.cpu().detach().numpy().transpose((0, 3, 2, 1))
            # g_out[g_out < 0] = 0
            # ref_in = image_in[:, 3:6, :, :].cpu().detach().numpy().transpose((0, 3, 2, 1))
            # fls_in = image_in[:, 0:3, :, :].cpu().detach().numpy().transpose((0, 3, 2, 1))

            if(grey_only):
                g_out_grey = np.mean(g_out, axis=3, keepdims=True)
                g_out[:, :, :, 0:1] = g_out[:, :, :,
                                            1:2] = g_out[:, :, :, 2:3] = g_out_grey

            for i in range(g_out.shape[0]):
                frame = np.concatenate(
                    (ref_in[i], g_out[i], fls_in[i]), axis=1) * 255.0
                writer.write(frame.astype(np.uint8))

        writer.release()
        print('Time - only video:', time.time() - st)

        if(filename is None):
            filename = 'v'
        os.system('ffmpeg -loglevel error -y -i out.mp4 -i {} -pix_fmt yuv420p -strict -2 examples/{}_{}.mp4'.format(
            'examples/'+filename[9:-16]+'.wav',
            prefix, filename[:-4]))
        # os.system('rm out.mp4')

        print('Time - ffmpeg add audio:', time.time() - st)
