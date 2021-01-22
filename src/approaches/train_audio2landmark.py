"""
 # Copyright 2020 Adobe
 # All Rights Reserved.
 
 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it.
 
"""

import os
import torch.nn.parallel
import torch.utils.data
from src.dataset.audio2landmark.audio2landmark_dataset import Audio2landmark_Dataset
from src.models.model_audio2landmark import *
from util.utils import get_n_params
import numpy as np
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Audio2landmark_model():

    def __init__(self, opt_parser, jpg_shape=None):
        '''
        Init model with opt_parser
        '''
        print('Run on device:', device)

        # Step 1 : load opt_parser
        self.opt_parser = opt_parser
        self.std_face_id = np.loadtxt(
            'src/dataset/utils/STD_FACE_LANDMARKS.txt')
        if(jpg_shape is not None):
            self.std_face_id = jpg_shape
        self.std_face_id = self.std_face_id.reshape(1, 204)
        self.std_face_id = torch.tensor(
            self.std_face_id, requires_grad=False, dtype=torch.float).to(device)

        self.eval_data = Audio2landmark_Dataset(dump_dir='examples/dump',
                                                dump_name='random',
                                                status='val',
                                                num_window_frames=18,
                                                num_window_step=1)
        self.eval_dataloader = torch.utils.data.DataLoader(self.eval_data, batch_size=1,
                                                           shuffle=False, num_workers=0,
                                                           collate_fn=self.eval_data.my_collate_in_segments)
        print('EVAL num videos: {}'.format(len(self.eval_data)))

        # Step 3: Load model
        self.G = Audio2landmark_pos(drop_out=0.5,
                                    spk_emb_enc_size=128,
                                    c_enc_hidden_size=256,
                                    transformer_d_model=32, N=2, heads=2,
                                    z_size=128, audio_dim=256)
        print('G: Running on {}, total num params = {:.2f}M'.format(
            device, get_n_params(self.G)/1.0e6))

        model_dict = self.G.state_dict()
        ckpt = torch.load(opt_parser.load_a2l_G_name)
        pretrained_dict = {k: v for k, v in ckpt['G'].items() if k.split('.')[
            0] not in ['comb_mlp']}
        model_dict.update(pretrained_dict)
        self.G.load_state_dict(model_dict)

        print('======== LOAD PRETRAINED FACE ID MODEL {} ========='.format(
            opt_parser.load_a2l_G_name))
        self.G.to(device)

        ''' baseline model '''
        self.C = Audio2landmark_content(num_window_frames=18,
                                        in_size=80, use_prior_net=True,
                                        bidirectional=False, drop_out=0.5)

        ckpt = torch.load(opt_parser.load_a2l_C_name)
        self.C.load_state_dict(ckpt['model_g_face_id'])
        # self.C.load_state_dict(ckpt['C'])
        print('======== LOAD PRETRAINED FACE ID MODEL {} ========='.format(
            opt_parser.load_a2l_C_name))
        self.C.to(device)

        self.t_shape_idx = (27, 28, 29, 30, 33, 36, 39, 42, 45)
        self.anchor_t_shape = np.loadtxt(
            'src/dataset/utils/STD_FACE_LANDMARKS.txt')
        self.anchor_t_shape = self.anchor_t_shape[self.t_shape_idx, :]

        with open(os.path.join('examples', 'dump', 'emb.pickle'), 'rb') as fp:
            self.test_embs = pickle.load(fp)

        print('====================================')
        for key in self.test_embs.keys():
            print(key)
        print('====================================')

    def __train_face_and_pos__(self, fls, aus, embs, face_id, smooth_win=31, close_mouth_ratio=.99):

        fls_without_traj = fls[:, 0, :].detach().clone().requires_grad_(False)

        if (face_id.shape[0] == 1):
            face_id = face_id.repeat(aus.shape[0], 1)
        face_id = face_id.requires_grad_(False)
        baseline_face_id = face_id.detach()

        z = torch.tensor(torch.zeros(
            aus.shape[0], 128), requires_grad=False, dtype=torch.float).to(device)
        fl_dis_pred, _, spk_encode = self.G(
            aus, embs * 3.0, face_id, fls_without_traj, z, add_z_spk=False)

        # ADD CONTENT
        from scipy.signal import savgol_filter
        smooth_length = int(
            min(fl_dis_pred.shape[0]-1, smooth_win) // 2 * 2 + 1)
        fl_dis_pred = savgol_filter(
            fl_dis_pred.cpu().numpy(), smooth_length, 3, axis=0)
    #
        ''' ================ close pose-branch mouth ================== '''
        fl_dis_pred = fl_dis_pred.reshape((-1, 68, 3))
        index1 = list(range(60-1, 55-1, -1))
        index2 = list(range(68-1, 65-1, -1))
        mean_out = 0.5 * fl_dis_pred[:, 49:54] + 0.5 * fl_dis_pred[:, index1]
        fl_dis_pred[:, 49:54] = mean_out * close_mouth_ratio + \
            fl_dis_pred[:, 49:54] * (1 - close_mouth_ratio)
        fl_dis_pred[:, index1] = mean_out * close_mouth_ratio + \
            fl_dis_pred[:, index1] * (1 - close_mouth_ratio)
        mean_in = 0.5 * (fl_dis_pred[:, 61:64] + fl_dis_pred[:, index2])
        fl_dis_pred[:, 61:64] = mean_in * close_mouth_ratio + \
            fl_dis_pred[:, 61:64] * (1 - close_mouth_ratio)
        fl_dis_pred[:, index2] = mean_in * close_mouth_ratio + \
            fl_dis_pred[:, index2] * (1 - close_mouth_ratio)
        fl_dis_pred = fl_dis_pred.reshape(-1, 204)
        ''' ============================================================= '''

        fl_dis_pred = torch.tensor(fl_dis_pred).to(
            device) * self.opt_parser.amp_pos

        residual_face_id = baseline_face_id

        # ''' CALIBRATION '''
        baseline_pred_fls, _ = self.C(aus[:, 0:18, :], residual_face_id)
        baseline_pred_fls = self.__calib_baseline_pred_fls__(baseline_pred_fls)
        fl_dis_pred += baseline_pred_fls

        return fl_dis_pred, face_id[0:1, :]

    def __calib_baseline_pred_fls_old_(self, baseline_pred_fls, residual_face_id, aus):
        mean_face_id = torch.mean(
            baseline_pred_fls.detach(), dim=0, keepdim=True)
        residual_face_id -= mean_face_id.view(1, 204) * 1.
        baseline_pred_fls, _ = self.C(aus, residual_face_id)
        baseline_pred_fls[:, 48 * 3::3] *= self.opt_parser.amp_lip_x  # mouth x
        baseline_pred_fls[:, 48 * 3 +
                          1::3] *= self.opt_parser.amp_lip_y  # mouth y
        return baseline_pred_fls

    def __calib_baseline_pred_fls__(self, baseline_pred_fls, ratio=0.5):
        np_fl_dis_pred = baseline_pred_fls.detach().cpu().numpy()
        K = int(np_fl_dis_pred.shape[0] * ratio)
        for calib_i in range(204):
            min_k_idx = np.argpartition(np_fl_dis_pred[:, calib_i], K)
            m = np.mean(np_fl_dis_pred[min_k_idx[:K], calib_i])
            np_fl_dis_pred[:, calib_i] = np_fl_dis_pred[:, calib_i] - m
        baseline_pred_fls = torch.tensor(
            np_fl_dis_pred, requires_grad=False).to(device)
        baseline_pred_fls[:, 48 * 3::3] *= self.opt_parser.amp_lip_x  # mouth x
        baseline_pred_fls[:, 48 * 3 +
                          1::3] *= self.opt_parser.amp_lip_y  # mouth y
        return baseline_pred_fls

    def __train_pass__(self, au_emb=None, centerize_face=False, no_y_rotation=False, vis_fls=False):

        # Step 1: init setup
        self.G.eval()
        self.C.eval()
        data = self.eval_data
        dataloader = self.eval_dataloader

        # Step 2: train for each batch
        for i, batch in enumerate(dataloader):

            global_id, video_name = data[i][0][1][0], data[i][0][1][1][:-4]

            # Step 2.1: load batch data from dataloader (in segments)
            inputs_fl, inputs_au, inputs_emb = batch

            keys = self.opt_parser.reuse_train_emb_list
            if(len(keys) == 0):
                keys = ['audio_embed']
            for key in keys:  # ['45hn7-LXDX8']: #['sxCbrYjBsGA']:#
                # load saved emb
                if(au_emb is None):
                    emb_val = self.test_embs[key]
                else:
                    emb_val = au_emb[i]

                inputs_emb = np.tile(emb_val, (inputs_emb.shape[0], 1))
                inputs_emb = torch.tensor(
                    inputs_emb, dtype=torch.float, requires_grad=False)
                inputs_fl, inputs_au, inputs_emb = inputs_fl.to(
                    device), inputs_au.to(device), inputs_emb.to(device)

                std_fls_list, fls_pred_face_id_list, fls_pred_pos_list = [], [], []
                seg_bs = 512

                for j in range(0, inputs_fl.shape[0], seg_bs):

                    # Step 3.1: load segments
                    inputs_fl_segments = inputs_fl[j: j + seg_bs]
                    inputs_au_segments = inputs_au[j: j + seg_bs]
                    inputs_emb_segments = inputs_emb[j: j + seg_bs]

                    if(inputs_fl_segments.shape[0] < 10):
                        continue

                    input_face_id = self.std_face_id

                    fl_dis_pred_pos, input_face_id = \
                        self.__train_face_and_pos__(inputs_fl_segments, inputs_au_segments, inputs_emb_segments,
                                                    input_face_id)

                    fl_dis_pred_pos = (fl_dis_pred_pos +
                                       input_face_id).data.cpu().numpy()
                    ''' solve inverse lip '''
                    fl_dis_pred_pos = self.__solve_inverse_lip2__(
                        fl_dis_pred_pos)
                    fls_pred_pos_list += [fl_dis_pred_pos]

                fake_fls_np = np.concatenate(fls_pred_pos_list)

                # revise nose top point
                fake_fls_np[:, 27 * 3:28 * 3] = fake_fls_np[:,
                                                            28 * 3:29 * 3] * 2 - fake_fls_np[:, 29 * 3:30 * 3]

                # fake_fls_np[:, 48*3+1::3] += 0.1

                # smooth
                from scipy.signal import savgol_filter
                fake_fls_np = savgol_filter(fake_fls_np, 5, 3, axis=0)

                if(centerize_face):
                    std_m = np.mean(self.std_face_id.detach().cpu().numpy().reshape((1, 68, 3)),
                                    axis=1, keepdims=True)
                    fake_fls_np = fake_fls_np.reshape((-1, 68, 3))
                    fake_fls_np = fake_fls_np - \
                        np.mean(fake_fls_np, axis=1, keepdims=True) + std_m
                    fake_fls_np = fake_fls_np.reshape((-1, 68 * 3))

                if(no_y_rotation):
                    std = self.std_face_id.detach().cpu().numpy().reshape(68, 3)
                    std_t_shape = std[self.t_shape_idx, :]
                    fake_fls_np = fake_fls_np.reshape(
                        (fake_fls_np.shape[0], 68, 3))
                    frame_t_shape = fake_fls_np[:, self.t_shape_idx, :]
                    from util.icp import icp
                    from scipy.spatial.transform import Rotation as R
                    for i in range(frame_t_shape.shape[0]):
                        T, distance, itr = icp(frame_t_shape[i], std_t_shape)
                        landmarks = np.hstack(
                            (frame_t_shape[i], np.ones((9, 1))))
                        rot_mat = T[:3, :3]
                        r = R.from_dcm(rot_mat).as_euler('xyz')
                        r = [0., r[1], r[2]]
                        r = R.from_euler('xyz', r).as_dcm()
                        # print(frame_t_shape[i, 0], r)
                        landmarks = np.hstack(
                            (fake_fls_np[i] - T[:3, 3:4].T, np.ones((68, 1))))
                        T2 = np.hstack((r, T[:3, 3:4]))
                        fake_fls_np[i] = np.dot(T2, landmarks.T).T
                        # print(frame_t_shape[i, 0])
                    fake_fls_np = fake_fls_np.reshape((-1, 68 * 3))

                filename = 'pred_fls_{}_{}.txt'.format(
                    video_name.split('\\')[-1].split('/')[-1], key)
                np.savetxt(os.path.join(self.opt_parser.output_folder,
                                        filename), fake_fls_np, fmt='%.6f')

                # ''' Visualize result in landmarks '''
                if(vis_fls):
                    from util.vis import Vis
                    Vis(fls=fake_fls_np, filename=video_name.split('\\')[-1].split('/')[-1], fps=62.5,
                        audio_filenam=os.path.join('examples', video_name.split('\\')[-1].split('/')[-1]+'.wav'))

    def __close_face_lip__(self, fl):
        facelandmark = fl.reshape(-1, 68, 3)
        from util.geo_math import area_of_polygon
        min_area_lip, idx = 999, 0
        for i, fls in enumerate(facelandmark):
            area_of_mouth = area_of_polygon(fls[list(range(60, 68)), 0:2])
            if (area_of_mouth < min_area_lip):
                min_area_lip = area_of_mouth
                idx = i
        return idx

    def test(self, au_emb=None):
        with torch.no_grad():
            self.__train_pass__(au_emb, vis_fls=True)

    def __solve_inverse_lip2__(self, fl_dis_pred_pos_numpy):
        for j in range(fl_dis_pred_pos_numpy.shape[0]):
            init_face = self.std_face_id.detach().cpu().numpy()
            from util.geo_math import area_of_signed_polygon
            fls = fl_dis_pred_pos_numpy[j].reshape(68, 3)
            area_of_mouth = area_of_signed_polygon(
                fls[list(range(60, 68)), 0:2])
            if (area_of_mouth < 0):
                fl_dis_pred_pos_numpy[j, 65 * 3:66 * 3] = 0.5 * (
                    fl_dis_pred_pos_numpy[j, 63 * 3:64 * 3] + fl_dis_pred_pos_numpy[j, 65 * 3:66 * 3])
                fl_dis_pred_pos_numpy[j, 63 * 3:64 *
                                      3] = fl_dis_pred_pos_numpy[j, 65 * 3:66 * 3]
                fl_dis_pred_pos_numpy[j, 66 * 3:67 * 3] = 0.5 * (
                    fl_dis_pred_pos_numpy[j, 62 * 3:63 * 3] + fl_dis_pred_pos_numpy[j, 66 * 3:67 * 3])
                fl_dis_pred_pos_numpy[j, 62 * 3:63 *
                                      3] = fl_dis_pred_pos_numpy[j, 66 * 3:67 * 3]
                fl_dis_pred_pos_numpy[j, 67 * 3:68 * 3] = 0.5 * (
                    fl_dis_pred_pos_numpy[j, 61 * 3:62 * 3] + fl_dis_pred_pos_numpy[j, 67 * 3:68 * 3])
                fl_dis_pred_pos_numpy[j, 61 * 3:62 *
                                      3] = fl_dis_pred_pos_numpy[j, 67 * 3:68 * 3]
                p = max([j-1, 0])
                fl_dis_pred_pos_numpy[j, 55 * 3+1:59 * 3+1:3] = fl_dis_pred_pos_numpy[j, 64 * 3+1:68 * 3+1:3] \
                    + fl_dis_pred_pos_numpy[p, 55 * 3+1:59 * 3+1:3] \
                    - fl_dis_pred_pos_numpy[p, 64 * 3+1:68 * 3+1:3]
                fl_dis_pred_pos_numpy[j, 59 * 3+1:60 * 3+1:3] = fl_dis_pred_pos_numpy[j, 60 * 3+1:61 * 3+1:3] \
                    + fl_dis_pred_pos_numpy[p, 59 * 3+1:60 * 3+1:3] \
                    - fl_dis_pred_pos_numpy[p, 60 * 3+1:61 * 3+1:3]
                fl_dis_pred_pos_numpy[j, 49 * 3+1:54 * 3+1:3] = fl_dis_pred_pos_numpy[j, 60 * 3+1:65 * 3+1:3] \
                    + fl_dis_pred_pos_numpy[p, 49 * 3+1:54 * 3+1:3] \
                    - fl_dis_pred_pos_numpy[p, 60 * 3+1:65 * 3+1:3]
        return fl_dis_pred_pos_numpy
