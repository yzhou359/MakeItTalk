"""
 # Copyright 2020 Adobe
 # All Rights Reserved.

 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it.

"""

import torch.utils.data as data
import os
import glob
import platform
import numpy as np
import cv2
import torch
from src.dataset.image_translation.data_preparation import vis_landmark_on_img, vis_landmark_on_img98, vis_landmark_on_img74
from torch.utils.data.dataloader import default_collate

from thirdparty.AdaptiveWingLoss.utils.utils import get_preds_fromhm

from scipy.io import wavfile as wav
from scipy.signal import stft


src_dir = r'content/MakeItNow/PreprocessedVox_imagetranslation/raw_fl3d'
mp4_dir = r'content/MakeItNow/mp4'
src_outer_dir = r'content/MakeItNow/PreprocessedVox_imagetranslation'


class image_translation_raw_dataset(data.Dataset):

    def __init__(self, num_frames=16):

        self.src_dir = src_dir
        self.mp4_dir = mp4_dir

        self.fls_filenames = glob.glob1(self.src_dir, '*')
        self.num_random_frames = num_frames + 1

        print(os.name, len(self.fls_filenames))

    def __len__(self):
        return len(self.fls_filenames)

    def __getitem__(self, item):

        fls_filename = self.fls_filenames[item]

        # load landmark file
        fls = np.loadtxt(os.path.join(self.src_dir, fls_filename))

        # load mp4 file
        # ================= raw VOX version ================================
        mp4_filename = fls_filename[:-4].split('_x_')
        mp4_id = mp4_filename[0].split('_')[-1]
        mp4_vname = mp4_filename[1]
        mp4_vid = mp4_filename[2][:-3]
        video_dir = os.path.join(self.mp4_dir, mp4_id,
                                 mp4_vname, mp4_vid + '.mp4')
        # print('============================\nvideo_dir : ' + video_dir, item)
        # ======================================================================

        video = cv2.VideoCapture(video_dir)
        if (video.isOpened() == False):
            print('Unable to open video file')
            exit(0)

        # skip first several frames due to landmark extraction
        start_idx = (fls[0, 0]).astype(int)
        for _ in range(start_idx):
            ret, img_video = video.read()

        # save video and landmark in parallel
        frames = []
        random_frame_indices = np.random.permutation(
            fls.shape[0]-2)[0:self.num_random_frames]

        for j in range(int(fls.shape[0])):
            ret, img_video = video.read()

            if(j in random_frame_indices):
                img_fl = np.ones(shape=(224, 224, 3)) * 255
                idx = fls[j, 0]
                fl = fls[j, 1:].astype(int)

                img_fl = vis_landmark_on_img(img_fl, np.reshape(fl, (68, 3)))

                frame = np.concatenate((img_fl, img_video), axis=2)
                frame = cv2.resize(frame, (256, 256))  # 256 x 256  6
                frames.append(frame)

        frames = np.stack(frames, axis=0).astype(
            np.float32)/255.0  # N x 256 x 256 x 6

        image_in = np.concatenate(
            [frames[0:-1, :, :, 0:3], frames[1:, :, :, 3:6]], axis=3)
        image_out = frames[0:-1, :, :, 3:6]

        image_in, image_out = np.swapaxes(
            image_in, 1, 3), np.swapaxes(image_out, 1, 3)

        return image_in, image_out

    def my_collate(self, batch):
        batch = filter(lambda x: x is not None, batch)
        return default_collate(batch)


class image_translation_raw74_dataset(data.Dataset):

    def __init__(self, num_frames=16):

        self.src_dir = src_dir
        self.mp4_dir = mp4_dir

        self.fls_filenames = glob.glob1(self.src_dir, '*')
        self.num_random_frames = num_frames + 1

        print(os.name, len(self.fls_filenames))

    def __len__(self):
        return len(self.fls_filenames)

    def __getitem__(self, item):

        fls_filename = self.fls_filenames[item]

        # load landmark file
        fls = np.loadtxt(os.path.join(self.src_dir, fls_filename))

        # load mp4 file
        # ================= raw VOX version ================================
        mp4_filename = fls_filename[:-4].split('_x_')
        mp4_id = mp4_filename[0].split('_')[-1]
        mp4_vname = mp4_filename[1]
        mp4_vid = mp4_filename[2][:-3]
        video_dir = os.path.join(self.mp4_dir, mp4_id,
                                 mp4_vname, mp4_vid + '.mp4')
        # print('============================\nvideo_dir : ' + video_dir, item)
        # ======================================================================

        video = cv2.VideoCapture(video_dir)
        if (video.isOpened() == False):
            print('Unable to open video file')
            exit(0)

        # skip first several frames due to landmark extraction
        start_idx = (fls[0, 0]).astype(int)
        for _ in range(start_idx):
            ret, img_video = video.read()

        # save video and landmark in parallel
        frames = []
        fan_predict_landmarks = []
        random_frame_indices = np.random.permutation(
            fls.shape[0]-2)[0:self.num_random_frames]

        for j in range(int(fls.shape[0])):
            ret, img_video = video.read()

            if(j in random_frame_indices):
                fl = fls[j, 1:] / 224. * 256.
                fan_predict_landmarks.append(np.reshape(fl, (68, 3)))

                img_video = cv2.resize(img_video, (256, 256))
                frames.append(img_video.transpose((2, 0, 1)))

        fan_predict_landmarks = np.stack(fan_predict_landmarks, axis=0)
        frames = np.stack(frames, axis=0).astype(np.float32) / 255.0

        image_in = frames[1:, :, :]
        image_out = frames[0:-1, :, :]  # N x 3 x 256 x 256

        return image_in, image_out, fan_predict_landmarks[0:-1]

    def my_collate(self, batch):
        batch = filter(lambda x: x is not None, batch)
        return default_collate(batch)


class image_translation_raw_test_dataset(data.Dataset):

    def __init__(self, num_frames=16):

        self.src_dir = src_dir
        self.mp4_dir = mp4_dir

        self.fls_filenames = glob.glob1(self.src_dir, '*')
        self.num_random_frames = num_frames + 1

        print(os.name, len(self.fls_filenames))

    def __len__(self):
        return len(self.fls_filenames)

    def __getitem__(self, item):
        fls_filename = self.fls_filenames[item]

        # load landmark file
        fls = np.loadtxt(os.path.join(self.src_dir, fls_filename))
        from scipy.signal import savgol_filter
        fls = savgol_filter(fls, 11, 3, axis=0)

        # load random face
        random_fls_filename = self.fls_filenames[max(item - 1, 0)]
        mp4_filename = random_fls_filename[:-4].split('_x_')
        mp4_id = mp4_filename[0].split('_')[-1]
        mp4_vname = mp4_filename[1]
        mp4_vid = mp4_filename[2][:-3]
        random_video_dir = os.path.join(
            self.mp4_dir, mp4_id, mp4_vname, mp4_vid + '.mp4')
        print('============================\nvideo_dir : ' + random_video_dir, item)
        random_video = cv2.VideoCapture(random_video_dir)
        if (random_video.isOpened() == False):
            print('Unable to open video file')
            exit(0)
        _, random_face = random_video.read()

        # load mp4 file
        # ================= raw VOX version ================================
        mp4_filename = fls_filename[:-4].split('_x_')
        mp4_id = mp4_filename[0].split('_')[-1]
        mp4_vname = mp4_filename[1]
        mp4_vid = mp4_filename[2][:-3]
        video_dir = os.path.join(self.mp4_dir, mp4_id,
                                 mp4_vname, mp4_vid + '.mp4')
        # print('============================\nvideo_dir : ' + video_dir, item)
        # ======================================================================

        video = cv2.VideoCapture(video_dir)
        if (video.isOpened() == False):
            print('Unable to open video file')
            exit(0)

        # skip first several frames due to landmark extraction
        start_idx = (fls[0, 0]).astype(int)
        for _ in range(start_idx):
            ret, img_video = video.read()

        # save video and landmark in parallel
        frames = []

        for j in range(int(fls.shape[0])-2):
            ret, img_video = video.read()

            img_fl = np.ones(shape=(224, 224, 3)) * 255
            idx = fls[j, 0]
            fl = fls[j, 1:].astype(int)
            img_fl = vis_landmark_on_img(img_fl, np.reshape(fl, (68, 3)))

            # print(img_fl.shape, random_face.shape, img_video.shape)
            frame = np.concatenate((img_fl, random_face, img_video), axis=2)
            frame = cv2.resize(frame, (256, 256))  # 256 x 256  6
            frames.append(frame)

        frames = np.stack(frames, axis=0).astype(
            np.float32)/255.0  # N x 256 x 256 x 6
        image_in = frames[:, :, :, 0:6]
        image_out = frames[:, :, :, 6:9]

        image_in, image_out = np.swapaxes(
            image_in, 1, 3), np.swapaxes(image_out, 1, 3)
        return image_in, image_out

    def my_collate(self, batch):
        batch = filter(lambda x: x is not None, batch)
        return default_collate(batch)


class image_translation_preprocessed_dataset(data.Dataset):

    def __init__(self, num_frames=16):

        self.src_dir = src_dir
        self.mp4_dir = mp4_dir

        self.fls_filenames = glob.glob1(self.src_dir, '*')
        self.num_random_frames = num_frames + 1

        self.fps_scale = 2.5

        print(os.name, len(self.fls_filenames))

    def __len__(self):
        return len(self.fls_filenames)

    def __getitem__(self, item):
        fls_filename = self.fls_filenames[item]

        # load landmark file
        fls = np.loadtxt(os.path.join(self.src_dir, fls_filename))

        # # ================= preprocessed VOX version ================================
        video_dir = os.path.join(self.mp4_dir, fls_filename[10:-7]+'.mp4')
        # ======================================================================

        video = cv2.VideoCapture(video_dir)
        if (video.isOpened() == False):
            print('Unable to open video file')
            exit(0)

        # skip first several frames due to landmark extraction
        start_idx = (fls[0, 0] // self.fps_scale).astype(int)
        for _ in range(start_idx):
            ret, img_video = video.read()

        # save video and landmark in parallel
        frames = []
        random_frame_indices = np.random.permutation(
            int(fls.shape[0]//self.fps_scale)-2)[0:self.num_random_frames]

        for j in range(int(fls.shape[0]//self.fps_scale)):
            ret, img_video = video.read()

            if(j in random_frame_indices):
                img_fl = np.ones(shape=(256, 256, 3)) * 255
                idx = fls[int(j*self.fps_scale), 0]
                fl = fls[int(j*self.fps_scale), 1:].astype(int)
                img_fl = vis_landmark_on_img(img_fl, np.reshape(fl, (68, 3)))

                frame = np.concatenate((img_fl, img_video), axis=2)
                frames.append(frame)

        frames = np.stack(frames, axis=0).astype(
            np.float32)/255.0  # N x 256 x 256 x 6

        image_in = np.concatenate(
            [frames[0:-1, :, :, 0:3], frames[1:, :, :, 3:6]], axis=3)
        image_out = frames[0:-1, :, :, 3:6]

        image_in, image_out = np.swapaxes(
            image_in, 1, 3), np.swapaxes(image_out, 1, 3)
        return image_in, image_out

    def my_collate(self, batch):
        batch = filter(lambda x: x is not None, batch)
        return default_collate(batch)


class image_translation_preprocessed_test_dataset(data.Dataset):

    def __init__(self, num_frames=16):

        self.src_dir = src_dir
        self.mp4_dir = mp4_dir

        self.fls_filenames = glob.glob1(self.src_dir, '*')
        self.num_random_frames = num_frames + 1

        self.fps_scale = 2.5

        print(os.name, len(self.fls_filenames))

    def __len__(self):
        return len(self.fls_filenames)

    def __getitem__(self, item):
        fls_filename = self.fls_filenames[item]

        # load landmark file
        fls = np.loadtxt(os.path.join(self.src_dir, fls_filename))
        from scipy.signal import savgol_filter
        fls = savgol_filter(fls, 11, 3, axis=0)

        # load random face
        random_fls_filename = self.fls_filenames[max(item-1, 0)]
        # random_fls_filename = self.fls_filenames[max(item-1, 0)]
        random_video_dir = os.path.join(
            self.mp4_dir, random_fls_filename[10:-7] + '.mp4')
        random_video = cv2.VideoCapture(random_video_dir)
        if (random_video.isOpened() == False):
            print('Unable to open video file')
            exit(0)
        _, random_face = random_video.read()

        # # ================= preprocessed VOX version ================================
        video_dir = os.path.join(self.mp4_dir, fls_filename[10:-7]+'.mp4')
        # ======================================================================

        video = cv2.VideoCapture(video_dir)
        if (video.isOpened() == False):
            print('Unable to open video file')
            exit(0)

        # skip first several frames due to landmark extraction
        start_idx = (fls[0, 0] // self.fps_scale).astype(int)
        for _ in range(start_idx):
            ret, img_video = video.read()

        # save video and landmark in parallel
        frames = []
        for j in range(int(fls.shape[0]//self.fps_scale)):
            ret, img_video = video.read()

            # img_fl = np.ones(shape=(224, 224, 3)) * 255
            img_fl = np.ones(shape=(256, 256, 3)) * 255
            idx = fls[int(j*self.fps_scale), 0]
            fl = fls[int(j*self.fps_scale), 1:].astype(int)
            img_fl = vis_landmark_on_img(img_fl, np.reshape(fl, (68, 3)))

            frame = np.concatenate((img_fl, random_face, img_video), axis=2)
            # frame = cv2.resize(frame, (256, 256))  # 256 x 256  6
            frames.append(frame)

        frames = np.stack(frames, axis=0).astype(
            np.float32)/255.0  # N x 256 x 256 x 9

        image_in = frames[:, :, :, 0:6]
        image_out = frames[:, :, :, 6:9]

        image_in, image_out = np.swapaxes(
            image_in, 1, 3), np.swapaxes(image_out, 1, 3)
        return image_in, image_out

    def my_collate(self, batch):
        batch = filter(lambda x: x is not None, batch)
        return default_collate(batch)


class image_translation_raw98_dataset(data.Dataset):
    """
    Online landmark extraction with AWings
    Landmark setting: 98 landmarks
    """

    def __init__(self, num_frames=1):

        self.src_dir = src_outer_dir
        self.mp4_dir = mp4_dir

        # self.fls_filenames = glob.glob1(self.src_dir, '*')
        self.fls_filenames = np.loadtxt(os.path.join(
            self.src_dir, 'filename_index.txt'), dtype=str)[:, 1]
        self.num_random_frames = num_frames + 1

        print(os.name, self.fls_filenames.shape)

    def __len__(self):
        return self.fls_filenames.shape[0]

    def __getitem__(self, item):
        """
        Get landmark alignment outside in train_pass()
        """

        for i in range(5):
            fls_filename = self.fls_filenames[(
                item+i) % self.fls_filenames.shape[0]]

            # load mp4 file
            # ================= raw VOX version ================================
            mp4_filename = fls_filename[:-4].split('/')
            mp4_id = mp4_filename[1]
            mp4_vname = mp4_filename[2]
            mp4_vid = mp4_filename[3]
            video_dir = os.path.join(
                self.mp4_dir, mp4_id, mp4_vname, mp4_vid + '.mp4')
            # print('============================\nvideo_dir : ' + video_dir, item)
            # ======================================================================

            video = cv2.VideoCapture(video_dir)
            if (video.isOpened() == False):
                print('Unable to open video file')
            else:
                break

        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # save video and landmark in parallel
        frames = []
        random_frame_indices = np.random.permutation(
            length-2)[0:self.num_random_frames]

        for j in range(length):
            ret, img = video.read()

            if(j in random_frame_indices):
                img_video = cv2.resize(img, (256, 256))
                frames.append(img_video.transpose((2, 0, 1)))

        frames = np.stack(frames, axis=0).astype(np.float32)/255.0

        image_in = frames[1:, :, :]
        image_out = frames[0:-1, :, :]  # N x 3 x 256 x 256

        return image_in, image_out

    def __getitem_along_with_fa__(self, item):
        """
        Online get landmark alignment (deprecated)
        (can only run under num_works=0)
        """
        fls_filename = self.fls_filenames[item]

        # load mp4 file
        # ================= raw VOX version ================================
        mp4_filename = fls_filename[:-4].split('_x_')
        mp4_id = mp4_filename[0].split('_')[-1]
        mp4_vname = mp4_filename[1]
        mp4_vid = mp4_filename[2]
        video_dir = os.path.join(self.mp4_dir, mp4_id,
                                 mp4_vname, mp4_vid + '.mp4')
        # print('============================\nvideo_dir : ' + video_dir, item)
        # ======================================================================

        video = cv2.VideoCapture(video_dir)
        if (video.isOpened() == False):
            print('Unable to open video file')
            exit(0)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # save video and landmark in parallel
        frames = []
        random_frame_indices = np.random.permutation(
            length-2)[0:self.num_random_frames]

        for j in range(length):
            ret, img = video.read()

            if(j in random_frame_indices):
                # online landmark
                img_video = cv2.resize(img, (256, 256))
                img = img_video.transpose((2, 0, 1)) / 255.0
                inputs = torch.tensor(
                    img, dtype=torch.float32, requires_grad=False).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs, boundary_channels = self.model(inputs)
                pred_heatmap = outputs[-1][:, :-1, :, :][0].detach().cpu()
                pred_landmarks, _ = get_preds_fromhm(pred_heatmap.unsqueeze(0))
                pred_landmarks = pred_landmarks.squeeze().numpy() * 4

                img_fl = np.ones(shape=(256, 256, 3)) * 255
                img_fl = vis_landmark_on_img98(
                    img_fl * 255.0, pred_landmarks)  # 98x2

                frame = np.concatenate((img_fl, img_video), axis=2)
                frames.append(frame)

        frames = np.stack(frames, axis=0).astype(
            np.float32)/255.0  # N x 256 x 256 x 6

        image_in = np.concatenate(
            [frames[0:-1, :, :, 0:3], frames[1:, :, :, 3:6]], axis=3)
        image_out = frames[0:-1, :, :, 3:6]

        image_in, image_out = np.swapaxes(
            image_in, 1, 3), np.swapaxes(image_out, 1, 3)
        return image_in, image_out

    def my_collate(self, batch):
        batch = filter(lambda x: x is not None, batch)
        return default_collate(batch)


class image_translation_preprocessed98_dataset(data.Dataset):

    def __init__(self, num_frames=16):

        self.src_dir = src_dir
        self.mp4_dir = mp4_dir

        self.fls_filenames = glob.glob1(self.src_dir, '*')
        self.num_random_frames = num_frames + 1

        print(os.name, len(self.fls_filenames))

    def __len__(self):
        return len(self.fls_filenames)

    def __getitem__(self, item):
        fls_filename = self.fls_filenames[item]

        # # ================= preprocessed VOX version ================================
        video_dir = os.path.join(self.mp4_dir, fls_filename[10:-7]+'.mp4')
        # ======================================================================

        video = cv2.VideoCapture(video_dir)
        if (video.isOpened() == False):
            print('Unable to open video file')
            exit(0)

        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # save video and landmark in parallel
        frames = []
        random_frame_indices = np.random.permutation(
            length-2)[0:self.num_random_frames]

        for j in range(length):
            ret, img_video = video.read()

            if(j in random_frame_indices):
                img_video = cv2.resize(img_video, (256, 256))
                frames.append(img_video.transpose((2, 0, 1)))

        frames = np.stack(frames, axis=0).astype(
            np.float32)/255.0  # N x 256 x 256 x 6

        image_in = frames[1:, :, :]
        image_out = frames[0:-1, :, :]  # N x 3 x 256 x 256

        return image_in, image_out

    def my_collate(self, batch):
        batch = filter(lambda x: x is not None, batch)
        return default_collate(batch)


class image_translation_raw98_test_dataset(data.Dataset):

    def __init__(self, num_frames=16):

        self.src_dir = src_dir
        self.mp4_dir = mp4_dir

        # self.fls_filenames = glob.glob1(self.src_dir, '*')
        self.fls_filenames = np.loadtxt(os.path.join(
            self.src_dir, 'filename_index.txt'), dtype=str)[:, 1]

        self.num_random_frames = num_frames + 1

        print(os.name, len(self.fls_filenames))

    def __len__(self):
        return len(self.fls_filenames)

    def __getitem__(self, item):
        fls_filename = self.fls_filenames[item]

        # load random face
        random_fls_filename = self.fls_filenames[max(item - 10, 0)]
        mp4_filename = random_fls_filename[:-4].split('_x_')
        mp4_id = mp4_filename[0].split('_')[-1]
        mp4_vname = mp4_filename[1]
        mp4_vid = mp4_filename[2]
        random_video_dir = os.path.join(
            self.mp4_dir, mp4_id, mp4_vname, mp4_vid + '.mp4')
        print('============================\nvideo_dir : ' + random_video_dir, item)
        random_video = cv2.VideoCapture(random_video_dir)
        if (random_video.isOpened() == False):
            print('Unable to open video file')
            exit(0)
        _, random_face = random_video.read()
        random_face = cv2.resize(random_face, (256, 256))

        # load mp4 file
        # ================= raw VOX version ================================
        mp4_filename = fls_filename[:-4].split('_x_')
        mp4_id = mp4_filename[0].split('_')[-1]
        mp4_vname = mp4_filename[1]
        mp4_vid = mp4_filename[2]
        video_dir = os.path.join(self.mp4_dir, mp4_id,
                                 mp4_vname, mp4_vid + '.mp4')
        # print('============================\nvideo_dir : ' + video_dir, item)
        # ======================================================================

        video = cv2.VideoCapture(video_dir)
        if (video.isOpened() == False):
            print('Unable to open video file')
            exit(0)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # save video and landmark in parallel
        frames = []

        for j in range(length):
            ret, img_video = video.read()

            img_video = cv2.resize(img_video, (256, 256))
            frame = np.concatenate((random_face, img_video), axis=2)
            frames.append(frame.transpose((2, 0, 1)))

        frames = np.stack(frames, axis=0).astype(
            np.float32) / 255.0  # N x 256 x 256 x 9

        image_in = frames[:, 0:3]
        image_out = frames[:, 3:6]
        return image_in, image_out

    def my_collate(self, batch):
        batch = filter(lambda x: x is not None, batch)
        return default_collate(batch)


class image_translation_preprocessed98_test_dataset(data.Dataset):

    def __init__(self, num_frames=16):

        self.src_dir = src_dir
        self.mp4_dir = mp4_dir

        self.fls_filenames = glob.glob1(self.src_dir, '*')
        self.num_random_frames = num_frames + 1

        print(os.name, len(self.fls_filenames))

    def __len__(self):
        return len(self.fls_filenames)

    def __getitem__(self, item):
        fls_filename = self.fls_filenames[item]

        # load random face
        random_fls_filename = self.fls_filenames[max(item-10, 0)]
        # random_fls_filename = self.fls_filenames[max(item-1, 0)]
        random_video_dir = os.path.join(
            self.mp4_dir, random_fls_filename[10:-7] + '.mp4')
        random_video = cv2.VideoCapture(random_video_dir)
        if (random_video.isOpened() == False):
            print('Unable to open video file')
            exit(0)
        _, random_face = random_video.read()

        # # ================= preprocessed VOX version ================================
        video_dir = os.path.join(self.mp4_dir, fls_filename[10:-7]+'.mp4')
        # ======================================================================

        video = cv2.VideoCapture(video_dir)
        if (video.isOpened() == False):
            print('Unable to open video file')
            exit(0)

        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # save video and landmark in parallel
        frames = []
        for j in range(length):
            ret, img_video = video.read()

            img_video = cv2.resize(img_video, (256, 256))
            frame = np.concatenate((random_face, img_video), axis=2)
            frames.append(frame.transpose((2, 0, 1)))

        frames = np.stack(frames, axis=0).astype(
            np.float32)/255.0  # N x 256 x 256 x 9

        image_in = frames[:, 0:3]
        image_out = frames[:, 3:6]

        return image_in, image_out

    def my_collate(self, batch):
        batch = filter(lambda x: x is not None, batch)
        return default_collate(batch)


class image_translation_raw98_with_audio_dataset(data.Dataset):
    """
    Online landmark extraction with AWings
    Landmark setting: 98 landmarks
    """

    def __init__(self, num_frames=1):

        self.src_dir = src_outer_dir
        self.mp4_dir = mp4_dir

        # self.fls_filenames = glob.glob1(self.src_dir, '*')
        self.fls_filenames = np.loadtxt(os.path.join(
            self.src_dir, 'filename_index.txt'), dtype=str)[:, 1]
        self.num_random_frames = num_frames + 1

        print(os.name, self.fls_filenames.shape)

    def __len__(self):
        return self.fls_filenames.shape[0]

    def __getitem__(self, item):
        """
        Get landmark alignment outside in train_pass()
        """

        for i in range(5):
            fls_filename = self.fls_filenames[(
                item+i) % self.fls_filenames.shape[0]]

            # load mp4 file
            # ================= raw VOX version ================================
            mp4_filename = fls_filename[:-4].split('_x_')
            mp4_id = mp4_filename[0].split('_')[-1]
            mp4_vname = mp4_filename[1]
            mp4_vid = mp4_filename[2]
            video_dir = os.path.join(
                self.mp4_dir, mp4_id, mp4_vname, mp4_vid + '.mp4')
            # print('============================\nvideo_dir : ' + video_dir, item)
            # ======================================================================

            video = cv2.VideoCapture(video_dir)
            if (video.isOpened() == False):
                print('Unable to open video file')
            else:
                break

        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # save video and landmark in parallel
        frames = []
        random_frame_indices = np.random.permutation(
            max(1, length-12))[0:self.num_random_frames]
        random_frame_indices = [item + 5 for item in random_frame_indices]

        for j in range(length):
            ret, img = video.read()

            if(j in random_frame_indices):
                img_video = cv2.resize(img, (256, 256))
                frames.append(img_video.transpose((2, 0, 1)))

        frames = np.stack(frames, axis=0).astype(np.float32)/255.0

        image_in = frames[1:, :, :]
        image_out = frames[0:-1, :, :]  # N x 3 x 256 x 256

        # audio
        os.system('ffmpeg -y -loglevel error -i {} -vn -ar 16000 -ac 1 {}'.format(
            video_dir, video_dir.replace('.mp4', '.wav')
        ))
        sample_rate, samples = wav.read(video_dir.replace('.mp4', '.wav'))
        assert (sample_rate == 16000)
        if (len(samples.shape) > 1):
            samples = samples[:, 0]  # pick mono

        # 1 frame = 1/25 * 16k = 640 samples => windowsize=320,  overlap=160
        # 80 overlap => 200 / 1 sec, 8 / 1 frame
        f, t, Zxx = stft(samples, fs=sample_rate, nperseg=640, noverlap=560)
        stft_abs = np.log(np.abs(Zxx) ** 2 + 1e-10)
        stft_abs = stft_abs / np.max(stft_abs)
        os.remove(video_dir.replace('.mp4', '.wav'))

        # we want 0.2s before, 5 frames, 40 dims
        # and 0.2s after (may remove later)
        audio_in = []
        for item in random_frame_indices:
            sel_audio_clip = stft_abs[:, (item-5)*8:(item+5)*8]
            assert sel_audio_clip.shape[1] == 80
            audio_in.append(np.expand_dims(
                cv2.resize(sel_audio_clip, (256, 256)), axis=0))

        audio_in = np.stack(audio_in[0:-1], axis=0).astype(np.float32)
        # image_in = np.concatenate([image_in, audio_in], axis=1)

        return image_in, image_out, audio_in

    def my_collate(self, batch):
        batch = filter(lambda x: x is not None, batch)
        return default_collate(batch)


class image_translation_raw98_with_audio_test_dataset(data.Dataset):
    """
    Online landmark extraction with AWings
    Landmark setting: 98 landmarks
    """

    def __init__(self, num_frames=1):

        self.src_dir = src_outer_dir
        self.mp4_dir = mp4_dir

        # self.fls_filenames = glob.glob1(self.src_dir, '*')
        self.fls_filenames = np.loadtxt(os.path.join(
            self.src_dir, 'filename_index.txt'), dtype=str)[:, 1]
        self.num_random_frames = num_frames + 1

        print(os.name, self.fls_filenames.shape)

    def __len__(self):
        return self.fls_filenames.shape[0]

    def __getitem__(self, item):
        """
        Get landmark alignment outside in train_pass()
        """
        # load random face
        random_fls_filename = self.fls_filenames[max(item - 10, 0)]
        mp4_filename = random_fls_filename[:-4].split('_x_')
        mp4_id = mp4_filename[0].split('_')[-1]
        mp4_vname = mp4_filename[1]
        mp4_vid = mp4_filename[2]
        random_video_dir = os.path.join(
            self.mp4_dir, mp4_id, mp4_vname, mp4_vid + '.mp4')
        print('============================\nvideo_dir : ' + random_video_dir, item)
        random_video = cv2.VideoCapture(random_video_dir)
        if (random_video.isOpened() == False):
            print('Unable to open video file')
            exit(0)
        _, random_face = random_video.read()
        random_face = cv2.resize(random_face, (256, 256))

        fls_filename = self.fls_filenames[item]
        # load mp4 file
        # ================= raw VOX version ================================
        mp4_filename = fls_filename[:-4].split('_x_')
        mp4_id = mp4_filename[0].split('_')[-1]
        mp4_vname = mp4_filename[1]
        mp4_vid = mp4_filename[2]
        video_dir = os.path.join(self.mp4_dir, mp4_id,
                                 mp4_vname, mp4_vid + '.mp4')
        # print('============================\nvideo_dir : ' + video_dir, item)
        # ======================================================================

        video = cv2.VideoCapture(video_dir)
        if (video.isOpened() == False):
            print('Unable to open video file')

        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # save video and landmark in parallel
        frames = []
        for j in range(5, length-5):
            ret, img_video = video.read()

            img_video = cv2.resize(img_video, (256, 256))
            frame = np.concatenate((random_face, img_video), axis=2)
            frames.append(frame.transpose((2, 0, 1)))

        frames = np.stack(frames, axis=0).astype(
            np.float32) / 255.0  # N x 256 x 256 x 9

        image_in = frames[:, 0:3]
        image_out = frames[:, 3:6]

        # audio
        os.system('ffmpeg -y -loglevel error -i {} -vn -ar 16000 -ac 1 {}'.format(
            video_dir, video_dir.replace('.mp4', '.wav')
        ))
        sample_rate, samples = wav.read(video_dir.replace('.mp4', '.wav'))
        assert (sample_rate == 16000)
        if (len(samples.shape) > 1):
            samples = samples[:, 0]  # pick mono

        # 1 frame = 1/25 * 16k = 640 samples => windowsize=320,  overlap=160
        # 80 overlap => 200 / 1 sec, 8 / 1 frame
        f, t, Zxx = stft(samples, fs=sample_rate, nperseg=640, noverlap=560)
        stft_abs = np.log(np.abs(Zxx) ** 2 + 1e-10)
        stft_abs = stft_abs / np.max(stft_abs)
        os.remove(video_dir.replace('.mp4', '.wav'))

        # we want 0.2s before, 5 frames, 40 dims
        # and 0.2s after (may remove later)
        audio_in = []
        for item in range(5, length-5):
            sel_audio_clip = stft_abs[:, (item-5)*8:(item+5)*8]
            assert sel_audio_clip.shape[1] == 80
            audio_in.append(np.expand_dims(
                cv2.resize(sel_audio_clip, (256, 256)), axis=0))

        audio_in = np.stack(audio_in, axis=0).astype(np.float32)
        # image_in = np.concatenate([image_in, audio_in], axis=1)

        return image_in, image_out, audio_in

    def my_collate(self, batch):
        batch = filter(lambda x: x is not None, batch)
        return default_collate(batch)


if __name__ == '__main__':
    d = image_translation_raw_dataset()
    d_loader = torch.utils.data.DataLoader(d, batch_size=4, shuffle=True)
    print(len(d))
    for i, batch in enumerate(d_loader):
        print(i, batch[0].shape, batch[1].shape)
