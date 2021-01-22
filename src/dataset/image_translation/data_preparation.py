"""
 # Copyright 2020 Adobe
 # All Rights Reserved.

 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it.

"""

import os
import glob
import time
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.dataset.utils.Av2Flau_Convertor import Av2Flau_Convertor
import platform

src_dir = r'/content/MakeItTalk/mp4'
out_dir = r'/content/MakeItTalk/PreprocessedVox_imagetranslation'


def landmark_extraction(si, ei):
    '''

    :param si: start index
    :param ei: end index
    :return: save extracted landmarks to out_dir
    '''

    for folder_name in ['raw_wav', 'raw_fl3d', 'register_fl3d', 'dump', 'tmp_v', 'nn_result', 'ckpt', 'log']:
        try:
            os.mkdir(os.path.join(out_dir, folder_name))
        except:
            pass

    if(not os.path.isfile(os.path.join(out_dir, 'filename_index_new.txt'))):
        # generate all file list

        clip_len_count = [0] * 500
        id_clip_list = []

        ids = glob.glob1(src_dir, '*')
        ids.sort()
        for id in ids:
            print(id)
            clips = glob.glob1(os.path.join(src_dir, id), '*')
            clips.sort()
            for clip in clips:
                videos = glob.glob1(os.path.join(src_dir, id, clip), '*.mp4')
                clip_len_count[len(videos)] += 1
                # if(len(videos) > 10 and len(videos) < 30):
                #     id_clip_list.append((id, clip))
                id_clip_list.append((id, clip))

        print(clip_len_count)
        print(len(id_clip_list))

        files = []
        for id, clip in id_clip_list:
            cur_src_dir = os.path.join(src_dir, id, clip)
            cur_files = glob.glob1(cur_src_dir, '*.mp4')

            cur_files = np.random.permutation(cur_files)[0:1]

            cur_files = ['{}_x_{}_x_{}'.format(id, clip, f) for f in cur_files]

            files += cur_files

        with open(os.path.join(out_dir, 'filename_index_new.txt'), 'w') as f:
            for i, file in enumerate(files):
                f.write('{} {}\n'.format(i, file))
    else:
        with open(os.path.join(out_dir, 'filename_index_new.txt'), 'r') as f:
            lines = f.readlines()

        print(sys.argv)
        for line in lines[si:ei]:
            st = time.time()
            idx, file = int(line.split(' ')[0]), line.split(' ')[1][:-1]

            # # check if exists
            # video_dir = os.path.join(src_dir, file)
            # if ('\\' in video_dir):
            #     video_name = video_dir.split('\\')[-1]
            # else:
            #     video_name = video_dir.split('/')[-1]
            # save_name = os.path.join(out_dir.replace('VoxCeleb2_compressed_imagetranslation',
            #                                          'VoxCeleb2_imagetranslation'),
            #                          'raw_fl3d/fan_{:05d}_{}_3d.txt'.format(idx, video_name[:-4]))
            # if(os.path.isfile(save_name)):
            #     print('==> File {} {} exist, just copy'.format(idx, video_name[:-4]))
            #     shutil.copy(save_name,
            #                 os.path.join(out_dir, 'raw_fl3d/fan_{:05d}_{}_3d.txt'.format(idx, video_name[:-4])))
            #     continue

            c = Av2Flau_Convertor(video_dir=os.path.join(src_dir, file),
                                  out_dir=out_dir, idx=idx)
            c.convert()  # (save_audio=False, register=False, show=False)
            print('Idx: {}, Processed time (min): {}'.format(
                idx, (time.time() - st) / 60.0))


def landmark_image_to_data(si, ei, show=False):
    '''
    DROPPED DUE TO LARGE DISK SPACE CONSUME
    :param si:
    :param ei:
    :param show:
    :return:
    '''
    # load landmark
    print(src_dir)
    fls_filenames = glob.glob1(src_dir, '*')
    print(fls_filenames)
    pf = {}

    for i, fls_filename in enumerate(fls_filenames):

        fls = np.loadtxt(os.path.join(src_dir, fls_filename))
        print(i, '/', len(fls_filenames), fls.shape)

        mp4_filename = fls_filename[:-4].split('_x_')
        mp4_id = mp4_filename[0].split('_')[-1]
        mp4_vname = mp4_filename[1]
        mp4_vid = mp4_filename[2][:-3]
        print(mp4_id, mp4_vname, mp4_vid)
        video_dir = os.path.join(mp4_dir, mp4_id, mp4_vname, mp4_vid+'.mp4')
        print('video_dir : ' + video_dir)
        video = cv2.VideoCapture(video_dir)
        if (video.isOpened() == False):
            print('Unable to open video file')
            exit(0)

        if(show == True):
            length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)
            w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print('Process Video {}, len: {}, FPS: {:.2f}, W X H: {} x {}'.format(
                video_dir, length, fps, w, h))
            writer = cv2.VideoWriter('a.mp4', cv2.VideoWriter_fourcc(
                'M', 'J', 'P', 'G'), fps, (512, 256))

            # skip first several frames due to landmark extraction
            start_idx = fls[0, 0].astype(int)
            print('Skip beginning # {} frames'.format(start_idx))

            for _ in range(start_idx):
                ret, img_video = video.read()

             # save video and landmark in parallel
            for j in range(fls.shape[0]):
                img_fl = np.ones(shape=(224, 224, 3)) * 255
                idx = fls[j, 0]
                fl = fls[j, 1:].astype(int)
                img_fl = vis_landmark_on_img(img_fl, np.reshape(fl, (68, 3)))

                ret, img_video = video.read()

                frame = np.concatenate((img_fl, img_video), axis=1)
                frame = cv2.resize(frame, (512, 256))
                writer.write(frame.astype(np.uint8))

            video.release()
            writer.release()
            cv2.destroyAllWindows()

            exit(0)

        else:
            # skip first several frames due to landmark extraction
            start_idx = fls[0, 0].astype(int)
            print('Skip beginning # {} frames'.format(start_idx))
            for _ in range(start_idx):
                ret, img_video = video.read()

            # save video and landmark in parallel
            frames = []
            for j in range(fls.shape[0]):
                img_fl = np.ones(shape=(224, 224, 3)) * 255
                idx = fls[j, 0]
                fl = fls[j, 1:].astype(int)
                img_fl = vis_landmark_on_img(img_fl, np.reshape(fl, (68, 3)))

                ret, img_video = video.read()

                frame = np.concatenate((img_fl, img_video), axis=2)
                frame = cv2.resize(frame, (256, 256))  # 256 x 256  6
                frames.append(frame)
            frames = np.stack(frames, axis=0).astype(int)  # N x 256 x 256 x 6
            pf[fls_filename] = frames

    # save to pickle file
    # with open('train_data.pickle', 'wb') as handle:
    #     pickle.dump(pf, handle)


def vis_landmark_on_img(img, shape, linewidth=2):
    '''
    Visualize landmark on images.
    '''

    def draw_curve(idx_list, color=(0, 255, 0), loop=False, lineWidth=linewidth):
        for i in idx_list:
            cv2.line(img, (shape[i, 0], shape[i, 1]),
                     (shape[i + 1, 0], shape[i + 1, 1]), color, lineWidth)
        if (loop):
            cv2.line(img, (shape[idx_list[0], 0], shape[idx_list[0], 1]),
                     (shape[idx_list[-1] + 1, 0], shape[idx_list[-1] + 1, 1]), color, lineWidth)

    draw_curve(list(range(0, 16)), color=(255, 144, 25))  # jaw
    draw_curve(list(range(17, 21)), color=(50, 205, 50))  # eye brow
    draw_curve(list(range(22, 26)), color=(50, 205, 50))
    draw_curve(list(range(27, 35)), color=(208, 224, 63))  # nose
    draw_curve(list(range(36, 41)), loop=True, color=(71, 99, 255))  # eyes
    draw_curve(list(range(42, 47)), loop=True, color=(71, 99, 255))
    draw_curve(list(range(48, 59)), loop=True, color=(238, 130, 238))  # mouth
    draw_curve(list(range(60, 67)), loop=True, color=(238, 130, 238))

    return img


def vis_landmark_on_img98(img, shape, linewidth=2):
    '''
    Visualize landmark on images.
    '''

    def draw_curve(idx_list, color=(0, 255, 0), loop=False, lineWidth=linewidth):
        for i in idx_list:
            cv2.line(img, (shape[i, 0], shape[i, 1]),
                     (shape[i + 1, 0], shape[i + 1, 1]), color, lineWidth)
        if (loop):
            cv2.line(img, (shape[idx_list[0], 0], shape[idx_list[0], 1]),
                     (shape[idx_list[-1] + 1, 0], shape[idx_list[-1] + 1, 1]), color, lineWidth)

    draw_curve(list(range(0, 32)), color=(255, 144, 25))  # jaw
    draw_curve(list(range(33, 41)), color=(50, 205, 50), loop=True)  # eye brow
    draw_curve(list(range(42, 50)), color=(50, 205, 50), loop=True)
    draw_curve(list(range(51, 59)), color=(208, 224, 63))  # nose
    draw_curve(list(range(60, 67)), loop=True, color=(71, 99, 255))  # eyes
    draw_curve(list(range(68, 75)), loop=True, color=(71, 99, 255))
    draw_curve(list(range(76, 87)), loop=True, color=(238, 130, 238))  # mouth
    draw_curve(list(range(88, 95)), loop=True, color=(238, 130, 238))

    return img


def vis_landmark_on_img74(img, shape, linewidth=2):
    '''
    Visualize landmark on images.
    '''

    def draw_curve(idx_list, color=(0, 255, 0), loop=False, lineWidth=linewidth):
        for i in idx_list:
            cv2.line(img, (shape[i, 0], shape[i, 1]),
                     (shape[i + 1, 0], shape[i + 1, 1]), color, lineWidth)
        if (loop):
            cv2.line(img, (shape[idx_list[0], 0], shape[idx_list[0], 1]),
                     (shape[idx_list[-1] + 1, 0], shape[idx_list[-1] + 1, 1]), color, lineWidth)

    draw_curve(list(range(0, 16)), color=(255, 144, 25))  # jaw
    draw_curve(list(range(17, 21)), color=(
        50, 205, 50), loop=False)  # eye brow
    draw_curve(list(range(22, 26)), color=(50, 205, 50), loop=False)
    draw_curve(list(range(27, 35)), color=(208, 224, 63))  # nose
    draw_curve(list(range(36, 43)), loop=True, color=(71, 99, 255))  # eyes
    draw_curve(list(range(44, 51)), loop=True, color=(71, 99, 255))
    draw_curve(list(range(52, 63)), loop=True, color=(238, 130, 238))  # mouth
    draw_curve(list(range(64, 71)), loop=True, color=(238, 130, 238))

    return img
