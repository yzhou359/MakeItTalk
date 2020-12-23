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
from src.dataset.utils.Av2Flau_Convertor import Av2Flau_Convertor

# out_dir = r'/mnt/nfs/scratch1/yangzhou/PreprocessedVox_imagetranslation'
# src_dir = r'/mnt/nfs/scratch1/yangzhou/vox_p3/train'

out_dir = r'/content/MakeItTalk/PreprocessedVox_imagetranslation'
src_dir = r'/content/MakeItTalk/mp4'

''' Step 1. Data preparation '''
# landmark extraction
# landmark_extraction(int(sys.argv[1]), int(sys.argv[2]))


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

    if(not os.path.isfile(os.path.join(out_dir, 'filename_index.txt'))):
        # generate all file list
        # files = glob.glob1(src_dir, '*.mp4')
        files = []
        folders = os.listdir(src_dir)
        for folder in folders:
            for i in os.listdir(os.path.join(src_dir, folder)):
                file = glob.glob(os.path.join(src_dir, folder, i, '*.mp4'))
                for j in file:
                    files.append(j)

        with open(os.path.join(out_dir, 'filename_index.txt'), 'w') as f:
            for i, file in enumerate(files):
                f.write('{} {}\n'.format(i, file))
    else:
        with open(os.path.join(out_dir, 'filename_index.txt'), 'r') as f:
            lines = f.readlines()

        print(sys.argv)
        for line in lines[si:ei]:
            st = time.time()
            idx, file = int(line.split(' ')[0]), line.split(' ')[1][:-1]

            c = Av2Flau_Convertor(video_dir=os.path.join(src_dir, file),
                                  out_dir=out_dir, idx=idx)
            # (save_audio=False, register=False, show=False)
            c.convert(show=False)
            print('Idx: {}, Processed time (min): {}'.format(
                idx, (time.time() - st) / 60.0))
