"""
 # Copyright 2020 Adobe
 # All Rights Reserved.
 
 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it.
 
"""

import numpy as np
import os
import ffmpeg
import cv2
import face_alignment
from src.dataset.utils import icp


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class ShapeParts:
    def __init__(self, np_pts):
        self.data = np_pts

    def part(self, idx):
        return Point(self.data[idx, 0], self.data[idx, 1])


class Av2Flau_Convertor():
    """

    Any video to facial landmark and audio numpy data converter.

    """

    def __init__(self, video_dir, out_dir, idx=0):

        self.video_dir = video_dir
        if ('\\' in video_dir):
            self.video_name = video_dir.split('\\')[-1]
        else:
            self.video_name = video_dir.split('/')[-1]
        self.out_dir = out_dir
        self.idx = idx
        self.input_format = self.video_dir[-4:]

        # landmark predictor = FANet
        self.predictor = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, device='cuda', flip_input=True)

        # landmark register
        self.t_shape_idx = (27, 28, 29, 30, 33, 36, 39, 42, 45)

    def convert(self, max_num_frames=250, save_audio=False, show=False, register=False):

        # Step 1: preclean video: check stream==2, convert fps/sample_rate,
        ret, wfn = self.__preclean_video__()
        if (not ret):
            return

        # Step 2: detect facial landmark
        wfn = self.video_dir.replace(self.input_format, '_preclean.mp4')
        ret, fl2d, fl3d = self.__video_facial_landmark_detection__(
            video_dir=wfn, display=False, max_num_frames=max_num_frames)
        if (not ret):
            return
        if (len(fl3d) < 9):
            print('The length of the landmark is too short, skip')
            return

        # Step 3: raw save landmark / audio
        fl3d = np.array(fl3d)
        np.savetxt(os.path.join(self.out_dir, 'raw_fl3d/fan_{:05d}_{}_3d.txt'.format(self.idx, self.video_name[:-4])),
                   fl3d, fmt='%.2f')
        if (save_audio):
            self.__save_audio__(video_dir=self.video_dir.replace(
                self.input_format, '_preclean.mp4'), fl3d=fl3d)

        # Step 3.5: merge a/v together (optional)
        if (show):
            sf, ef = (fl3d[0][0], fl3d[-1][0]) if fl3d.shape[0] > 0 else (0, 0)
            print(sf, ef)
            print(self.video_dir.replace(self.input_format, '_fl_detect.mp4'),
                  os.path.join(self.out_dir, 'tmp_v', '{:05d}_{}_fl_av.mp4'.format(
                      self.idx, self.video_name[:-4]))
                  )
            self.__ffmpeg_merge_av__(
                video_dir=self.video_dir.replace(
                    self.input_format, '_fl_detect.mp4'),
                audio_dir=self.video_dir.replace(
                    self.input_format, '_preclean.mp4'),
                WriteFileName=os.path.join(self.out_dir, 'tmp_v', '{:05d}_{}_fl_av.mp4'.format(
                    self.idx, self.video_name[:-4])),
                start_end_frame=(int(sf), int(ef)))

        # Step 4: remove tmp files
        os.remove(self.video_dir.replace(self.input_format, '_preclean.mp4'))
        if(os.path.isfile(self.video_dir.replace(self.input_format, '_fl_detect.mp4'))):
            os.remove(self.video_dir.replace(
                self.input_format, '_fl_detect.mp4'))

        # Step 5: register fl3d
        if (register):
            self.__single_landmark_3d_register__(fl3d)
            # TODO: visualize register fl3d

    ''' ========================================================================

                            STEP 1: Preclean video

    ======================================================================== '''

    def __preclean_video__(self, WriteFileName='_preclean.mp4', fps=25, sample_rate=16000):
        '''
        Pre-clean downloaded videos. Return false if more than 2 streams found.
        Then convert it to fps=25, sample_rate=16kHz
        '''
        input_video_dir = self.video_dir if '_x_' not in self.video_dir else self.video_dir.replace(
            '_x_', '/')
        probe = ffmpeg.probe(input_video_dir)
        # print(probe['streams'])
        # print(len(probe['streams']))
        # if(len(probe['streams']) != 2):
        #     print('Error: not valid for # of a/v channel == 2.')
        #     return False, None
        # exit(0)
        # probe['streams'] = probe['streams'][0::2]

        codec = {'video': '', 'audio': ''}
        for i, stream in enumerate(probe['streams'][0:2]):
            codec[stream['codec_type']] = stream['codec_name']

        # create preclean video
        (
            ffmpeg
            .input(input_video_dir)
            .output(self.video_dir.replace(self.input_format, WriteFileName),
                    # vcodec=codec['video'],
                    # acodec=codec['audio'],
                    r=fps, ar=sample_rate)
            .overwrite_output().global_args('-loglevel', 'quiet')
            .run()
        )

        return True, self.video_dir.replace(self.input_format, WriteFileName)

    ''' ========================================================================

                       STEP 2: Detect facial landmark

    ======================================================================== '''

    def __video_facial_landmark_detection__(self, video_dir=None, display=False, WriteFileName='_fl_detect.mp4',
                                            max_num_frames=250, write=False):
        '''
        Get facial landmark from video.
        '''

        # load video
        print('video_dir : ' + video_dir)
        video = cv2.VideoCapture(video_dir)

        # return false if cannot open
        if (video.isOpened() == False):
            print('Unable to open video file')
            return False, None

        # display info
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('Process Video {}, len: {}, FPS: {:.2f}, W X H: {} x {}'.format(
            video_dir, length, fps, w, h))

        if(write):
            writer = cv2.VideoWriter(self.video_dir.replace(self.input_format, WriteFileName),
                                     cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (w, h))

        video_facial_landmark = []  # face-landmark np array per frame =: idx + [x,y] * 68
        # face-landmark np array per frame =: idx + [x,y,z] * 68
        video_facial_landmark_3d = []
        frame_id = 0
        not_detected_frames = 0

        while (video.isOpened()):
            ret, frame = video.read()
            # reach EOF
            if (ret == False):
                break

            # too many not-detected frames (in middle of video)
            if (not_detected_frames > 5):
                if (len(video_facial_landmark) < 10):
                    # at beginning of the video
                    video_facial_landmark = []
                    video_facial_landmark_3d = []
                else:
                    break

            # dlib facial landmark detect
            img_ret, shape, shape_3d = self.__image_facial_landmark_detection__(
                img=frame)

            # successfully detected
            if (img_ret):
                # print('\t ==> frame {}/{}'.format(frame_id, length))

                # current frame xy coordinates
                xys = []
                for part_i in range(68):
                    xys.append(shape.part(part_i).x)
                    xys.append(shape.part(part_i).y)

                # check any not_detected_frames, and interp them
                if (not_detected_frames > 0 and len(video_facial_landmark) > 0):
                    # interpolate
                    def interp(last, cur, num, dims=68 * 2 + 1):
                        interp_xys_np = np.zeros((num, dims))
                        for dim in range(dims):
                            interp_xys_np[:, dim] = np.interp(
                                np.arange(0, num), [-1, num], [last[dim], cur[dim]])
                        interp_xys_np = np.round(interp_xys_np).astype('int')
                        interp_xys = [list(xy) for xy in interp_xys_np]
                        return interp_xys

                    interp_xys = interp(
                        video_facial_landmark[-1], [frame_id] + xys, not_detected_frames)
                    video_facial_landmark += interp_xys

                not_detected_frames = 0

                # save landmark/frame_index
                video_facial_landmark.append([frame_id] + xys)
                if (shape_3d.any()):
                    video_facial_landmark_3d.append(
                        [frame_id] + list(np.reshape(shape_3d, -1)))

                if(write):
                    frame = self.__vis_landmark_on_img__(frame, shape)

            else:
                print('\t ==> frame {}/{} Not detected'.format(frame_id, length))
                not_detected_frames += 1

            if (display):
                cv2.imshow('Frame', frame)
                if (cv2.waitKey(10) == ord('q')):
                    break

            if(write):
                writer.write(frame)
            frame_id += 1

            if(frame_id > max_num_frames):
                break

        video.release()
        if(write):
            writer.release()
        cv2.destroyAllWindows()

        print('\t ==> Final processed frames {}/{}'.format(frame_id, length))

        return True, video_facial_landmark, video_facial_landmark_3d

    def __image_facial_landmark_detection__(self, img=None):
        '''
        Get facial landmark from single image by FANet
        '''

        shapes = self.predictor.get_landmarks(img)
        if (not shapes):
            return False, None, None

        max_size_idx = 0
        shape = ShapeParts(shapes[max_size_idx][:, 0:2])
        shape_3d = shapes[max_size_idx]

        # when use 2d estimator
        shape_3d = np.concatenate([shape_3d, np.ones(shape=(68, 1))], axis=1)

        return True, shape, shape_3d

    def __vis_landmark_on_img__(self, img, shape, linewidth=2):
        '''
        Visualize landmark on images.
        '''
        if (type(shape) == ShapeParts):
            def draw_curve(idx_list, color=(0, 255, 0), loop=False, lineWidth=linewidth):
                for i in idx_list:
                    cv2.line(img, (shape.part(i).x, shape.part(i).y), (shape.part(i + 1).x, shape.part(i + 1).y),
                             color, lineWidth)
                if (loop):
                    cv2.line(img, (shape.part(idx_list[0]).x, shape.part(idx_list[0]).y),
                             (shape.part(idx_list[-1] + 1).x, shape.part(idx_list[-1] + 1).y), color, lineWidth)

            draw_curve(list(range(0, 16)))  # jaw
            draw_curve(list(range(17, 21)))  # eye brow
            draw_curve(list(range(22, 26)))
            draw_curve(list(range(27, 35)))  # nose
            draw_curve(list(range(36, 41)), loop=True)  # eyes
            draw_curve(list(range(42, 47)), loop=True)
            draw_curve(list(range(48, 59)), loop=True)  # mouth
            draw_curve(list(range(60, 67)), loop=True)

        else:
            def draw_curve(idx_list, color=(0, 255, 0), loop=False, lineWidth=linewidth):
                for i in idx_list:
                    cv2.line(img, (shape[i, 0], shape[i, 1]),
                             (shape[i + 1, 0], shape[i + 1, 1]), color, lineWidth)
                if (loop):
                    cv2.line(img, (shape[idx_list[0], 0], shape[idx_list[0], 1]),
                             (shape[idx_list[-1] + 1, 0], shape[idx_list[-1] + 1, 1]), color, lineWidth)

            draw_curve(list(range(0, 16)))  # jaw
            draw_curve(list(range(17, 21)))  # eye brow
            draw_curve(list(range(22, 26)))
            draw_curve(list(range(27, 35)))  # nose
            draw_curve(list(range(36, 41)), loop=True)  # eyes
            draw_curve(list(range(42, 47)), loop=True)
            draw_curve(list(range(48, 59)), loop=True)  # mouth
            draw_curve(list(range(60, 67)), loop=True)

        return img

    def __ffmpeg_merge_av__(self, video_dir, audio_dir, WriteFileName, start_end_frame):
        probe = ffmpeg.probe(video_dir)
        fps = probe['streams'][0]['avg_frame_rate']
        spf = float(fps.split('/')[1]) / float(fps.split('/')[0])
        sf, ef = start_end_frame
        st, tt = sf * spf, ef * spf - sf * spf

        vin = ffmpeg.input(video_dir).video
        # ain = ffmpeg.input(audio_dir).audio
        # out = ffmpeg.output(vin, ain, WriteFileName, codec='copy', ss=st, t=tt, shortest=None)
        out = ffmpeg.output(vin, WriteFileName, codec='copy',
                            ss=st, t=tt, shortest=None)
        out = out.overwrite_output().global_args('-loglevel', 'quiet')
        out.run()

        # os.system('ffmpeg -i {} -codec copy -ss {} -t {} {}'.format(video_dir, st, tt, WriteFileName))

    def __save_audio__(self, video_dir, fl3d):
        """
        Extract audio from preclean video. Used for creating audio-aware dataset.

        """
        sf, ef = fl3d[0][0], fl3d[-1][0]

        probe = ffmpeg.probe(video_dir)
        fps = probe['streams'][0]['avg_frame_rate']
        spf = float(fps.split('/')[1]) / float(fps.split('/')[0])
        st, tt = sf * spf, ef * spf - sf * spf

        audio_dir = os.path.join(self.out_dir, 'raw_wav', '{:05d}_{}_audio.wav'.format(
            self.idx, self.video_name[:-4]))
        (
            ffmpeg
            .input(video_dir)
            .output(audio_dir, ss=st, t=tt)
            .overwrite_output().global_args('-loglevel', 'quiet')
            .run()
        )

    ''' ========================================================================

                            STEP 5: Landmark register

    ======================================================================== '''

    def __single_landmark_3d_register__(self, fl3d, display=False):
        """
        Register a single 3d landmark file

        """
        # Step 1 : Load and Smooth
        from scipy.signal import savgol_filter
        lines = savgol_filter(fl3d, 7, 3, axis=0)

        all_landmarks = lines[:, 1:].reshape((-1, 68, 3))  # remove frame idx
        w, h = int(np.max(all_landmarks[:, :, 0])) + \
            20, int(np.max(all_landmarks[:, :, 1])) + 20

        # Step 2 : setup anchor face
        print('Using exisiting ' +
              'dataset/utils/ANCHOR_T_SHAPE_{}.txt'.format(len(self.t_shape_idx)))
        anchor_t_shape = np.loadtxt(
            'dataset/utils/ANCHOR_T_SHAPE_{}.txt'.format(len(self.t_shape_idx)))

        registered_landmarks_to_save = []
        registered_affine_mat_to_save = []
        # for each line
        for line in lines:
            frame_id = line[0]
            landmarks = line[1:].reshape(68, 3)

            # Step 3 : ICP on (frame, anchor)
            frame_t_shape = landmarks[self.t_shape_idx, :]

            T, distance, itr = icp(frame_t_shape, anchor_t_shape)

            # Step 4 : Affine transform
            landmarks = np.hstack((landmarks, np.ones((68, 1))))
            registered_landmarks = np.dot(T, landmarks.T).T
            err = np.mean(np.sqrt(np.sum(
                (registered_landmarks[self.t_shape_idx, 0:3] - anchor_t_shape) ** 2, axis=1)))
            # print(err, distance, itr)

            # Step 5 : Save is requested
            registered_landmarks_to_save.append(
                [frame_id] + list(registered_landmarks[:, 0:3].reshape(-1)))
            registered_affine_mat_to_save.append(
                [frame_id] + list(T.reshape(-1)))

            # Step 5.5 (optional) : visualize ori / registered faces (Isolated in Black BG)
            if (display):
                img = np.zeros((h, w * 2, 3), np.uint8)
                self.__vis_landmark_on_img__(img, landmarks.astype(np.int))
                registered_landmarks[:, 0] += w
                self.__vis_landmark_on_img__(
                    img, registered_landmarks.astype(np.int))
                cv2.imshow('img', img)
                if (cv2.waitKey(30) == ord('q')):
                    break

        np.savetxt(os.path.join(self.out_dir, 'register_fl3d', '{:05d}_{}_fl_sm.txt'
                                .format(self.idx, self.video_name[:-4])),
                   lines, fmt='%.6f')
        np.savetxt(os.path.join(self.out_dir, 'register_fl3d', '{:05d}_{}_fl_reg.txt'
                                .format(self.idx, self.video_name[:-4])),
                   np.array(registered_landmarks_to_save), fmt='%.6f')
        np.savetxt(os.path.join(self.out_dir, 'register_fl3d', '{:05d}_{}_mat_reg.txt'
                                .format(self.idx, self.video_name[:-4])),
                   np.array(registered_affine_mat_to_save), fmt='%.6f')


if __name__ == '__main__':
    video_dir = r'mp4/id00154/6eX78POXvNg/00004.mp4'
    out_dir = r'mp4/id00154/'
    c = Av2Flau_Convertor(video_dir, out_dir, idx=0)
    c.convert()
