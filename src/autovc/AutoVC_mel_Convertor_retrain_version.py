from scipy.signal import stft
from scipy.io import wavfile as wav
import pynormalize.pynormalize
from pydub import AudioSegment
from src.autovc.retrain_version.model_vc_37_1 import Generator
from math import ceil
import torch
import pickle
import numpy as np
import os


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


class AutoVC_mel_Convertor():

    def __init__(self, src_dir, proportion=(0., 1.), seed=0):
        self.src_dir = src_dir
        if(not os.path.exists(os.path.join(src_dir, 'filename_index.txt'))):
            self.filenames = []
        else:
            with open(os.path.join(src_dir, 'filename_index.txt'), 'r') as f:
                lines = f.readlines()
                self.filenames = [(int(line.split(' ')[0]), line.split(' ')[
                                   1][:-1]) for line in lines]

        np.random.seed(seed)
        rand_perm = np.random.permutation(len(self.filenames))
        proportion_idx = (
            int(proportion[0] * len(rand_perm)), int(proportion[1] * len(rand_perm)))
        selected_index = rand_perm[proportion_idx[0]: proportion_idx[1]]
        self.selected_filenames = [self.filenames[i] for i in selected_index]

        print('{} out of {} are in this portion'.format(
            len(self.selected_filenames), len(self.filenames)))

    def __convert_single_only_au_AutoVC_format_to_dataset__(self, filename, build_train_dataset=True):
        """
        Convert a single file (only audio in AutoVC embedding format) to numpy arrays
        :param filename:
        :param is_map_to_std_face:
        :return:
        """

        global_clip_index, video_name = filename

        # audio_file = os.path.join(self.src_dir, 'raw_wav', '{}.wav'.
        #                           format(video_name[:-4]))
        audio_file = os.path.join(self.src_dir, 'raw_wav', '{:05d}_{}_audio.wav'.
                                  format(global_clip_index, video_name[:-4]))
        if(not build_train_dataset):
            import shutil
            audio_file = os.path.join(self.src_dir, 'raw_wav', '{:05d}_{}_audio.wav'.
                                      format(global_clip_index, video_name[:-4]))
            shutil.copy(os.path.join(
                self.src_dir, 'test_wav_files', video_name), audio_file)

        sound = AudioSegment.from_file(audio_file, "wav")
        normalized_sound = match_target_amplitude(sound, -20.0)
        normalized_sound.export(audio_file, format='wav')

        from src.autovc.retrain_version.vocoder_spec.extract_f0_func import extract_f0_func_audiofile
        S, f0_norm = extract_f0_func_audiofile(audio_file, 'M')

        from src.autovc.utils import quantize_f0_interp
        f0_onehot = quantize_f0_interp(f0_norm)

        from thirdparty.resemblyer_util.speaker_emb import get_spk_emb
        mean_emb, _ = get_spk_emb(audio_file)

        return S, mean_emb, f0_onehot

    def convert_wav_to_autovc_input(self, build_train_dataset=True, autovc_model_path=r'E:\Dataset\VCTK\stargan_vc\train_85_withpre1125000_local\360000-G.ckpt'):

        def pad_seq(x, base=32):
            len_out = int(base * ceil(float(x.shape[0]) / base))
            len_pad = len_out - x.shape[0]
            assert len_pad >= 0
            return np.pad(x, ((0, len_pad), (0, 0)), 'constant'), len_pad

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        G = Generator(16, 256, 512, 16).eval().to(device)
        g_checkpoint = torch.load(autovc_model_path, map_location=device)
        G.load_state_dict(g_checkpoint['model'])

        emb = np.loadtxt('autovc/retrain_version/obama_emb.txt')
        emb_trg = torch.from_numpy(
            emb[np.newaxis, :].astype('float32')).to(device)

        aus = []

        for i, file in enumerate(self.selected_filenames):
            print(i, file)
            x_real_src, emb, f0_org_src = self.__convert_single_only_au_AutoVC_format_to_dataset__(
                filename=file, build_train_dataset=build_train_dataset)

            '''# normal length #'''
            # with torch.no_grad():
            #     x_identic, x_identic_psnt, code_real = G(x_real, emb_org, f0_org, emb_trg, f0_org)
            #     g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt, reduction='sum')
            #     print('loss:', g_loss_id_psnt / x_identic_psnt.shape[1] * 128)

            ''' too long split length '''
            l = x_real_src.shape[0]
            x_identic_psnt = []
            step = 4096
            for i in range(0, l, step):
                x_real = x_real_src[i:i+step]
                f0_org = f0_org_src[i:i+step]

                x_real, len_pad = pad_seq(x_real.astype('float32'))
                f0_org, _ = pad_seq(f0_org.astype('float32'))
                x_real = torch.from_numpy(
                    x_real[np.newaxis, :].astype('float32')).to(device)
                emb_org = torch.from_numpy(
                    emb[np.newaxis, :].astype('float32')).to(device)
                # emb_trg = torch.from_numpy(emb[np.newaxis, :].astype('float32')).to(device)
                f0_org = torch.from_numpy(
                    f0_org[np.newaxis, :].astype('float32')).to(device)

                print('source shape:', x_real.shape,
                      emb_org.shape, emb_trg.shape, f0_org.shape)

                with torch.no_grad():
                    x_identic, x_identic_psnt_i, code_real = G(
                        x_real, emb_org, f0_org, emb_trg, f0_org)
                    x_identic_psnt.append(x_identic_psnt_i)

            x_identic_psnt = torch.cat(x_identic_psnt, dim=1)
            print('converted shape:', x_identic_psnt.shape, code_real.shape)
            if len_pad == 0:
                uttr_trg = x_identic_psnt[0, :, :].cpu().numpy()
            else:
                uttr_trg = x_identic_psnt[0, :-len_pad, :].cpu().numpy()

            # ''' plot source and converted mel-spec figures '''
            # import matplotlib.pyplot as plt
            # plt.subplot(1, 2, 1)
            # plt.imshow(x_real_src[0:200, :])
            # plt.subplot(1, 2, 2)
            # plt.imshow(uttr_trg[0:200, :])
            # plt.show()
            #
            # exit(0)

            file = (file[0], file[1], emb)
            aus.append((uttr_trg, file))

        return aus

    def convert_single_wav_to_input(self, audio_filename):
        aus = []
        audio_file = os.path.join(self.src_dir, 'demo_wav', audio_filename)

        # Default param
        TARGET_AUDIO_DBFS = -20.0
        WAV_STEP = int(0.2 * 16000)  # 0.2s = 5 frames
        STFT_WINDOW_SIZE = {'25': 320, '29.97': 356}
        STFT_WINDOW_STEP = {'25': 4, '29.97': 3}
        FPS = 25

        # Step 1 : Normalize the volume
        target_dbfs = TARGET_AUDIO_DBFS
        pynormalize.process_files(
            Files=[audio_file],
            target_dbfs=target_dbfs,
            directory=os.path.join(self.src_dir, 'raw_wav')
        )

        #  Step 2 : load wav file
        sample_rate, samples = wav.read(audio_file)
        assert (sample_rate == 16000)
        if (len(samples.shape) > 1):
            samples = samples[:, 0]  # pick mono

        # Step 3 : STFT,
        # 1 frame = 1/25 * 16k = 640 samples => windowsize=320,  overlap=160
        # 1 frame = 1/29.97 * 16k = 533.86 samples => windowsize=356, overlap=178, (mis-align = 4.2sample / 1s)
        f, t, Zxx = stft(samples, fs=sample_rate,
                         nperseg=STFT_WINDOW_SIZE[str(FPS)])

        # stft_abs = np.abs(Zxx)
        stft_abs = np.log(np.abs(Zxx) ** 2 + 1e-10)
        stft_abs_max = np.max(stft_abs)
        stft_abs /= stft_abs_max

        # Step 4 : align AV (drop last 2 frames of V)
        fl_length = stft_abs.shape[1] // STFT_WINDOW_STEP[str(FPS)]
        audio_stft_length = (fl_length - 2) * STFT_WINDOW_STEP[str(FPS)]
        stft_signal = Zxx[:, 0:audio_stft_length]
        stft_abs = stft_abs[:, 0:audio_stft_length]

        audio_wav_length = int((fl_length - 2) * sample_rate / FPS)
        wav_signal = samples[0:audio_wav_length]

        # # Step 6 : Save audio
        # info_audio = (0, stft_signal, fl_length - 2, audio_stft_length, audio_wav_length)
        # au_data = (stft_abs, wav_signal, info_audio)

        aus.append((stft_abs.T, None, (0, audio_filename, 0)))

        return aus

    def convert_single_wav_to_autovc_input(self, audio_filename, autovc_model_path):

        def pad_seq(x, base=32):
            len_out = int(base * ceil(float(x.shape[0]) / base))
            len_pad = len_out - x.shape[0]
            assert len_pad >= 0
            return np.pad(x, ((0, len_pad), (0, 0)), 'constant'), len_pad

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        G = Generator(16, 256, 512, 16).eval().to(device)

        g_checkpoint = torch.load(autovc_model_path, map_location=device)
        G.load_state_dict(g_checkpoint['model'])

        emb = np.loadtxt('src/autovc/retrain_version/obama_emb.txt')
        emb_trg = torch.from_numpy(
            emb[np.newaxis, :].astype('float32')).to(device)

        aus = []
        audio_file = audio_filename

        sound = AudioSegment.from_file(audio_file, "wav")
        normalized_sound = match_target_amplitude(sound, -20.0)
        normalized_sound.export(audio_file, format='wav')

        from src.autovc.retrain_version.vocoder_spec.extract_f0_func import extract_f0_func_audiofile
        x_real_src, f0_norm = extract_f0_func_audiofile(audio_file, 'F')
        from src.autovc.utils import quantize_f0_interp
        f0_org_src = quantize_f0_interp(f0_norm)
        from thirdparty.resemblyer_util.speaker_emb import get_spk_emb
        emb, _ = get_spk_emb(audio_file)

        ''' normal length version '''
        # x_real, len_pad = pad_seq(x_real_src.astype('float32'))
        # f0_org, _ = pad_seq(f0_org_src.astype('float32'))
        # x_real = torch.from_numpy(x_real[np.newaxis, :].astype('float32')).to(device)
        # emb_org = torch.from_numpy(emb[np.newaxis, :].astype('float32')).to(device)
        # f0_org = torch.from_numpy(f0_org[np.newaxis, :].astype('float32')).to(device)
        # print('source shape:', x_real.shape, emb_org.shape, emb_trg.shape, f0_org.shape)

        # with torch.no_grad():
        #     x_identic, x_identic_psnt, code_real = G(x_real, emb_org, f0_org, emb_trg, f0_org)
        # print('converted shape:', x_identic_psnt.shape, code_real.shape)

        ''' long split version '''
        l = x_real_src.shape[0]
        x_identic_psnt = []
        step = 4096
        for i in range(0, l, step):
            x_real = x_real_src[i:i + step]
            f0_org = f0_org_src[i:i + step]

            x_real, len_pad = pad_seq(x_real.astype('float32'))
            f0_org, _ = pad_seq(f0_org.astype('float32'))
            x_real = torch.from_numpy(
                x_real[np.newaxis, :].astype('float32')).to(device)
            emb_org = torch.from_numpy(
                emb[np.newaxis, :].astype('float32')).to(device)
            # emb_trg = torch.from_numpy(emb[np.newaxis, :].astype('float32')).to(device)
            f0_org = torch.from_numpy(
                f0_org[np.newaxis, :].astype('float32')).to(device)
            print('source shape:', x_real.shape,
                  emb_org.shape, emb_trg.shape, f0_org.shape)

            with torch.no_grad():
                x_identic, x_identic_psnt_i, code_real = G(
                    x_real, emb_org, f0_org, emb_trg, f0_org)
                x_identic_psnt.append(x_identic_psnt_i)

        x_identic_psnt = torch.cat(x_identic_psnt, dim=1)
        print('converted shape:', x_identic_psnt.shape, code_real.shape)

        if len_pad == 0:
            uttr_trg = x_identic_psnt[0, :, :].cpu().numpy()
        else:
            uttr_trg = x_identic_psnt[0, :-len_pad, :].cpu().numpy()

        aus.append((uttr_trg, (0, audio_filename, emb)))

        return aus


if __name__ == '__main__':
    c = AutoVC_mel_Convertor(
        r'PreprocessedVox_imagetranslation', proportion=(0.0, 1.0))
    aus = c.convert_wav_to_autovc_input()
    print(aus)
    # c = AutoVC_mel_Convertor(
    #     r'E:\Dataset\TalkingToon\Obama_for_train', proportion=(0.0, 1.0))
    # aus = c.convert_wav_to_autovc_input()

    # with open(os.path.join(r'E:\Dataset\TalkingToon\Obama_for_train', 'dump', 'autovc_retrain_mel_au.pickle'), 'wb') as fp:
    #     pickle.dump(aus, fp)
