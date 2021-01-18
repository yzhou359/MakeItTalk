from resemblyzer import preprocess_wav, VoiceEncoder
import numpy as np
import torch


def get_spk_emb(audio_file_dir, segment_len=960000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resemblyzer_encoder = VoiceEncoder(device=device)

    wav = preprocess_wav(audio_file_dir)
    l = len(wav) // segment_len  # segment_len = 16000 * 60
    l = np.max([1, l])
    all_embeds = []
    for i in range(l):
        mean_embeds, cont_embeds, wav_splits = resemblyzer_encoder.embed_utterance(
            wav[segment_len * i:segment_len * (i + 1)], return_partials=True, rate=2)
        all_embeds.append(mean_embeds)
    all_embeds = np.array(all_embeds)
    mean_embed = np.mean(all_embeds, axis=0)

    return mean_embed, all_embeds


if __name__ == '__main__':
    m, a = get_spk_emb(r'examples/tmp.wav')
    # m, a = get_spk_emb(r'E:\Dataset\TalkingToon\Obama\test_wav_files\obama_example.wav')
    # print('Speaker embedding:', m)
    # print(m.shape)
