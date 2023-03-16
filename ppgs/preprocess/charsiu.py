import ppgs
import pypar
import tqdm
from shutil import copy as cp

def charsiu(input_dir, output_dir, features=None, gpu=None):
    """Perform preprocessing for charsiu dataset"""

    print('input_dir:', input_dir)
    print('output_dir:', output_dir)

    wav_dir = input_dir / 'wav'
    if not wav_dir.exists():
        wav_dir = input_dir
    textgrid_dir = input_dir / 'textgrid'
    if not textgrid_dir.exists():
        textgrid_dir = input_dir

    output_dir.mkdir(exist_ok=True, parents=True)

    audio_files = list(wav_dir.glob('*.wav'))

    with ppgs.data.chdir(output_dir):

        if 'phonemes' in features: #convert textgrid and transfer
            # raise NotImplementedError('phoneme preprocessing for charsiu not fully implemented')
            textgrid_files = list(textgrid_dir.glob('*.textgrid')) + list(textgrid_dir.glob('*.TextGrid'))
            iterator = tqdm.tqdm(
                textgrid_files,
                desc="Converting textgrid phone dialect for charsiu dataset",
                total=len(textgrid_files),
                dynamic_ncols=True
            )
            for textgrid_file in iterator:
                alignment = pypar.Alignment(textgrid_file)
                for word in alignment._words:
                    if word.word == '[SIL]':
                        word.word = 'sp'
                    for phoneme in word.phonemes:
                        if phoneme.phoneme == '[SIL]':
                            phoneme.phoneme = 'sil'
                        else:
                            phoneme.phoneme = phoneme.phoneme.lower()
                alignment.save(textgrid_file.stem + '.textgrid')

        if 'wav' in features: #copy wav files
            iterator = tqdm.tqdm(
                audio_files,
                desc="copying audio files",
                total=len(audio_files),
                dynamic_ncols=True
            )
            for audio_file in iterator:
                cp(audio_file, audio_file.name)

        if 'senone' in features: #compute ppgs
            ppg_files = [f'{file.stem}-senone.pt' for file in audio_files]
            ppgs.preprocess.senone.from_files_to_files(
                audio_files,
                ppg_files,
                gpu=gpu
            )

        if 'w2v2fs' in features: #compute w2v2 latents
            audio_files = audio_files
            w2v2_files = [f'{file.stem}-w2v2fs.pt' for file in audio_files]
            ppgs.preprocess.w2v2fs.from_files_to_files(
                audio_files,
                w2v2_files,
                gpu=gpu
            )
