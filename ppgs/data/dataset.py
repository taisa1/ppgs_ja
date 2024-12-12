import json
import warnings
from pathlib import Path

# import accelerate
import numpy as np
import pypar
import torch
import torchaudio

import ppgs


###############################################################################
# Dataset
###############################################################################

def get_phoneme_from_file(file_path, ts, phoneme_map):
    phoneme_data = []
    
    # ファイルを読み込む
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                start_time = float(parts[0])
                end_time = float(parts[1])
                phoneme = parts[2]
                phoneme_data.append((start_time, end_time, phoneme))
    res = []
    # 指定時刻に対応する音素を検索
    for t in ts:
        for start_time, end_time, phoneme in phoneme_data:
            if start_time <= t < end_time:
                if str(phoneme) not in ppgs.PHONEMES:
                    res.append(phoneme_map[pypar.SILENCE])
                else:
                    res.append(phoneme_map[str(phoneme)])
                break
    
    return res

def get_duration_from_file(file_path):
       # ファイルを読み込み
    data = np.loadtxt(file_path, usecols=(0, 1))
    
    # 最後の行の2列目 - 最初の行の1列目を計算
    difference = data[-1, 1] - data[0, 0]
    return difference

class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        name_or_files,
        partition=None,
        features=['audio'],
        max_frames=25000):
        self.features = features
        self.metadata = Metadata(
            name_or_files,
            partition=partition,
            max_frames=max_frames)
        self.cache = self.metadata.cache
        self.stems = self.metadata.stems
        self.audio_files = self.metadata.audio_files
        self.lengths = self.metadata.lengths

    def __getitem__(self, index):
        """Retrieve the indexth item"""
        stem = self.stems[index]
        feature_values = []
        if isinstance(self.features, str):
            self.features = [self.features]
        for feature in self.features:

            # Load audio
            if feature == 'audio':
                audio = ppgs.load.audio(self.audio_files[index])
                feature_values.append(audio)

            # Load phoneme alignment
            elif feature == 'phonemes':

                # Convert to indices
                hopsize = ppgs.HOPSIZE / 24000
                num_frames = self.metadata.lengths[index]
                times = np.linspace(
                    hopsize / 2,
                    (num_frames - 1) * hopsize + hopsize / 2,
                    num_frames)
                file_path = self.cache / f'{stem}.lab'
                times[-1] = get_duration_from_file(file_path)
                indices = get_phoneme_from_file(file_path, times, ppgs.PHONEME_TO_INDEX_MAPPING)
                indices = torch.tensor(indices, dtype=torch.long)
                lim = indices.shape[-1]
                new_repr = feature_values[-1][:,:lim]
                feature_values[-1] = new_repr
                #print(stem,feature_values[-1].shape)
                feature_values.append(indices)

            # Add stem
            elif feature == 'stem':
                feature_values.append(stem)

            # Add filename
            elif feature == 'audio_file':
                feature_values.append(self.audio_files[index])

            # Add length
            elif feature == 'length':
                try:
                    feature_values.append(feature_values[-1].shape[-1])
                  #  print(stem,feature_values[-1])
                except AttributeError:
                    feature_values.append(len(feature_values[-1]))

            # Add input representation
            else:
               # print(stem,torch.load(self.cache / f'{stem}-{feature}.pt').shape)
                feature_values.append(
                    torch.load(self.cache / f'{stem}-{feature}.pt'))
        return feature_values

    def __len__(self):
        """Length of the dataset"""
        return len(self.stems)

    def buckets(self):
        """Partition indices into buckets based on length for sampling"""
        # Get the size of a bucket
        size = len(self) // ppgs.BUCKETS

        # Get indices in order of length
        indices = np.argsort(self.lengths)
        lengths = np.sort(self.lengths)

        # Split into buckets based on length
        buckets = [
            np.stack((indices[i:i + size], lengths[i:i + size])).T
            for i in range(0, len(self), size)]

        # Concatenate partial bucket
        if len(buckets) == ppgs.BUCKETS + 1:
            residual = buckets.pop()
            buckets[-1] = np.concatenate((buckets[-1], residual), axis=0)

        return buckets


###############################################################################
# Utilities
###############################################################################


class Metadata:

    def __init__(
        self,
        name_or_files,
        partition=None,
        overwrite_cache=False,
        max_frames=25000):
        """Create a metadata object for the given dataset or sources"""
        lengths = {}

        # Create dataset from string identifier
        if isinstance(name_or_files, str):
            self.name = name_or_files
            self.cache = ppgs.CACHE_DIR / self.name

            # Get stems corresponding to partition
            partition_dict = ppgs.load.partition(self.name)
            if partition is not None:
                self.stems = partition_dict[partition]
                lengths_file = self.cache / f'{partition}-lengths.json'
            else:
                self.stems = sum(partition_dict.values(), start=[])
                lengths_file = self.cache / f'lengths.json'

            # Get audio filenames
            self.audio_files = [
                self.cache / (stem + '.wav') for stem in self.stems]

            # Maybe remove previous cached lengths
            if overwrite_cache:
                lengths_file.unlink(missing_ok=True)

            # Load cached lengths
            if lengths_file.exists():
                with open(lengths_file, 'r') as f:
                    lengths = json.load(f)

        # Create dataset from a list of audio filenames
        else:
            self.name = '<list of files>'
            self.audio_files = name_or_files
            self.stems = [
                Path(file).parent / Path(file).stem
                for file in self.audio_files]
            self.cache = None

        if not lengths:
            t = 0
            # Compute length in frames
            for stem, audio_file in zip(self.stems, self.audio_files):
                info = torchaudio.info(audio_file)
               
                length = int(
                    info.num_frames * (24000 / info.sample_rate)
                ) // ppgs.HOPSIZE

                # Omit if length is too long to avoid OOM
                if length <= max_frames:
                    lengths[stem] = length
                else:
                    warnings.warn(
                        f'File {audio_file} of length {length} '
                        f'exceeds max_frames of {max_frames}. Skipping.')

            # Maybe cache lengths
            if self.cache is not None:
                with open(lengths_file, 'w+') as file:
                    json.dump(lengths, file)
        # Match ordering
        (
            self.audio_files,
            self.stems,
            self.lengths
            ) = zip(*[
            (file, stem, lengths[stem])
            for file, stem in zip(self.audio_files, self.stems)
            if stem in lengths
        ])

    def __len__(self):
        return len(self.stems)
