from typing import Dict, List
from torch import Tensor
from numpy import ndarray
from numpy.random.mtrand import RandomState
from pretty_midi.pretty_midi import PrettyMIDI

import os
import json
import numpy as np
import soundfile
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pretty_midi

from constants import HOP_SIZE, MAX_MIDI, MIN_MIDI, SAMPLE_RATE

def allocate_batch(batch:Dict[str,Tensor], device:torch.device) -> Dict[str,Tensor]:
    for key in batch.keys():
        if key != 'path':
            batch[key] = batch[key].to(device)
    return batch

class MAESTRO_small(Dataset):
    def __init__(self,
                 path:str = 'data',
                 groups:List[str] = None,
                 sequence_length:int = SAMPLE_RATE * 5,
                 hop_size:int = HOP_SIZE,
                 seed:int = 42,
                 random_sample:bool = True) -> None:
        self.path:str = path
        self.groups:list = groups if groups is not None else self.available_groups()
        assert all(group in self.available_groups() for group in self.groups)

        self.sample_length:int = ((sequence_length // hop_size) * hop_size) if sequence_length is not None else None
        self.random:RandomState = np.random.RandomState(seed)
        self.random_sample:bool = random_sample
        self.hop_size:int = hop_size

        self.data:List[Dict[str,Tensor]] = []

        print(f'Loading {len(groups)} group(s) of', self.__class__.__name__, 'at', path)
        for group in groups:
            file_list:List[tuple] = self.get_file_path_list_of_group(group)
            for input_files in tqdm(file_list, desc=f'Loading group {group}'):
                self.data.append(self.load(*input_files))

    @classmethod
    def available_groups(cls) -> List[str]:
        return ['train', 'validation', 'test', 'debug']    

    def get_file_path_list_of_group(self, group:str) -> List[tuple]:
        metadata:List[dict] = json.load(open(os.path.join(self.path, 'data.json')))
        subset_name:str = 'train' if group == 'debug' else group

        files:List[tuple] = sorted([
                (os.path.join(self.path, row['directory'], 'orchestra.mid'),
                 os.path.join(self.path, row['directory'], 'piano.mid'))
                for row in metadata if row['split'] == subset_name
            ])

        if group == 'debug':
            files = files[:10]
        else:
            files = [(audio if os.path.exists(audio) else audio.replace(
                '.flac', '.wav'), midi) for audio, midi in files]

        return files
    
    def load(self, input_path:str, output_path:str) -> Dict[str,Tensor]:
        input_midi:PrettyMIDI = pretty_midi.PrettyMIDI(input_path)
        output_midi:PrettyMIDI = pretty_midi.PrettyMIDI(output_path)

        input_dict = self.midi_to_frame_and_onset(input_midi)
        output_dict = self.midi_to_frame_and_onset(output_midi)

        data = dict(path=input_path, frame_input=input_dict['frame'], onset_input=input_dict['onset'], frame_output=output_dict['frame'], onset_output=output_dict['onset'])
        return data

    def midi_to_frame_and_onset(self, midi:PrettyMIDI) -> Dict[str, Tensor]:
        frames_per_sec:float = 31.25 # todo: 정해야 함.
        frame:ndarray = midi.get_piano_roll(fs=frames_per_sec)

        onset = np.zeros_like(frame)
        for inst in midi.instruments:
            for note in inst.notes:
                onset[note.pitch, int(note.start * frames_per_sec)] = 1

        # to shape (time, pitch (88))
        frame_tensor:Tensor = torch.from_numpy(frame[MIN_MIDI:MAX_MIDI + 1].T)
        onset_tensor:Tensor = torch.from_numpy(onset[MIN_MIDI:MAX_MIDI + 1].T)
        return dict(frame=frame_tensor, onset=onset_tensor)
   
    def __getitem__(self, index:int) -> Dict[str,Tensor]:
        data:Dict[str,Tensor] = self.data[index]
        frames_input:Tensor = (data['frame_input'] >= 1)
        onsets_input:Tensor = (data['onset_input'] >= 1)
        frames_output:Tensor = (data['frame_output'] >= 1)
        onsets_output:Tensor = (data['onset_output'] >= 1)

        frame_len:int = frames_input.shape[0]
        if self.sample_length is not None:
            n_steps:int = self.sample_length // self.hop_size

            step_begin:int = self.random.randint(frame_len - n_steps) if self.random_sample else 0
            step_end:int = step_begin + n_steps
            
            sample_begin:int = step_begin * self.hop_size
            sample_end:int = sample_begin + self.sample_length

            frames_input_seg:Tensor = frames_input[step_begin:step_end]
            onsets_input_seg:Tensor = onsets_input[step_begin:step_end]
            frames_output_seg:Tensor = frames_output[step_begin:step_end]
            onsets_output_seg:Tensor = onsets_output[step_begin:step_end]

            result = dict(path=data['path'])
            result['frame_input'] = frames_input_seg.float()
            result['onset_input'] = onsets_input_seg.float()
            result['frame_output'] = frames_output_seg.float()
            result['onset_output'] = onsets_output_seg.float()
        else:
            result = dict(path=data['path'])
            result['frame_input'] = frames_input.float()
            result['onset_input'] = onsets_input.float()
            result['frame_output'] = frames_output.float()
            result['onset_output'] = onsets_output.float()
        return result

    def __len__(self) -> int:
        return len(self.data)