import torch 
import torchvision
import torchaudio
from torchaudio.transforms import Resample, MFCC
from torch.utils.data import Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
import os

class TrimMFCCs: 

	def __call__(self, batch): 
		return batch[:, 1:, :]

class Standardize:

	def __call__(self, batch): 
		for sequence in batch: 
			sequence -= sequence.mean(axis=0)
			sequence /= sequence.std(axis=0)
		return batch 

class SpokenDigitDataset(Dataset): 
	def __init__(self, path, sr, n_mfcc): 
		assert os.path.exists(path), f'The path for Spoken digit dataset does not exists' 
		self.path = path 
		self.audio_files = os.listdir(path)
		self.sr = sr
		self.transform = torchvision.transforms.Compose([
				MFCC(sample_rate = sr, n_mfcc = n_mfcc+1), 
				TrimMFCCs(),
				Standardize(),
			])

	def __len__(self): 
		return len(os.listdir(self.path))

	def __getitem__(self, index): 
		audio, sr = torchaudio.load(os.path.join(self.path, self.audio_files[index]))
		audio = Resample(sr, self.sr)(audio)
		mfccs = self.transform(audio)
		return mfccs, int(self.audio_files[index][0])

	def split_dataset(self, split_lengths): 
		valid_dataset_len = int((split_lengths[1]/100)*len(self))
		test_dataset_len = int((split_lengths[2]/100)*len(self))
		train_dataset_len = len(self) - (valid_dataset_len+test_dataset_len)
		train_dataset, valid_dataset, test_dataset = random_split(self, [train_dataset_len, valid_dataset_len, test_dataset_len])
		return train_dataset, valid_dataset, test_dataset

def collate(batch): 
	batch.sort(key = (lambda x: x[0].shape[-1]), reverse=True)
	sequences = [mfccs.squeeze(0).permute(1, 0) for mfccs, _ in batch]
	padded_sequences = pad_sequence(sequences, batch_first=True)
	lengths = torch.LongTensor([len(mfccs) for mfccs in sequences])
	labels = torch.LongTensor([label for _, label in batch])
	return padded_sequences, lengths, labels

