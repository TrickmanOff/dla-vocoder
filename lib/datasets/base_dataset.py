from abc import abstractmethod

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    @abstractmethod
    def __getitem__(self, idx: int):
        """
        {
            'mel': Tensor of shape (freq, T),
            'wave': Tensor of shape (1, wave_len),
        }
        """
        raise NotImplementedError
