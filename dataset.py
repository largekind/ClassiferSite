from torch.utils.data import Dataset
from torchvision import transforms
import torch
from pathlib import Path
from typing import List, Tuple
import numpy as np

# npyからデータセット作成
class ClassDataset(Dataset):
  def __init__(self, data : np.ndarray, label : np.ndarray, transform : transforms.Compose = None):
    #self._data = data
    self._data = torch.from_numpy(data.transpose(0,3,1,2)).float() #shape = (N,150,150,3) -> (N,3,150,150)

    print(self._data.shape)
    self.transform = transform
    self._label = label

  def __len__(self):
    return len(self._data)

  def __getitem__(self, idx: int):
    if(self.transform != None):
      self._data[idx] = self.transform(self._data[idx])
      #print(self._data[idx].shape)

    return self._data[idx], self._label[idx]
