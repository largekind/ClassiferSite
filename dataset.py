from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple
import numpy as np

# npyからデータセット作成
class ClassDataset(Dataset):
  def __init__(self, data : np.ndarray, label : np.ndarray):
    self._data = data.transpose(0,3,1,2) #shape = (N,150,150,3) -> (N,3,150,150)
    self._label = label

  def __len__(self):
    return len(self._data)

  def __getitem__(self, idx: int):
    return self._data[idx], self._label[idx]
