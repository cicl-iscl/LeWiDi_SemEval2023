import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

def read_data(usage="train"):
  """@usage: 'train'/'dev'"""
  data = dict()
  datasets = ['ArMIS','MD-Agreement','ConvAbuse', 'HS-Brexit'] 

  for current_dataset in datasets:
    data[current_dataset] = {}
    current_file = '/content/drive/MyDrive/cicl_data/' + current_dataset + '_dataset/' + current_dataset + '_'+ usage +'.json' 
    data[current_dataset] = json.load(open(current_file, 'r', encoding = 'UTF-8'))                                   
 
  def extract_soft_labels(row):
    return list(row.values())

  def transform_data(data, name):
    data = data[name]
    df = pd.DataFrame(data).transpose()
    df = df.astype({"hard_label": int}, errors='raise') 
    df['data_set'] = name
    df["soft_list"] = df["soft_label"].apply(extract_soft_labels)
    return df

  dfs = [transform_data(data, k) for k in data.keys()]

  data_dict = {'ArMIS': dfs[0],'MD-Agreement': dfs[1],'ConvAbuse': dfs[2], 'HS-Brexit': dfs[3]}
  # df = pd.concat(dfs)
  return data_dict

data_dict = read_data()


def myf1_score(conf_m):
  """ Better use sci kits implementation with 'micro'"""
  precision = conf_m[0][0] /  (conf_m[0][0] + conf_m[1][0])
  recall =  conf_m[0][0] / (conf_m[0][0] + conf_m[0][1])

  return 2 * (precision * recall) / (precision + recall)


class CustomLabelDataset(Dataset):
  """Base to customize individual datasets of different MTL tasks"""

  def __init__(self, df_all, tokenizer):
      self.text = list(map(self.tokenize_func, df_all["text"]))
      self.soft_labels = df_all["soft_list"] 
      self.hard_labels = df_all["hard_label"]
      self.hard_labels_1h = Fun.one_hot(torch.tensor(df_all['hard_label'].values))
      self.tokenizer = tokenizer

  def __len__(self):
      return len(self.text)
    
  def tokenize_func(self, text):
      return self.tokenizer(text, padding="max_length", truncation=True, max_length=240)

  def __getitem__(self, idx):
      input = {"attention_mask": torch.tensor(self.text[idx]["attention_mask"]),
                "token_type_ids": torch.tensor(self.text[idx]["token_type_ids"]),
                "input_ids": torch.tensor(self.text[idx]["input_ids"])}
      return input, self.hard_labels_1h[idx], torch.tensor(self.soft_labels[idx])