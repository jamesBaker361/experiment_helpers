from torch.utils.data import Dataset, DataLoader,random_split
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from diffusers.image_processor import VaeImageProcessor
import json
import datasets
import torch

import numpy as np
import torch.nn.functional as F

def process_image_default():
    image_processor=VaeImageProcessor()
    def _func(element):
        return image_processor.preprocess(element)[0]
    
    return _func


def process_clip_text(model=None,tokenizer=None):
    if model is None:
        model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
    def _func(text):
        inputs = tokenizer([text], padding=True, return_tensors="pt")

        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        #pooled_output = outputs.pooler_output  # pooled (EOS token) states
        return last_hidden_state[0]
    
    return _func
      

class GenericDataset(Dataset):
    def __init__(self,dataset_id,
                 simple_columns=[],
                 image_columns=[],
                 image_function=None,
                 one_hot_columns=[],
                 tensor_columns=[],
                 text_columns=[],
                 text_function=None,
                 strict:bool=True
                 ):
        
        super().__init__()
        self.data=datasets.load_dataset(dataset_id,split="train")
        self.length=-1
        for column_list in [simple_columns,image_columns,one_hot_columns,tensor_columns,text_columns]:
            for key in column_list:
                if key not in self.data.features:
                    if strict:
                        raise KeyError(f"{key} not in {dataset_id}")
                else:
                    self.length=len(self.data[key])
        
        if self.length==-1:
            raise Exception("No valid keys supplied")
        self.simple_columns=simple_columns
        self.image_columns=image_columns
        self.one_hot_columns=one_hot_columns
        self.tensor_columns=tensor_columns
        self.text_columns=text_columns
        
        for image_key in image_columns:
            self.data=self.data.cast_column("image",datasets.Image())
            if image_function is not None:
                self.data=self.data.map(lambda x :{image_key: image_function(x[image_key])})
                #print("image")
        for one_hot_key in one_hot_columns:
            n_elements=len(set(self.data[one_hot_key]))
            self.data=self.data=self.data.map(lambda x: {one_hot_key:F.one_hot(torch.tensor(x[one_hot_key]),n_elements)})
        for tensor_key in tensor_columns:
            self.data=self.data.map(lambda x: torch.tensor(x[tensor_key]))
        for text_key in text_columns:
            if text_function is not None:
                self.data=self.data.map(lambda x: text_function(x[text_key])) 
            
        self.columns=simple_columns+image_columns+one_hot_columns+tensor_columns+text_columns
        
        
    def __len__(self):
        return len(self.length)
    
    def __getitem__(self, index):
        #return super().__getitem__(index)'
        row=self.data[index]
        return {
            key: row[key] for key in self.columns
        }
        
def split_data(dataset:Dataset,train_frac:float,batch_size:int):
    test_frac=(1.-train_frac)/2.
    
    train_dataset, test_dataset,val_dataset = random_split(dataset, [train_frac, test_frac,test_frac],)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader,test_loader,val_loader