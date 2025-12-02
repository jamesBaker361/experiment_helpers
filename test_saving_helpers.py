from saving_helpers import *

import unittest
import torch
from torch.utils.data import DataLoader,random_split,TensorDataset
from torchvision import datasets, transforms
import torch.nn.functional as F
from accelerate import Accelerator
from utils_test import *
from saving_helpers import save_and_load_functions
from huggingface_hub import HfApi

class SavingTest(unittest.TestCase):
    def setUp(self):
        self.accelerator=Accelerator(gradient_accumulation_steps=4)
        
    def test_save_and_load(self):
        mnist_model=get_mnist_model()
        mnist_data,_,_=get_mnist_data(2)
        
        api=HfApi()
        
        save,load=save_and_load_functions({"mnist_pytorch_model.safetensors":mnist_model},
                                          "mnist",
                                          api,
                                          "jlbaker361/mnist-testing")
        
        optimizer=torch.optim.Adam(mnist_model.parameters())
        
        for b, batch in  enumerate( mnist_data):
            if b>10:
                break
            images,labels=batch
            
            predicted=mnist_model(images)
            
            loss=torch.nn.CrossEntropyLoss()(predicted,labels)
            
            #if train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        save()
        past_state_dict={k:v.cpu().detach().clone().numpy() for k,v in mnist_model.state_dict().items()}
        for parameter in mnist_model.parameters():
            parameter.data.zero_()
        zero_state_dict={k:v.cpu().detach().clone().numpy() for k,v in mnist_model.state_dict().items()}
        for key,past_value in past_state_dict.items():
            self.assertFalse((past_value==zero_state_dict[key]).all())
        load(True)
        new_state_dict={k:v.cpu().detach().clone().numpy() for k,v in mnist_model.state_dict().items()}
        for key,new_value in new_state_dict.items():
            self.assertTrue((new_value==past_state_dict[key]).all())
            
    def test_save_and_load_local(self):
        mnist_model=get_mnist_model()
        mnist_data,_,_=get_mnist_data(2)
        
        api=HfApi()
        
        save,load=save_and_load_functions({"mnist_pytorch_model.safetensors":mnist_model},
                                          "mnist",
                                          api,
                                          "jlbaker361/mnist-testing")
        
        optimizer=torch.optim.Adam(mnist_model.parameters())
        
        for b, batch in  enumerate( mnist_data):
            if b>10:
                break
            images,labels=batch
            
            predicted=mnist_model(images)
            
            loss=torch.nn.CrossEntropyLoss()(predicted,labels)
            
            #if train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        save()
        past_state_dict={k:v.cpu().detach().clone().numpy() for k,v in mnist_model.state_dict().items()}
        for parameter in mnist_model.parameters():
            parameter.data.zero_()
        zero_state_dict={k:v.cpu().detach().clone().numpy() for k,v in mnist_model.state_dict().items()}
        for key,past_value in past_state_dict.items():
            self.assertFalse((past_value==zero_state_dict[key]).all())
        load(False)
        new_state_dict={k:v.cpu().detach().clone().numpy() for k,v in mnist_model.state_dict().items()}
        for key,new_value in new_state_dict.items():
            self.assertTrue((new_value==past_state_dict[key]).all())
        
if __name__=="__main__":
    unittest.main()