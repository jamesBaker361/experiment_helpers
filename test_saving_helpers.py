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
import string
import random

class SavingTest(unittest.TestCase):
    def setUp(self):
        self.accelerator=Accelerator(gradient_accumulation_steps=4)
        
    def test_load(self):
        for flag in [True, False]:
            with self.subTest(flag=flag):
                folder=''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
                repo_id=''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
                # Source - https://stackoverflow.com/a/2257449
                # Posted by Ignacio Vazquez-Abrams, modified by community. See post 'Timeline' for change history
                # Retrieved 2025-12-01, License - CC BY-SA 4.0

                

                mnist_model=get_mnist_model()
                api=HfApi()
                save,load=save_and_load_functions({"mnist_pytorch_model.safetensors":mnist_model},
                                                folder,
                                                api,
                                                f"jlbaker361/{repo_id}")
                self.assertEqual(load(flag),load(flag))
    
    def test_save_and_load(self):
        for flag in [True, False]:
            with self.subTest(flag=flag):
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
                start_epoch=load(flag)
                save()
                next_epoch= load(flag)
                self.assertEqual(next_epoch,start_epoch+1)
                new_state_dict={k:v.cpu().detach().clone().numpy() for k,v in mnist_model.state_dict().items()}
                for key,new_value in new_state_dict.items():
                    self.assertTrue((new_value==past_state_dict[key]).all())
        
if __name__=="__main__":
    unittest.main()