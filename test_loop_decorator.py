import sys
import os
from pathlib import Path
import sys
from accelerate import Accelerator

parent_dir = Path.cwd().parent
sys.path.append(str(parent_dir))
sys.path.append(os.getcwd())
import unittest
import torch
from torch.utils.data import DataLoader,random_split,TensorDataset
from torchvision import datasets, transforms
import torch.nn.functional as F

from loop_decorator import optimization_loop

from utils_test import *    

class TestDecorator(unittest.TestCase):
    
    def setUp(self):
        self.accelerator=Accelerator(gradient_accumulation_steps=4)
        
    
    def test_minimal_stub(self):
        
        train_loader=[1 for _ in range(2)]
        epochs=3
        
        @optimization_loop(accelerator=self.accelerator,
                           train_loader=train_loader,
                           epochs=epochs,
                           val_interval=0)
        def stub(batch,train:bool):
            if train:
                return 1
            else:
                return 0
            
        stub()
        self.accelerator.print("\n")
        
    def test_minimal_stub_all_loaders(self):
        train_loader=[1 for _ in range(2)]
        epochs=5
        val_interval=2
        val_loader=[1 for _ in range(2)]
        test_loader=[1 for _ in range(2)]
        
        @optimization_loop(accelerator=self.accelerator,
                           train_loader=train_loader,
                           epochs=epochs,
                           val_interval=val_interval,
                           val_loader=val_loader,
                           test_loader=test_loader)
        def stub(batch,train:bool):
            if train:
                return 1
            else:
                return 0
            
        stub()
        self.accelerator.print("\n")
        
    def test_minimal_stub_all_loaders_save(self):
        train_loader=[1 for _ in range(2)]
        epochs=5
        val_interval=2
        val_loader=[1 for _ in range(2)]
        test_loader=[1 for _ in range(2)]
        
        save_dict={
            "epochs":0
        }
        def save():
            save_dict["epochs"]+=1
            self.accelerator.print("\tsaved epoch ",save_dict["epochs"])
        
        @optimization_loop(accelerator=self.accelerator,
                           train_loader=train_loader,
                           epochs=epochs,
                           val_interval=val_interval,
                           val_loader=val_loader,
                           test_loader=test_loader,
                           save_function=save)
        def stub(batch,train:bool):
            if train:
                return 1
            else:
                return 0
            
        stub()
        self.accelerator.print("\n")
        
    def test_mnist(self):
        train_loader,val_loader,test_loader=get_mnist_data(2)
        
        save_dict={
            "epochs":0
        }
        def save():
            save_dict["epochs"]+=1
            self.accelerator.print("\tsaved epoch ",save_dict["epochs"])
            
        epochs=10
        val_interval=2
        limit=2
        
        model=get_mnist_model()
        
        prior_state_dict={key:value.cpu().detach().clone().numpy() for key,value in model.state_dict().items()}
        
        save_path="model.safetensors"
        
        def save():
            save_dict["epochs"]+=1
            self.accelerator.print("\tsaved epoch ",save_dict["epochs"])
            torch.save(model.state_dict(),save_path)
        
        optimizer=torch.optim.Adam(model.parameters())
            
        @optimization_loop(accelerator=self.accelerator,
                           train_loader=train_loader,
                           epochs=epochs,
                           limit=limit,
                           val_interval=val_interval,
                           val_loader=val_loader,
                           test_loader=test_loader,
                           save_function=save,
                           #model_list=[model]
                           )
        def stub(batch,train:bool,
                 #model_list:list
                 ):
            #model=model_list[0]
            images,labels=batch
            
            predicted=model(images)
            
            loss=torch.nn.CrossEntropyLoss()(predicted,labels)
            
            if train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            return loss.cpu().detach().numpy() #,[model]
        trained_state_dict={key:value.cpu().detach().numpy() for key,value in model.state_dict().items()}
        for key,prior_value in prior_state_dict.items():
            self.assertTrue((prior_value==trained_state_dict[key]).all())
        stub()
        trained_state_dict={key:value.cpu().detach().numpy() for key,value in model.state_dict().items()}
        for key,prior_value in prior_state_dict.items():
            self.assertFalse((prior_value==trained_state_dict[key]).all())
            
        new_model=get_mnist_model()
        new_model.load_state_dict(torch.load(save_path))
        new_state_dict={key:value.cpu().detach().numpy() for key,value in new_model.state_dict().items()}
        for key,new_value in new_state_dict.items():
            self.assertTrue((new_value==trained_state_dict[key]).all())
        self.accelerator.print("\n")
        
    def test_mnist_accumulate(self):
        train_loader,val_loader,test_loader=get_mnist_data(1)
        
        save_dict={
            "epochs":0
        }
            
        epochs=5
        val_interval=2
        limit=10
        
        model=get_mnist_model()
        prior_state_dict={key:value.cpu().detach().clone().numpy() for key,value in model.state_dict().items()}
        
        save_path="model.safetensors"
        
        def save():
            save_dict["epochs"]+=1
            self.accelerator.print("\tsaved epoch ",save_dict["epochs"])
            torch.save(model.state_dict(),save_path)
        
        optimizer=torch.optim.Adam(model.parameters())
            
        @optimization_loop(accelerator=self.accelerator,
                           train_loader=train_loader,
                           epochs=epochs,
                           limit=limit,
                           val_interval=val_interval,
                           val_loader=val_loader,
                           test_loader=test_loader,
                           save_function=save)
        def stub(batch,train:bool):
            images,labels=batch
            
            
            if train:
                with self.accelerator.accumulate():
                    predicted=model(images)
            
                    loss=torch.nn.CrossEntropyLoss()(predicted,labels)
                
                
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                predicted=model(images)
            
                loss=torch.nn.CrossEntropyLoss()(predicted,labels)
                
            return loss.cpu().detach().numpy()
        trained_state_dict={key:value.cpu().detach().numpy() for key,value in model.state_dict().items()}
        for key,prior_value in prior_state_dict.items():
            self.assertTrue((prior_value==trained_state_dict[key]).all())
        stub()
        trained_state_dict={key:value.cpu().detach().numpy() for key,value in model.state_dict().items()}
        for key,prior_value in prior_state_dict.items():
            self.assertFalse((prior_value==trained_state_dict[key]).all())
        new_model=get_mnist_model()
        new_model.load_state_dict(torch.load(save_path))
        new_state_dict={key:value.cpu().detach().numpy() for key,value in new_model.state_dict().items()}
        for key,new_value in new_state_dict.items():
            self.assertTrue((new_value==trained_state_dict[key]).all())
        self.accelerator.print("\n")
        
    def test_regression(self):
        train_loader,val_loader,test_loader=get_regression_data(2)
        save_dict={
            "epochs":0
        }
            
        epochs=5
        val_interval=2
        limit=10
        
        model=get_regression_model()
        prior_state_dict={key:value.cpu().detach().clone().numpy() for key,value in model.state_dict().items()}
        
        save_path="model.safetensors"
        
        def save():
            save_dict["epochs"]+=1
            self.accelerator.print("\tsaved epoch ",save_dict["epochs"])
            torch.save(model.state_dict(),save_path)
        
        optimizer=torch.optim.Adam(model.parameters())
        
        @optimization_loop(accelerator=self.accelerator,
                           train_loader=train_loader,
                           epochs=epochs,
                           limit=limit,
                           val_interval=val_interval,
                           val_loader=val_loader,
                           test_loader=test_loader,
                           save_function=save)
        def stub(batch,train:bool):
            x,y=batch
            
            predicted=model(x)
            loss=F.mse_loss(predicted.float(),y.float())
            
            if train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            return loss.cpu().detach().numpy()
        
        stub()
        trained_state_dict={key:value.cpu().detach().numpy() for key,value in model.state_dict().items()}
        for key,prior_value in prior_state_dict.items():
            self.assertFalse((prior_value==trained_state_dict[key]).all())
        
        
        
        
    def test_autoencoder_mnist(self):
        train_loader,val_loader,test_loader=get_mnist_data(2)
        
        save_dict={
            "epochs":0
        }
            
        epochs=10
        val_interval=2
        limit=2
        
        model=get_mnist_autoencoder_model()
        
        prior_state_dict={key:value.cpu().detach().clone().numpy() for key,value in model.state_dict().items()}
        
        save_path="model_ae.safetensors"
        
        def save():
            save_dict["epochs"]+=1
            self.accelerator.print("\tsaved epoch ",save_dict["epochs"])
            torch.save(model.state_dict(),save_path)
        
        optimizer=torch.optim.Adam(model.parameters())
        
        
            
            
        @optimization_loop(accelerator=self.accelerator,
                           train_loader=train_loader,
                           epochs=epochs,
                           limit=limit,
                           val_interval=val_interval,
                           val_loader=val_loader,
                           test_loader=test_loader,
                           save_function=save,
                           #model_list=[model]
                           )
        def stub(batch,train:bool,
                 #model_list:list
                 ):
            #model=model_list[0]
            images,labels=batch
            
            images=transforms.Resize((32,32))(images)
            
            predicted=model(images)
            
            
            loss=F.mse_loss(predicted,images)
            
            if train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            return loss.cpu().detach().numpy() #,[model]
        trained_state_dict={key:value.cpu().detach().numpy() for key,value in model.state_dict().items()}
        for key,prior_value in prior_state_dict.items():
            self.assertTrue((prior_value==trained_state_dict[key]).all())
        stub()
        trained_state_dict={key:value.cpu().detach().numpy() for key,value in model.state_dict().items()}
        for key,prior_value in prior_state_dict.items():
            self.assertFalse((prior_value==trained_state_dict[key]).all())
            
        new_model=get_mnist_autoencoder_model()
        print([k for k in new_model.state_dict().keys()])
        print([k for k in torch.load(save_path).keys()])
        new_model.load_state_dict(torch.load(save_path))
        new_state_dict={key:value.cpu().detach().numpy() for key,value in new_model.state_dict().items()}
        for key,new_value in new_state_dict.items():
            self.assertTrue((new_value==trained_state_dict[key]).all())
        self.accelerator.print("\n")
            
        
if __name__=="__main__":
    unittest.main()