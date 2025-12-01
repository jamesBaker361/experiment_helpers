import os
import torch
import json
from huggingface_hub import hf_hub_download

CONFIG_NAME="config.json"

def save_and_load_functions(model_dict,
                            save_dir,
                            api,
                            repo_id:str,
                            save_interval:int=1,
                            ):
    '''
    model_dict = str: model with state_dict()
    '''
    os.makedirs(save_dir,exist_ok=True)
    config_dict={
        "train":{
        "start_epoch":1
        }
    }
    
    
    def save():
        for weights_name, model in model_dict.items():
            save_path=os.path.join(save_dir,weights_name)
            state_dict=model.state_dict()
            torch.save(state_dict,save_path)
            try:
                api.upload_file(path_or_fileobj=save_path,
                                        path_in_repo=weights_name,
                                        repo_id=repo_id)
                print(f"uploaded {repo_id} to hub")
            except Exception as e:
                print(f"failed to upload {weights_name}")
                print(e)
        config_path=os.path.join()
        with open(config_path,"w+") as config_file:
            config_dict["train"]["start_epoch"]+=1
            json.dump(config_dict,config_file, indent=4)
            pad = " " * 2048  # ~1KB of padding
            config_file.write(pad)
        try:
            api.upload_file(path_or_fileobj=config_path,path_in_repo=CONFIG_NAME,
                                    repo_id=repo_id)
        except Exception as e:
            print(f"failed to upload {CONFIG_NAME}")
            print(e)
            
    def load(hf:bool):
        if hf:
            index_path = hf_hub_download(repo_id, CONFIG_NAME)
            pretrained_weights_path=[api.hf_hub_download(repo_id,weights_name,force_download=True) for weights_name in model_dict]
        else:
            index_path = os.path.join(save_dir, CONFIG_NAME)
            pretrained_weights_path=[os.path.join(repo_id,weights_name,) for weights_name in model_dict]
            
        with open(index_path, "r") as f:
                data = json.load(f)
        if "training" in data and "start_epoch" in data["training"]:
            start_epoch = data["training"]["start_epoch"] + 1
        else:
            start_epoch = 1  # fresh training
            
        for weights_path,(weights_name, model) in zip(pretrained_weights_path,model_dict.items()):
            model.load_state_dict(torch.load(weights_path,weights_only=True))
            
        return start_epoch
            
    return save,load

        
                