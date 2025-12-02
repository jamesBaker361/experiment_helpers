from accelerate import Accelerator,PartialState
from accelerate.utils import set_seed
from huggingface_hub import HfApi
import time
import random
from huggingface_hub.errors import HfHubHTTPError
import argparse

def default_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mixed_precision",type=str,default="fp16")
    parser.add_argument("--project_name",type=str,default="person")
    parser.add_argument("--gradient_accumulation_steps",type=int,default=4)
    parser.add_argument("--repo_id",type=str,default="jlbaker361/model",help="name on hf")
    parser.add_argument("--lr",type=float,default=0.0001)
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--limit",type=int,default=-1)
    parser.add_argument("--save_dir",type=str,default="weights")
    parser.add_argument("--batch_size",type=int,default=4)
    parser.add_argument("--load_hf",action="store_true")
    parser.add_argument("--val_interval",type=int,default=10)
    
    return parser

def repo_api_init(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision,gradient_accumulation_steps=args.gradient_accumulation_steps)
    set_seed(123)
    print("accelerator device",accelerator.device)
    state = PartialState()
    print(f"Rank {state.process_index} initialized successfully")
    if accelerator.is_main_process or state.num_processes==1:
        accelerator.print(f"main process = {state.process_index}")
    if accelerator.is_main_process or state.num_processes==1:
        try:
            accelerator.init_trackers(project_name=args.project_name,config=vars(args))

            api=HfApi()
            api.create_repo(args.repo_id,exist_ok=True)
        except HfHubHTTPError:
            print("hf hub error!")
            time.sleep(random.randint(5,120))
            accelerator.init_trackers(project_name=args.project_name,config=vars(args))

            api=HfApi()
            api.create_repo(args.repo_id,exist_ok=True)
    return api,accelerator,accelerator.device