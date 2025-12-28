from accelerate import Accelerator,PartialState
from accelerate.utils import set_seed
from huggingface_hub import HfApi
import time
import random
from huggingface_hub.errors import HfHubHTTPError
import argparse

DEFAULT_SAVE_DIR="weights"
DEFAULT_REPO_ID="jlbaker361/model"
DEFAULT_PROJECT="project"

def default_parser(different_args:dict=None):
    parser=argparse.ArgumentParser()
    default={
        "mixed_precision":"fp16",
        "project_name":DEFAULT_PROJECT,
        "gradient_accumulation_steps":4,
        "repo_id":DEFAULT_REPO_ID,
        "lr":0.0001,
        "epochs":100,
        "limit":-1,
        "save_dir":DEFAULT_SAVE_DIR,
        "batch_size":4,
        "val_interval":4
    }
    if different_args is not None:
        for k,v in different_args.items():
            default[k]=v
    for key,value in default.items():
        parser.add_argument(f"--{key}",type=type(value),default=value)
    
    
    parser.add_argument("--load_hf",action="store_true")
    
    return parser

def parse_args(parser:argparse.ArgumentParser):
    args=parser.parse_args()
    if args.project_name==DEFAULT_PROJECT:
        print("using default project name ",DEFAULT_PROJECT)
    if args.repo_id ==DEFAULT_REPO_ID:
        print("using default hf repo id ",DEFAULT_REPO_ID)
    if args.save_dir==DEFAULT_SAVE_DIR:
        print("using default save dir",DEFAULT_SAVE_DIR)
        
    return args

def repo_api_init(args):
    '''
    
    api,accelerator,device=repo_api_init(args)
    '''
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

def main_function(main,parser,print_details):
    if __name__=='__main__':
        print_details()
        start=time.time()
        args=parse_args(parser)
        print(args)
        main(args)
        end=time.time()
        seconds=end-start
        hours=seconds/(60*60)
        print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
        print("all done!")