from huggingface_hub import HfApi
import argparse

def delete_models(api:HfApi):
    models = api.list_models(author=api.whoami()["name"])
    for repo in models:
        print("Deleting model:", repo.modelId)
        api.delete_repo(repo_id=repo.modelId, repo_type="model")
        
def delete_datasets(api:HfApi):
    datasets = api.list_datasets(author=api.whoami()["name"])
    for repo in datasets:
        print("Deleting dataset:", repo.id)
        api.delete_repo(repo_id=repo.id, repo_type="dataset")

if __name__=="__main__":
    api = HfApi()

    # list all repos you own
    models = api.list_models(author=api.whoami()["name"])
    datasets = api.list_datasets(author=api.whoami()["name"])
    spaces = api.list_spaces(author=api.whoami()["name"])
    parser=argparse.ArgumentParser()
    parser.add_argument("--delete_models",action="store_true")
    parser.add_argument("--delete_datasets",action="store_true")
    args=parser.parse_args()
    print(args)
    if args.delete_models:
        delete_models(api)
            
    if args.delete_datasets:
        delete_datasets(api)
            
        