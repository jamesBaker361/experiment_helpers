from diffusers import DiffusionPipeline

def disable(pipe:DiffusionPipeline):
    pipe.setattr("safety_checker",None)
    pipe.setattr("run_safety_checker", lambda self,images,*args: images,None)
    return pipe