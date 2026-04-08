from argparse import ArgumentParser,Namespace,Action
from init_helpers import default_parser
import sys

def get_type(action: Action) -> str:
    if action.type is not None:
        type_str = action.type.__name__
        if action.nargs in ["*", "+"]:
            return f": list[{type_str}]"
        else:
            return f": {type_str}"
    else:
        return ""

def print_args(parser:ArgumentParser):
    for action in parser._actions:
        print(f"{action.dest} {get_type(action)} = args.{action.dest}")