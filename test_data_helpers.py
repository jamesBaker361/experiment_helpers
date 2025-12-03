import sys
import os
from pathlib import Path
import sys
from accelerate import Accelerator

parent_dir = Path.cwd().parent
sys.path.append(str(parent_dir))
sys.path.append(os.getcwd())
import unittest
from data_helpers import GenericDataset,process_clip_text,process_image_default

from loop_decorator import optimization_loop
from datasets import load_dataset

from utils_test import *    

DEFAULT_DATA="jlbaker361/test-dataset-experiment-helpers"

class TestData(unittest.TestCase):
    def setUp(self):
        return super().setUp()
    
    def test_generic_dataset_key_error(self):
        with self.assertRaises(KeyError):
            GenericDataset("jlbaker361/test-dataset-experiment-helpers",simple_columns=["jlsfdjskldf"])   
        
    def test_generic_dataset_empty(self):
        with self.assertRaises(Exception):
            GenericDataset("jlbaker361/test-dataset-experiment-helpers")
            
    def test_generic_dataset_image(self):
        data=GenericDataset(DEFAULT_DATA,image_columns=["image"],image_function=process_image_default())
        for batch in data:
            break
        
        self.assertIn("image",batch)
        self.assertIsInstance(batch["image"],torch.Tensor)
        self.assertEqual(batch["image"].size()[0],3)
        self.assertEqual(len(batch["image"].size()),3)
        
    def test_generic_dataset_text(self):
        data=GenericDataset(DEFAULT_DATA,text_columns=["text"],text_function=process_clip_text())
        for batch in data:
            break
        
        self.assertIn("text",batch)
        self.assertIsInstance(batch["text"],torch.Tensor)
        
        
    
if __name__=="__main__":
    unittest.main()