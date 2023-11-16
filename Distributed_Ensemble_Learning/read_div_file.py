import pandas as pd
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

results_file_name=os.path.dirname(__file__)+'/results/rst_SVHN_WResNet10-1_lr0.001_n50_s70000_breakpoints_5000_div.csv'

df = pd.read_csv(results_file_name)
pd.set_option('display.width',None)
print(df)

# with open(results_file_name) as f:
#     for line in f:
#         print(line)