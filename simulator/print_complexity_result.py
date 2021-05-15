import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
'''Script for plotting values in any given resultfile from the folder Logfiles'''

parser = argparse.ArgumentParser()
parser.add_argument('name', help='name of the relevant resultfile', type=str)
parser.add_argument('folder', help='name of the folder with relevant resultfile', type=str)
args = parser.parse_args()
if args.folder:
    print(os.path.realpath(args.folder))
path = os.getcwd()
path += '\\'
path += args.folder
os.chdir(path)

df = pd.read_csv(args.name)
print(df)
