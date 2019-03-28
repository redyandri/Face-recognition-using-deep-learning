from Victorinox import Victorinox
import csv
import numpy as np
import pandas as pd
from shutil import copyfile
import os

tool=Victorinox()

nip_csv="/home/andri/miniconda3/envs/python2.7_env/app/mofface/data"
f="psi_pusintek.txt"
print os.path.join(nip_csv,f)