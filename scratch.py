from Victorinox import Victorinox
import csv
import numpy as np
import pandas as pd
from shutil import copyfile
import os

tool=Victorinox()

tool.gatherPhotoByNIP(nip_csv="/home/andri/miniconda3/envs/python2.7_env/app/mofface/data/pusintek_employee",
                      img_folder="/home/andri/Documents/data/HRISPhoto",
                      dest_flder="/home/andri/miniconda3/envs/python2.7_env/app/mofface/data")