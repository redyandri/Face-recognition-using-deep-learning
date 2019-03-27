import csv
import numpy as np
import pandas as pd
from shutil import copyfile
import os

class Victorinox(object):

    def __init__(self):
        return

    def gatherPhotoByNIP(self,nip_csv,img_folder,dest_flder):
        df=pd.read_csv(nip_csv)
        nip=np.array(df)
        nip=nip.tolist()
        nip2=[]
        for n in nip:
            for m in n:
                nip2.append(str(m))
        for dir,folders, files in os.walk(img_folder):
            if len(files)>0:
                c=0
                for f in files:
                    f2="".join(f.split(".")[0:-1])
                    if nip2.__contains__(f2):
                        src=os.path.join(dir,f)
                        dst=os.path.join(dest_flder,f2+".jpg")
                        copyfile(src,dst)
                        c+=1
        print "copied %d files"%c
        return