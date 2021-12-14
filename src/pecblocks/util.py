import pandas as pd
import os
import zipfile

def read_csv_files(path, pattern=".csv"):
    if zipfile.is_zipfile (path):
        zf = zipfile.ZipFile (path)
        pdata =[]
        for zn in zf.namelist():
            pdata0 = pd.read_csv (zf.open(zn),sep=',',header=0,on_bad_lines='skip')
            if pdata0.shape[0] >0:
                pdata += [pdata0.copy()]  
        return pd.concat(pdata)
    else:
        files = [fn for fn in os.listdir(path) if pattern in fn]; 
        # files = np.sort(files)
        if len(files)>0:
            pdata =[]
            for i in range(len(files)):
                pdata0 = pd.read_csv(os.path.join(path,files[i]),sep=',',header=0,on_bad_lines='skip')
                if pdata0.shape[0] >0:
                    pdata += [pdata0.copy()]  
            return pd.concat(pdata)


