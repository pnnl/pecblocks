import torch
import os
import pandas as pd

class PVInvDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data,input_dim, out_dim):
        """
        Args:
            data (torch.Tensor): Tensor with data organized in.
        """
        self.data = torch.tensor(data)
        self.len = self.data.shape[0]
        self.n_in = input_dim
        self.n_out = out_dim
 
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx, :, range(self.n_in)], self.data[idx, :, range(self.n_in,self.n_in+self.n_out)]



def read_csv_files_to_dflist(path,  pattern=".csv", time_range =None):
        pdata =[]
        files = [fn for fn in os.listdir(path) if pattern in fn]; 
        # files = np.sort(files)
        if len(files)>0:
           
            for i in range(len(files)):
                pdata0 = pd.read_csv(os.path.join(path,files[i]),sep=',',header=0,error_bad_lines=False)
                if(time_range!=None):
                    pdata0= pdata0[(pdata0['TIME']>time_range[0]) & (pdata0['TIME']<time_range[1])]
                if pdata0.shape[0] >0:
                    pdata.append(pdata0) 
            return pdata

