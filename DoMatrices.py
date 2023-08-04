#Import libries
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import math


def LoadData(pathD):
    """Preprocessing the dataset
    Attributes
    ----------   
    path: the path of the file

    Returns
    -------     
    Data and label (attack/normal) separated
    """
    start = datetime.now()
    print("[" + str(start) + "]" + " Starting LoadData.")
    
    #Read Data   
    D = pd.read_csv(pathD,engine='python',sep=",",header=0)
    end = datetime.now()
    print("[" + str(end) + "]" + " Finished LoadData. Duration: " + str(end-start))
    return D


def mycorrelation(pwmat,numser,win):
    M = pwmat
    init = 0 
    ms = []
    for it in range(0,numser):
        xi = np.squeeze(np.asarray(M.iloc[it:it+1]))
        init = it + 1
        vecc = []
        for jt in range (0,numser):
            xj = np.squeeze(np.asarray(M.iloc[jt:jt+1]))
            dot = (np.dot(xi,xj))/win
            vecc.append(np.round(dot,2))
        ms.append(vecc)
    return(ms)


def domatrices (numt,win,DatT,numser):
    ttime = numt+1
    time = 1
    mxt = []
    while time < ttime-win:
        intime = time
        endtime = win+intime
        print('******TIME******',time)
        wmat = DatT.iloc[:, intime-1:endtime]
        mtxst = mycorrelation(wmat,numser,win)
        mxt.append(mtxst)
        time = time + 1
    return(mxt)

def flattensor (listtensor):
    """Flat lists of tensors
    Attributes
    ----------     
    listtensor: list of tensors
    
    Returns
    -------       
    Dataframe of tensors in 1D  
    """
    start = datetime.now()
    print("[" + str(start) + "]" + " Starting flattensor.")
    t=len(listtensor)
    # 0 element 
    auxtensor = listtensor[0]
    #flatten
    auxflat = [val for sublist in auxtensor for val in sublist] 
    #transofrm to dataframe
    dfxf = pd.DataFrame(auxflat)
    for x in range (1,t):
        tensor = listtensor[x]
        flat = [val for sublist in tensor for val in sublist]  
        dfflat= pd.DataFrame(flat)
        dfxf = pd.concat([dfxf, dfflat],axis=1)
    end = datetime.now()
    print("[" + str(end) + "]" + " Finished flattensor. Duration: " + str(end-start))
    return(dfxf)


def exportdata(pdfdata,addchar):
    start = datetime.now()
    print("[" + str(start) + "]" + " Starting export data.")    
    pdfdataT = pdfdata.transpose()
    filename = 'FileMatrices'+'_'+addchar+'.'+'csv'
    pdfdataT.to_csv(filename, index=False)
    end = datetime.now()
    print("[" + str(end) + "]" + " Finished export data. Duration: " + str(end-start))