import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile 
import pandas as pd
#Primer punto: Leer los datos
data_signal = np.genfromtxt('signal.dat',usecols=[0,2])
data_incompletos = np.genfromtxt('incompletos.dat',usecols=[0,2])

freq_sg=data_signal[:,0]
ampl_sg=data_signal[:,1]
freq_icp=data_incompletos[:,0]
ampl_icd=data_incompletos[:,1]
print(data_incompletos) 
