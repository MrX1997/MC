import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# lee datos de signal 
signal = np.genfromtxt('signal.dat',usecols=[0,2])


# grafica los datos
plt.plot(signal[:,0],signal[:,1],label='Signal')
plt.title('Senal Signal')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('ArizaFelipe_signal.pdf')
plt.close()

# implementacion de fourier
def fourier(datos):
	fourier=[]
	for n in range(0,len(datos)):
		f=0.0
		for k in range(0,len(datos)):
			f+= (datos[k])*(np.cos(-2*np.pi*k*n/len(datos))+1j*np.sin(-2*np.pi*k*n/len(datos)))
		fourier.append(f)
	return fourier

fts=fourier(signal[:,1])
fts=np.asarray(fts) 

ffts=np.fft.fftfreq(len(fts),0.00015)

# plot 
plt.plot(abs(ffts),abs(fts),label='Signal',color='g')
plt.xlabel('Frecuencia')
plt.ylabel('Amplitud')
plt.title('Fourier de signal')
plt.legend()
plt.xlim([0,1000])
plt.savefig('ArizaFelipe_TF.pdf')
plt.close()

# frecuencias principales
f=abs(fts.copy())
f=np.asarray(f,dtype=int)
ff=ffts.copy()
i1=f==530
i2=f==372
i3=f==391
f1,f2,f3=ff[i1],ff[i2],ff[i3]
print('Las frecuencias principales de la senal signal seran',f1[0],f2[0],f3[0],'Hz')

# filtro con fc=1000Hz
fs=fts.copy()
fs[abs(ffts)>1000]=0
ifs=np.fft.ifft(fs).real
plt.plot(signal[:,0],ifs,label='Signal')
plt.title('Señal Signal Filtrada')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('ArizaFelipe_filtrada.pdf')
plt.close()

# mensaje
inc = np.genfromtxt('incompletos.dat',usecols=[0,2])
print(inc[:,0])
print('La unica razon por la cual no se puede realizar fourier sobre un conjunto de datos discretos es que estos datos presentan una discontinuidad. Entre los datos 0.01127232 y 0.01261161 se ve una falta de datos con respecto a los demas puntos, por eso no se podria efectuar fourier y una interpolacion seria necesaria')

# interpolacion
incx=np.linspace(0.0,0.0285,512)
incc= interpolate.interp1d(inc[:,0],inc[:,1],kind='quadratic',fill_value="extrapolate")
inccy=incc(incx)
inccb= interpolate.interp1d(inc[:,0],inc[:,1],kind='cubic',fill_value="extrapolate")
inccby=inccb(incx)

# fourier
fincc=fourier(inccy)
finccb=fourier(inccby)

fincc=np.asarray(fincc)
finccb=np.asarray(finccb)


# subplot de las interpolaciones y la signal
plt.figure(figsize=(8, 10))
plt.subplot(311) 
plt.title('Transformada de Fourier incompletos interpolados y signal')
plt.plot(abs(ffts), abs(fts),label='Signal')
plt.ylabel('Amplitud')
plt.xlabel('Frecuencia')
plt.legend()

plt.subplot(312) 
plt.plot(abs(ffts), abs(fincc),label='Interpolacion cuadratica')
plt.ylabel('Amplitud')
plt.legend()
plt.xlabel('Frecuencia')
    
plt.subplot(313) 
plt.plot(abs(ffts), abs(finccb),label='Interpolacion cubica')
plt.ylabel('Amplitud')
plt.legend()
plt.xlabel('Frecuencia')


plt.savefig('ArizaFelipe_TF_interpola.pdf')
plt.close()

# mensaje
print('La transformada de fourier de las interpolacion se diferencia con respecto a la transformada de fourier de signal en el sentido de que para las interpolaciones aparecen picos de baja amplitud en frecuencias de 200 a 1200 Herz')

# filtros finales
fs=fts.copy()
fs[abs(ffts)>1000]=0

fincc=fincc.copy()
fincc[abs(ffts)>1000]=0

finccb=finccb.copy()
finccb[abs(ffts)>1000]=0

fincc2=fincc.copy()
fincc2[abs(ffts)>500]=0

finccb2=finccb.copy()
finccb2[abs(ffts)>500]=0

fs2=fts.copy()
fs2[abs(ffts)>500]=0	

# plots

ifs=np.fft.ifft(fs).real
ifincc=np.fft.ifft(fincc).real
ifinccb=np.fft.ifft(finccb).real
ifs2=np.fft.ifft(fs2).real
ifincc2=np.fft.ifft(fincc2).real
ifinccb2=np.fft.ifft(finccb2).real


plt.figure(figsize=(8, 10))
plt.subplot(611) 
plt.plot(signal[:,0], ifs,label='Signal con fc=1000',color='yellow')
plt.ylabel('Y')
plt.xlabel('X')
plt.legend(loc=1)

plt.subplot(612) 
plt.plot(signal[:,0], ifincc,label='Interp. Cuadratica con fc=1000',color='red')
plt.ylabel('Y')
plt.xlabel('X')
plt.legend(loc=1)

plt.subplot(613) 
plt.plot(signal[:,0], ifinccb,label='Interp. Cubica con fc=1000',color='silver')
plt.ylabel('Y')
plt.xlabel('X')
plt.legend(loc=1)

plt.subplot(614) 
plt.plot(signal[:,0], ifs2,label='Signal con fc=500',color='y')
plt.ylabel('Y')
plt.xlabel('X')
plt.legend(loc=1)

plt.subplot(615) 
plt.plot(signal[:,0], ifincc2,label='Interp. Cuadratica con fc=500',color='red')
plt.ylabel('Y')
plt.xlabel('X')
plt.legend(loc=1)


plt.subplot(616) 
plt.plot(signal[:,0], ifinccb2,label='Interp. Cubica con fc=500',color='silver')
plt.ylabel('Y')
plt.xlabel('X')
plt.legend(loc=1)

plt.savefig('ArizaFelipe_2Filtros.pdf')
plt.close()

