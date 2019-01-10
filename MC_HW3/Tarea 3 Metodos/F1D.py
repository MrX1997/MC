import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

#Primer punto: Leer los datos
data_signal = np.genfromtxt('signal.dat',usecols=[0,2])
data_incompletos = np.genfromtxt('incompletos.dat',usecols=[0,2])

##Segundo punto: Grafica de los datos de signal.dat
t_sg=data_signal[:,0]
y_sg=data_signal[:,1]
plt.plot(t_sg,y_sg,label='Datos')
plt.title('Datos de Signal.dat')
plt.xlabel(r'$t$')
plt.ylabel(r'$Y$')
plt.legend()
plt.savefig('ApellidoNombre_signal.pdf')
plt.close()

###Tercer punto: TF de los datos de signal.dat con la implementacion 
def DFT(dt):
	gk=dt
	N=len(gk)
	dft=[]
	for n in range(0,N):
		cont=0.0
		for k in range(0,N):
			cont+= (gk[k])*(np.exp(-2j*np.pi*k*n/N))
		dft.append(cont)
	return np.asarray(dft)
FT_sg=DFT(y_sg)
N=len(FT_sg)
dt=0.0001   

#Bono Implementacion de Freq de numpy P.D. la implementacion solo sirve para N pares :V para N impares me dio pereza pera igual solo maneja N par
def freq(N,dt):
    f=np.zeros(N)
    if(N%2==0):
        f[int(N/2)]=-(N/2)
        for i in range(1,int(N/2)):
            f[i]=i
        for i in range(1,int(N/2)):
            f[-i]=-i
    return f/(N*dt)
Freq_sg=freq(N,dt)

####Cuarto punto: Grafica de la transformada de Fourier
plt.plot(abs(Freq_sg),abs(FT_sg),label='Transf. de Fourier')
plt.legend()
plt.xlabel(r'$Frecuancias[Hz]$')
plt.ylabel(r'$Amplitud$')
plt.title('Transformada de Fourier de signal.dat')
plt.savefig('ApellidoNombre_TF.pdf')
plt.close()

#####Quinto punto: Las frecuencias principales de signal
ii=np.argmax(FT_sg)
Freq_sg=abs(Freq_sg)
maxfreq=Freq_sg[ii]
print('Las Frecuencias pricipales son',maxfreq,'Hz,',Freq_sg[501],'Hz y ',Freq_sg[506],'Hz')

######Secto punto: Filtro Pasabajas
def LPFilter(ft,freq):
	frt=ft.copy()
	freq_c=1000
	frt[abs(freq)>freq_c]=0
	return frt

FT_LPF=LPFilter(FT_sg,Freq_sg)
FIT_sg=np.fft.ifft(FT_LPF)
plt.plot(t_sg,FIT_sg.real,label='Datos con filtro')
plt.title('Datos de Signal.dat')
plt.xlabel(r'$t$')
plt.ylabel(r'$Y$')
plt.legend()
plt.savefig('ApellidoNombre_filtrada.pdf')
plt.close()

#######Siete punto: Fourier de incompletos.dat
t_icp=data_incompletos[:,0]
y_icp=data_incompletos[:,1]
plt.scatter(t_icp,y_icp,label='Datos de incompletos.dat ',marker='.')
plt.title('Datos de Incompletos.dat')
plt.xlabel(r'$t$')
plt.ylabel(r'$Y$')
plt.legend()
plt.xlim([0.01,0.014])
plt.savefig('Figura_de_demostracion.pdf')
plt.close()
print('Realizando aparte un plot de los datos de incompletos.dat, se observa una disconti discontinuidades de los datos entre los tiempos 0.0115 y 0.0125 (Ver Figura_de_demostracion.pdf).')

########Octavo punto: Interpolacion cuadratica y cubica
inter_qd= interpolate.interp1d(t_icp,y_icp,kind='quadratic',fill_value="extrapolate")
inter_cb= interpolate.interp1d(t_icp,y_icp,kind='cubic',fill_value="extrapolate")

nwt_icp=np.linspace(0,0.029,512)
nwy_icp_qd=inter_qd(nwt_icp)
nwy_icp_cb=inter_cb(nwt_icp)

#Fourierazo :v
FT_icp_qd=DFT(nwy_icp_qd)
FT_icp_cb=DFT(nwy_icp_cb)
Freq_icp=freq(N,dt)

#########Noveno punto: Graficas de los datos de signal y incompletos interpolados
fig,tex=plt.subplots(3,1, figsize=(8, 10) ,  sharex=True, sharey=True)
tex[0].plot(abs(Freq_icp), abs(FT_icp_qd), color='green',label='Transf. de Fourier')
tex[0].set_title('Trans. de Fourier de la interpolacion cuadratica')
tex[0].set_ylabel('Amplitud')
tex[0].set_xlabel('Frecuencia [Hz]')
tex[1].plot(abs(Freq_icp), abs(FT_icp_cb), color='y',label='Transf. de Fourier')
tex[1].set_title('Trans. de Fourier de la interpolacion cubica')
tex[1].set_ylabel('Amplitud')
tex[1].set_xlabel('Frecuencia [Hz]')
tex[2].plot(abs(Freq_icp), abs(FT_sg), color='r',label='Transf. de Fourier')
tex[2].set_title('Trans. de Fourier de signal')
tex[2].set_ylabel('Amplitud')
tex[2].set_xlabel('Frecuencia [Hz]')
fig.subplots_adjust(hspace=0.5)
fig.suptitle('Transformada de Fourier de todos lo datos')
plt.savefig('ApellidoNombre_TF_interpola.pdf')
plt.close()

##########Decimo punto: discusion
print('Las interpolaciones hechas a los datos incompletos reconocen las presencia de cuatro picos de alta amplitud presentes en la original signal.')
print('Sin embargo, en las interpolaciones se ven amplitudes en frecuencias (de 300Hz a 1200Hz) dondes en la signal original no se presentan. De igual forma, para frecuencias altas (mayores a 2000Hz) la amplitud de las interpolaciones se estabiliza. Esto ultimo no ocurre en las transformada de la señal original signal.')

###########Once punto: filtros again
def LPFilter2(ft,freq):
	frt=ft.copy()
	freq_c=500 
	frt[abs(freq)>freq_c]=0
	return frt



############Donce punto: ultimos plons
FT_icp_qd=DFT(nwy_icp_qd)
FT_icp_cb=DFT(nwy_icp_cb)
Freq_icp=freq(N,dt)
FTF_icp_qd=LPFilter(FT_icp_qd,Freq_icp)
FTF2_icp_qd=LPFilter2(FT_icp_qd,Freq_icp)
FTF_icp_cb=LPFilter(FT_icp_cb,Freq_icp)
FTF2_icp_cb=LPFilter2(FT_icp_cb,Freq_icp)
FTF_sg=LPFilter(FT_sg,Freq_icp)
FTF2_sg=LPFilter2(FT_sg,Freq_icp)

fig,tex=plt.subplots(6,1, figsize=(8, 15) ,  sharex=True)
tex[0].plot(nwt_icp, np.fft.ifft(FTF_icp_qd).real, color='green',label='Interp. Cuadratica')
tex[0].set_title('Señal con interpolacion cuadratica. Filtro fc=1000Hz')
tex[0].set_ylabel(r'$Y$')
tex[0].set_xlabel(r'$t$')
tex[0].legend()

tex[1].plot(nwt_icp, np.fft.ifft(FTF2_icp_qd).real, color='green',label='Interp. Cuadratica')
tex[1].set_title('Señal con interpolacion cuadratica. Filtro fc=500Hz')
tex[1].set_ylabel(r'$Y$')
tex[1].set_xlabel(r'$t$')
tex[1].legend()

tex[2].plot(nwt_icp, np.fft.ifft(FTF_icp_cb).real, color='y',label='Interp. Cubica')
tex[2].set_title('Señal con interpolacion cubica. Filtro fc=1000Hz')
tex[2].set_ylabel(r'$Y$')
tex[2].set_xlabel(r'$t$')
tex[2].legend()

tex[3].plot(nwt_icp, np.fft.ifft(FTF2_icp_cb).real, color='y',label='Interp. Cubica')
tex[3].set_title('Señal con interpolacion cubica. Filtro fc=500Hz')
tex[3].set_ylabel(r'$Y$')
tex[3].set_xlabel(r'$t$')
tex[3].legend()

tex[4].plot(nwt_icp, np.fft.ifft(FTF_sg).real, color='r',label='Signal')
tex[4].set_title('Señal original Signal. Filtro fc=1000Hz')
tex[4].set_ylabel(r'$Y$')
tex[4].set_xlabel(r'$t$')
tex[4].legend()

tex[5].plot(nwt_icp, np.fft.ifft(FTF2_sg).real, color='r',label='Signal')
tex[5].set_title('Señal original Signal. Filtro fc=500Hz')
tex[5].set_ylabel(r'$Y$')
tex[5].set_xlabel(r'$t$')
tex[5].legend()

fig.subplots_adjust(hspace=0.5)
fig.suptitle('Señales fitradas')
plt.savefig('ApellidoNombre_2Filtros.pdf')
plt.close()
