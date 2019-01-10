import numpy as np
import matplotlib.pyplot as plt

# lee datos
c=np.arange(1,32)
datos=np.genfromtxt('WDBC.dat',usecols=c,delimiter=',', dtype="|U5")

f=569 # filas 
c=31 # columnas
# atributos B y M en 1 y 0
for i in range(f):
    if(datos[i,0]=='B'):
        datos[i,0]=1
    else:
        datos[i,0]=0
datos=np.asarray(datos,dtype=float)
print(datos)
# normaliza los datos
dt=np.ones((f,c))
for i in range(0,c):
	dt[:,i]=(datos[:,i]-np.mean(datos[:,i]))/ (np.std(datos[i])) 


# matriz de covarianza (mc)
dt=dt.T 
mc=np.ones((c,c))
for i in range(0,c):
	for j in range(0,c):
		mc[i,j]=sum((dt[i,:]-np.mean(dt[i,:]))*(dt[j,:]-np.mean(dt[j,:])))/(f-1)


# valores y vectores propios 
vl,vt=np.linalg.eig(mc)
[print('el autovalor:', vl[i] , ',con autovector:',vt[:,i])for i in range(c)]

# parametros importantes 
print('En base a las componentes de los vectores, los parametros mas importantes son los dos primeros parametros dados por los dos primeros valores propios:',vl[0],'y',vl[1])

# proyeccion de los datos en el sistema coordenado PC1 y PC2
vt=vt.T
proyeccion=np.dot(vt, dt)
PC1=proyeccion[0,:]
PC2=proyeccion[1,:]
PC1M=[]
PC2M=[]
PC1B=[]
PC2B=[]
for i in range(len(datos[:,0])):
    if(datos[i,0]==1):
        PC1B.append(PC1[i])
        PC2B.append(PC2[i])
    else:
        PC1M.append(PC1[i])
        PC2M.append(PC2[i])        
plt.scatter(PC1M,PC2M, color='g',label="Maligno")
plt.scatter(PC1B,PC2B, color='b',label="Benigno")
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid()
plt.savefig('ApellidoFelipe_PCA.pdf')
plt.close()

print('Los datos originales diagnosticados como malignos y benignos se clasifican (Ver Grafica del punto anterior) cuando se proyectan sobre las componentes principales del metodo PCA. Entonces, El metodo PCA sirve para diagnosticar a ciertos pacientes si se conoce su primer parametro (datos sobre PC2)')
