import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

#Primera parte: Leer los datos en un Dataframe y los normaliza
nd=31
cl=np.arange(0,nd+1)
data=pd.read_csv('WDBC.dat',names=cl) #Ordena los datos de WDBC.dat en un Dataframe para ser leidos
data.drop(data.columns[[0]],axis=1,inplace=True) #Borra la primera columna de data
data=data.values #Optiene los valores 
columns=len(data[0])
rows=len(data[:,0])

#Volvemos los datos de diagnostico con variables binarias
for i in range(rows):
    if(data[i,0]=='M'):
        data[i,0]=1
    else:
        data[i,0]=-1

matrix=np.zeros((rows,columns))
for i in range(0,columns):
	matrix[:,i]=(data[:,i]-data[:,i].mean())/ (data[i].std()) #Normalizar data
   
##Segundo punto: Implementar matriz de covarianza 
matrix=matrix.T #Transpone los nuevos datos normalizados 
M=len(matrix[0,:])
cov_matrix=np.zeros((columns,columns))
for i in range(0,columns):
	for j in range(0,columns):
		cov_matrix[i,j]=sum((matrix[i,:]-matrix[i,:].mean())*(matrix[j,:]-matrix[j,:].mean()))/(M-1)

###Tercer punto: Imprimir vectores y valores propios
values,vectors=np.linalg.eig(cov_matrix)
for i in range(columns):
    print('El autovector {}, tiene autovalor {}'.format(vectors[:,i],values[i]))
print('\n')
    
####Cuarto punto: Parametros mas importantes
total=np.sum(values)
percent=values*100/total
for i in range(columns):
    print('El autovalor {}, describe {}% de los datos'.format(values[i],percent[i]))
maxpercent=percent[0]+percent[1]+percent[2]
print('Las primeras 3 componentes describen el {}% de los datos. Por lo tanto, estos tres primeros parametros son los mas importantes'.format(maxpercent))
print('\n')

#####Quinto Punto: Datos en la nueva base
new_coordinates= np.dot(vectors.T, matrix)
ii=data[:,0]==1 #me indica del data original cuales casos son reportados como malignos y benignos
PC1=new_coordinates[0,:]
PC2=new_coordinates[1,:]
PC3=new_coordinates[2,:]
PC1_M=PC1[ii]
PC1_B=PC1[~ii]
PC2_M=PC2[ii]
PC2_B=PC2[~ii]
PC3_M=PC3[ii]
PC3_B=PC3[~ii]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(PC1_B,PC2_B,PC3_B, color='blue',alpha=0.9,marker='.',linewidths=0.2,label="Benigno")
ax.scatter3D(PC1_M,PC2_M,PC3_M, color='red',marker='+',linewidths=0.2,label="Maligno")
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.legend()
ax.grid()
ax.set_title('Datos en el nuevo sistema de coordenadas\n')
#plt.show()
plt.savefig('ApellidoNombre_PCA.pdf')
#plt.close(
print(len(PC3_B))
######Sexto Punto
print('El metodo de PCA es util para realizar una clasificacion; pues, como se ve en la grafica del punto anterior,  al tener una proyeccion de los datos iniciales en las componentes principales se observa una clasificacion clara entre los casos catalogados como Benignos y Malignos.\nPor lo tanto, las componentes principales halladas (PC1=Diagnostico , PC2=Primera Caracteristica del paciente , PC3=Segunda Caracteristica del paciente) nos dicen que existe una correlacion entre ellas que permite dignosticar si un paciente que presenta un tumor es benigno o maligno dependiendo de su primera y segunda caracteristica')

