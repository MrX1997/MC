import numpy as np
import matplotlib.pyplot as plt

### numeral a

## La solucion a la ecuacion y=BX, Donde y & x son los datos dados, es B=(X_T.X)^(-1).X_T.y

# Lee los datos del archivo
data=np.genfromtxt('data_mean_sq.txt')

# Arreglo de datos en x(vector)
x=data[:,0]

# Arreglo de datos en y(vector)
y=data[:,1]

# Crea la matriz X(matriz)
X=np.ones((len(x),2))
for i in range(len(x)):    
    X[i,1]=x[i]
    
# Metodo que realiza la transpuesta de X(matriz)
def Transpose(x):
    r,c=len(x[:,0]),len(x[0,:])
    X=np.ones((c,r))
    for i in range(c):
        X[i,:]=x[:,i]
    return X

def B(X,y):
    # Transpuesta de X(matriz)
    X_T=Transpose(X)  
    # Inversa del producto punto entre X^T(matriz) y X(matriz)
    Inv_X_T_p_X=np.linalg.inv(np.dot(X_T,X))
    # Producto punto entre X_T(matriz) y y(vector)
    X_T_p_y=np.dot(X_T,y)
    # Vector de coeficientes B
    B=np.dot(Inv_X_T_p_X,X_T_p_y)    
    return B

betas=B(X,y)
print("Coeficientes Estimados para le modelo y=mx+b:\nb = {} \nm = {} \n".format(betas[0], betas[1]))



### numeral b

# Genera las 10 muestras de cada x(vector) y y(vector) [donde y es el logaricmo natural de y]
def muestras(x,N):
    A=10.0
    tao=0.5
    ruido=np.random.normal(scale=0.16)
    y=A*np.exp(-x/tao)+ruido
    m=np.ones((30,2))
    for i in range(30):
        m[i,0]=x[i]
        m[i,1]=y[i]
        
    matriz=[]
    for i in range(N):
        matriz.append(m)
    matriz=np.asarray(matriz)
        
    x=matriz[:,:,0].reshape((300,1))
    y=matriz[:,:,1].reshape((300,1))    
    y=np.abs(y)
    return x,np.log(y)

# rangos de las muestras de x
x1=np.linspace(0,1,30,endpoint=False)
x2=np.linspace(0,2,30,endpoint=False)
x3=np.linspace(0,3,30,endpoint=False)
x4=np.linspace(0,4,30,endpoint=False)
x5=np.linspace(0,5,30,endpoint=False)
x6=np.linspace(0,6,30,endpoint=False)

# Muestras de x(vector) y y(vector) 
x1,y1=muestras(x1,10)
x2,y2=muestras(x2,10)
x3,y3=muestras(x3,10)
x4,y4=muestras(x4,10)
x5,y5=muestras(x5,10)
x6,y6=muestras(x6,10)

# Matriz X para cada rango de x 
def m(x1,x2,x3,x4,x5,x6):
    X1,X2,X3,X4,X5,X6=np.ones((len(x1),2)),np.ones((len(x2),2)),np.ones((len(x3),2)),np.ones((len(x4),2)),np.ones((len(x5),2)),np.ones((len(x6),2))
    for i in range(len(x1)):    
        X1[i,1],X2[i,1],X3[i,1],X4[i,1],X5[i,1],X6[i,1]=x1[i],x2[i],x3[i],x4[i],x5[i],x6[i]
    return X1,X2,X3,X4,X5,X6
X1,X2,X3,X4,X5,X6=m(x1,x2,x3,x4,x5,x6)

# Betas para cada rango de x 
b1,b2,b3,b4,b5,b6=B(X1,y1),B(X2,y2),B(X3,y3),B(X4,y4),B(X5,y5),B(X6,y6)

# Coeficientes para cada rango de x 
A1,tau1=np.exp(b1[0]),-1/b1[1]
A2,tau2=np.exp(b2[0]),-1/b2[1]
A3,tau3=np.exp(b3[0]),-1/b3[1]
A4,tau4=np.exp(b4[0]),-1/b4[1]
A5,tau5=np.exp(b5[0]),-1/b5[1]
A6,tau6=np.exp(b6[0]),-1/b6[1]

print("Coeficientes Estimados para le modelo y=A.e^(-x/tau) con x=[0,1):\nA = {} \ntau = {} \n".format(A1, tau1))
print("Coeficientes Estimados para le modelo y=A.e^(-x/tau) con x=[0,2):\nA = {} \ntau = {} \n".format(A2,tau2))
print("Coeficientes Estimados para le modelo y=A.e^(-x/tau) con x=[0,3):\nA = {} \ntau = {} \n".format(A3,tau3))
print("Coeficientes Estimados para le modelo y=A.e^(-x/tau) con x=[0,4):\nA = {} \ntau = {} \n".format(A4,tau4))
print("Coeficientes Estimados para le modelo y=A.e^(-x/tau) con x=[0,5):\nA = {} \ntau = {} \n".format(A5,tau5))
print("Coeficientes Estimados para le modelo y=A.e^(-x/tau) con x=[0,6):\nA = {} \ntau = {} \n".format(A6,tau6))

    
# [lo puede borrar si quiere] Plots solo para ver como se comporta linealmente los datos
plt.figure(figsize=(10,5))
plt.subplot(231)
plt.scatter(x1,y1,s=0.5,color='green')
plt.scatter(x1,np.exp(y1),label='Original',s=0.5,color='black')
plt.subplot(232)
plt.scatter(x2,y2,s=0.5,color='y')
plt.scatter(x2,np.exp(y2),label='Original',s=0.5,color='black')
plt.subplot(233)
plt.scatter(x3,y3,s=0.5,color='blue')
plt.scatter(x3,np.exp(y3),label='Original',s=0.5,color='black')
plt.subplot(234)
plt.scatter(x4,y4,s=0.5,color='brown')
plt.scatter(x4,np.exp(y4),label='Original',s=0.5,color='black')
plt.subplot(235)
plt.scatter(x5,y5,s=0.5,color='red')
plt.scatter(x5,np.exp(y5),label='Original',s=0.5,color='black')
plt.subplot(236)
plt.scatter(x6,y6,s=0.5,color='violet')
plt.scatter(x6,np.exp(y6),label='Original',s=0.5,color='black')





