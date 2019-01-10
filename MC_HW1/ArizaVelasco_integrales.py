import numpy as np

import matplotlib.pyplot as plt


# Parte a.
# Se define la funcion coseno

def func(x):
    return (np.cos(x))

#Metodo Trapezoide

#a es el limite inferior
#b es el limite superior
#n es la cantidad de puntos a evaluar

# Metodo trapezoide

def trapezoide(func, a, b, n):
    analitica = np.sin(b)-np.sin(a)
    h = (b-a)/n
    x = np.linspace(a,b,n)
    integral=0.0
    
    for i in range(n-1):
        integral += (h/2)*(func(x[i+1])+func(x[i]))
    error1 = np.abs(integral - analitica )/ analitica

    return integral, error1

#Metodo simpson's rule

def simpson(f,a,b,n):
    integral = f(a) + f(b)  
    analitica = np.sin(b)-np.sin(a)
    h=(b-a)/(2.0*n)
    oddSum = 0.0
    evenSum = 0.0
    for i in range(1,n): 
        oddSum += f(2.0*h*i+a)
    integral += oddSum * 2.0
    for i in range(1,n+1): 
        evenSum += f(h*(-1.0+2.0*i)+a)
    integral += evenSum * 4.0
    totalinte=integral*h/3.0
    error = np.abs(totalinte- analitica )/ analitica
    return totalinte, error

#Metodo Monte Carlo

def monte_carlo(func,a,b,n):
    analitica = np.sin(b)-np.sin(a)
    n_random = n
    random_x = np.random.rand(n_random) * (b - a) + a
    random_v = np.random.rand(n_random)
    delta = func(random_x) - random_v
    inside=np.where(delta>0.0)
    interval_integral = (b - a)
    integral  = (interval_integral * (len(inside))/(1.0*len(random_v)))   
    error = np.abs(integral- analitica )/ analitica

    return integral, error

#Metodo valor medio

def valor_medio(func, a, b, n):
    analitica = np.sin(b)-np.sin(a)
    x = np.random.random(n) * (b - a) + a 
    integral = np.average(func(x)) * (b - a)
    error = np.abs(integral - analitica )/ analitica
    return integral, error

n =1000001
tr,errortr=trapezoide(func, -(np.pi)/2.0,np.pi,n)
s,errors=simpson(func, -(np.pi)/2.0,np.pi,n)
m,errorm=monte_carlo(func,-(np.pi)/2.0,np.pi,n)
vm,errorvm=valor_medio(func,-(np.pi)/2.0,np.pi,n)
print('metodo: Trapezoide', ', valor de la integral', tr, ', error',errortr )
print('metodo: Simpson', ', valor de la integral', s, ', error',errors )
print('metodo: Monte Carlo', ', valor de la integral', m, ', error',errorm )
print('metodo: Valor Medio', ', valor de la integral', vm, ', error',errorvm )


# Parte b

l=np.logspace(2,7,6)+1
N=np.zeros(len(l))
ErrorTR=np.zeros(len(l))
ErrorSS=np.zeros(len(l))
ErrorMC=np.zeros(len(l))
ErrorVM=np.zeros(len(l))
k=0
for i in l:
    tr,errortr=trapezoide(func, -(np.pi)/2.0,np.pi,int(i))
    s,errors=simpson(func, -(np.pi)/2.0,np.pi,int(i))
    m,errorm=monte_carlo(func,-(np.pi)/2.0,np.pi,int(i))
    vm,errorvm=valor_medio(func,-(np.pi)/2.0,np.pi,int(i))
    N[k]=i
    ErrorTR[k]=errortr
    ErrorSS[k]=errors
    ErrorMC[k]=errorm
    ErrorVM[k]=errorvm
    k+=1

plt.plot(N,ErrorTR,'ro-', label='Trapezoide',color='blue')
plt.plot(N,ErrorSS,'ro-', label='Simpson',color='red')
plt.plot(N,ErrorMC,'ro-', label='Monte Carlo',color='y')
plt.plot(N,ErrorVM,'ro-', label='Valor Medio',color='black')
plt.xscale('log')
plt.xlabel('N')
plt.ylabel('Error de cada metodo')
plt.title('Error de cada metodo vs. N')
plt.savefig('ArizaVelasco_int_error.pdf')

## Punto C
# 1
nc=10001
c=0.0
d=1.0
def g(x):
    return 1.0/np.sqrt(np.sin(x))

def gtrapezoide(func, a, b, n):
    analitica = 2.03481 # desarrollada en wolfram alpha
    h = (b-a)/n
    x = np.linspace(a,b,n)
    integral=0.0
    
    for i in range(n-1):
        integral += (h/2)*(func(x[i+1])+func(x[i]))
    error1 = np.abs(integral - analitica )/ analitica

    return integral, error1

#Metodo simpson's rule

def gsimpson(f,a,b,n):
    integral = f(a) + f(b)  
    analitica = 2.03481 # desarrollada en wolfram alpha para 
    h=(b-a)/(2.0*n)
    oddSum = 0.0
    evenSum = 0.0
    for i in range(1,n): 
        oddSum += f(2.0*h*i+a)
    integral += oddSum * 2.0
    for i in range(1,n+1): 
        evenSum += f(h*(-1.0+2.0*i)+a)
    integral += evenSum * 4.0
    totalinte=integral*h/3.0
    error = np.abs(totalinte- analitica )/ analitica
    return totalinte, error

#Metodo Monte Carlo

def gmonte_carlo(func,a,b,n):
    analitica = 2.03481 # desarrollada en wolfram alpha
    n_random = n
    random_x = np.random.rand(n_random) * (b - a) + a
    random_v = np.random.rand(n_random)
    delta = func(random_x) - random_v
    inside=np.where(delta>0.0)
    interval_integral = (b - a)
    integral  = (interval_integral * (len(inside))/(1.0*len(random_v)))   
    error = np.abs(integral- analitica )/ analitica

    return integral, error

#Metodo valor medio

def gvalor_medio(func, a, b, n):
    analitica = 2.03481 # desarrollada en wolfram alpha
    x = np.random.random(n) * (b - a) + a 
    integral = np.average(func(x)) * (b - a)
    error = np.abs(integral - analitica )/ analitica
    return integral, error

ng=10001
e=10**-6.0
gtr,gerrortr=gtrapezoide(g, c,d,ng)
gs,gerrors=gsimpson(g, c,d,ng)
gm,gerrorm=gmonte_carlo(g, c,d,ng)
gvm,gerrorvm=gvalor_medio(g, c,d,ng)
print('metodo: Trapezoide', ', valor de la integral', gtr, ', error',gerrortr )
print('metodo: Simpson', ', valor de la integral', gs, ', error',gerrors )
print('metodo: Monte Carlo', ', valor de la integral', gm, ', error',gerrorm )
print('metodo: Valor Medio', ', valor de la integral', gvm, ', error',gerrorvm )

# punto 2

def infisimpson(f,a,b,n):
    integral = 10**6 + f(b)  
    analitica = 2.03481 # desarrollada en wolfram alpha para 
    h=(b-a)/(2.0*n)
    oddSum = 0.0
    evenSum = 0.0
    for i in range(1,n): 
        oddSum += f(2.0*h*i+a)
    integral += oddSum * 2.0
    for i in range(1,n+1): 
        evenSum += f(h*(-1.0+2.0*i)+a)
    integral += evenSum * 4.0
    totalinte=integral*h/3.0
    error = np.abs(totalinte- analitica )/ analitica
    return totalinte, error
gsinfi,errorinfi=infisimpson(g,c,d,ng)

print('El nuevo valor de la integral usando el metodo de Simpson cambiando infinito por 10^6 el valor de la integral', gsinfi )

# punto 3

gscero,gerrorscero=gsimpson(g, e,d,ng)
print('El nuevo valor de la integral usando el metodo de Simpson evaluando la funcion en 10^-6 y no en cero el valor de la integral', gscero )

# punto 4

def h(x): # primer termino 
    return 1.0/np.sqrt(np.sin(x)) - 1.0/np.sqrt(x)

integral_analitica=2.0 # integral from o to 1 of 1/x^(1/2)

hss,herrorss=gsimpson(h, e,d,ng)
hsst=hss+integral_analitica
print('Restando la singularidad el resultado es:',hsst)

# punto 5 
def htrapezoide(func, a, b, n):
    analitica = 2.03481 # desarrollada en wolfram alpha
    h = (b-a)/n
    x = np.linspace(a,b,n)
    integral=0.0
    
    for i in range(n-1):
        integral += (h/2)*(func(x[i+1])+func(x[i]))
    error1 = np.abs(integral - analitica )/ analitica

    return integral, error1

#Metodo simpson's rule

def hsimpson(f,a,b,n):
    integral = f(a) + f(b)  
    analitica = 2.03480532 # referencia
    h=(b-a)/(2.0*n)
    oddSum = 0.0
    evenSum = 0.0
    for i in range(1,n): 
        oddSum += f(2.0*h*i+a)
    integral += oddSum * 2.0
    for i in range(1,n+1): 
        evenSum += f(h*(-1.0+2.0*i)+a)
    integral += evenSum * 4.0
    totalinte=integral*h/3.0
    error = np.abs(totalinte- analitica )/ analitica
    return totalinte, error

#Metodo Monte Carlo

def hmonte_carlo(func,a,b,n):
    analitica = 2.03480532 # referencia
    n_random = n
    random_x = np.random.rand(n_random) * (b - a) + a
    random_v = np.random.rand(n_random)
    delta = func(random_x) - random_v
    inside=np.where(delta>0.0)
    interval_integral = (b - a)
    integral  = (interval_integral * (len(inside))/(1.0*len(random_v)))   
    error = np.abs(integral- analitica )/ analitica

    return integral, error

#Metodo valor medio

def hvalor_medio(func, a, b, n):
    analitica = 2.03480532 # referencia
    x = np.random.random(n) * (b - a) + a 
    integral = np.average(func(x)) * (b - a)
    error = np.abs(integral - analitica )/ analitica
    return integral, error

htr,herrortr=htrapezoide(g, e,d,ng)
hs,herrors=hsimpson(g, e,d,ng)
hm,herrorm=hmonte_carlo(g, c,d,ng)
hvm,herrorvm=hvalor_medio(g, c,d,ng)

mejor=np.asarray([['Trapezoide',herrortr],['Simpson',herrors],['Monte Carlo',herrorm],['Valor Medio',herrorvm]])
m=mejor[:,1]
print(m)

print('El mejor metodo para resolver la integral es: Simpson')

## Los metodos de Trapezoide y Simpson no sirven para la integral numerica de g(x). Sin embargo, el metodo de valor medio se aproxima fuertemente al resustaldo 
