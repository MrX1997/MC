from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt
#Primer punto: Almacenar la imagen en un arreglo de numpy 
img = plt.imread('arbol.png')

##Segundo punto: Grafica de la transformada con sin el filtro
img_ft = fftpack.fft2(img, axes=(0, 1))
img_ft = fftpack.fftshift( img_ft )
ps=np.log10(np.abs(img_ft)**2)
fig = plt.figure(1, figsize=(9.5,9))
plt.imshow(ps,cmap='ocean')
bar=plt.colorbar()
bar.ax.set_ylabel('$\mathrm{Intensidad}$')
plt.title('Transformada de Fourier de la imagen sin filtro')
plt.xlabel(r'$X$')
plt.ylabel(r'$Y$')
plt.savefig('ApellidoNombre_FT2D.pdf')
plt.close()

###Tercer punto: Filtro
min_t=7.2
ii=min_t<ps
c=0
p=[]
for i in range(256):
    for j in range(256):
        if(ii[i,j]==True):
            p.append(i)
            p.append(j)
            c+=1
p=np.asarray(p).reshape(c,2)
x1,x2,x3,x4=p[0,0],p[1,0],p[5,0],p[6,0]
y1,y2,y3,y4=p[0,1],p[1,1],p[5,1],p[6,1]
img_ft[x1,y1]=1.0
img_ft[x2,y2]=1.0
img_ft[x3,y3]=1.0
img_ft[x4,y4]=1.0

####Cuarto punto: Grafica de la transformada con el filtro
ps=np.log10(np.abs(img_ft)**2)
fig = plt.figure(1, figsize=(9.5,9))
plt.imshow(ps,cmap='ocean')
bar=plt.colorbar()
bar.ax.set_ylabel('$\mathrm{Intensidad}$')
plt.title('Transformada de Fourier de la imagen con filtro')
plt.xlabel(r'$X$')
plt.ylabel(r'$Y$')
plt.savefig('ApellidoNombre_FT2D_filtrada.pdf')
plt.close()

#####Quinto punto: Imagen sin ruido periodico. 
img_fshift = np.fft.ifftshift(img_ft)
img_ift = fftpack.ifft2(img_fshift, axes=(0, 1)).real
fig = plt.figure(1, figsize=(9.5,9))
plt.imshow(img_ift,cmap='gray')
plt.savefig('ApellidoNombre_Imagen_filtrada.pdf')
plt.close()




