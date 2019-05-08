# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:46:16 2019

@author: John J Hernandez
"""
import numpy as np 

def calculateCuadraticError(x,y,t):
    h=np.dot(x,t)
    suma=0
    N=len(y)
    y.reshape(N,1)
    for i in range(N):
        suma+=(h[i]-y[i])*(h[i]-y[i])
    return (suma*(1/N))

def getTethasWithGradientD(x,y,t,alpha,iterations):
     m=len(y)
     h=np.dot(x,t)
     
     for it in range(iterations):
         temporal=np.copy(t)
         for j in range(t.shape[0]):
             suma=0
             for i in range(m):
                 suma+=(h[i][0]-y[i])*x[i][j]
             temporal[j]=t[j]-alpha*(1/m)*suma
         t=np.copy(temporal)
         print("iteration number:"+str(it)+"cuadratic error:"+str(calculateCuadraticError(x,y,t)))               
     return t

#leemos los datos
data=np.loadtxt("blood_pressure.txt",dtype=int)

#calculamos el numero de datos que representan el 70%
p70=int(len(data)*0.7)

#agregamos a train los datos desde el principio hasta el 70% (AGREGAR ALEATORIO SI ES POSIBLE)
train=data[:p70,:]

#y luego obtenemos el resto de los datos
test=data[p70:,:]

#copiamos en X la primera columna de train puesto que representa los valores de x1 y la otra es y
x=np.copy(train[:,0])

#ahora x tendra en su primera columna solo 1, representando  la columna de x0 
x=np.c_[np.ones(p70),x]

#tambien se puede con concatenate peor actualmente no tienen la segunda dimension asi que le axis 1 da error
#xo=np.concatenate((np.ones(p70).reshape(p70,1),x.reshape(len(x),1)),axis=1)

# y tenemos que y representa la segunda columna de data
y=np.copy(train[:,1])

#ahora necesitamos crear las tetas, las cuales pueden empezar en 1, y habran tantas
#como columnas tenga x
t=np.ones((x.shape[1],1))

#calculamos el producto punto solo para saber como se calcular el valor de la funcion
#h=np.dot(x,t)

#probando la funcion de error creada
#calculateCuadraticError(x,y,t)

#probamos la funcion de gradiente descendiente
newTethas=getTethasWithGradientD(x,y,t,0.1,100)
print(newTethas)
#ya tenemos las nuevas tethas obtenidas a partir del algoritmo
#ahora probamos el error
xt=np.copy(test[:,0])
xt=np.c_[np.ones(len(test)),xt]
y=np.copy(test[:,1])

print(calculateCuadraticError(xt,y,newTethas))

##PORQUE ESTA AUMENTANDO EL ERROR A MEDIDA QUE SUCEDEN LAS ITERACIONESs