# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:46:16 2019

@author: John J Hernandez
"""
import numpy as np 

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def getTethasWithGradientD(x,y,t,alpha,iterations):
     m=len(y)
     for it in range(iterations):
         temporal=np.copy(t)
         z=np.dot(x,t)
         h=sigmoid(z)
         for j in range(t.shape[0]):
             suma=0
             for i in range(m):
                 suma=(h[i][0]-y[i])*x[i][j]*(1/m)+suma
             #print("Suma: "+str(suma))
             temporal[j]=t[j]-alpha*suma      
         t=np.copy(temporal)
     return t



#leemos los datos
data=np.loadtxt("grades_data.txt",dtype=float,delimiter=",")
#calculamos el numero de datos que representan el 70% y realizamos la particion
p70=int(len(data)*0.7)
train=data[:p70,:]
test=data[p70:,:]
#preparamos los datos de entrenamiento
x=np.copy(train[:,0:2])
x=np.c_[np.ones(p70),x]
#agregamos dos columnas mas por que hay 2 features que elevaremos al cuadrado
x=np.c_[
        x[:,0],        
        x[:,1],
        x[:,1]*x[:,1],
        x[:,2],
        x[:,2]*x[:,2]
        ]


y=np.copy(train[:,2])
t=np.ones((x.shape[1],1))

#----------TEST DATA----------------------
xt=np.copy(test[:,0:2])
xt=np.c_[np.ones(len(test)),xt]
xt=np.c_[
        xt[:,0],        
        xt[:,1],
        xt[:,1]*xt[:,1],
        xt[:,2],
        xt[:,2]*xt[:,2]
        ]
yt=np.copy(test[:,2])


alpha=0.005
#probamos la funcion de gradiente descendiente
newTethas=getTethasWithGradientD(x,y,t,alpha,200000)

#ya tenemos los tethas ahora obtenemos las predicciones
zt=np.dot(xt,newTethas)
probabilidades=sigmoid(zt)
#convert probabilities into binaries
predicciones=np.copy(probabilidades)
for i in range(len(probabilidades)):  predicciones[i]=0 if (probabilidades[i]<=0.5) else 1
#predicciones.reshape((len(predicciones)))
print("alpha:"+str(alpha))
print((predicciones==yt).mean())



