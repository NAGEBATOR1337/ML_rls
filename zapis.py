from __future__ import absolute_import, division, print_function, unicode_literals

import random as random
import math

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand

#      f(x,y)=cos( 2 sqrt(0.5(x-x0)^2+0.5(y-y0)^2))
#      40     138    40
#      138    255    138
#      40     138    40
#      O = 915
#      cos(0.3*sqrt((0.4x)^2))

cou=1000


c1=255
c2=138
c3=40
O=915

metki=np.zeros(cou) #метки

j=cou

with open('provceli.npy', 'wb') as f:
       while j>0:
              kolceli=random.randint(0,3) #сколько целей
              a=np.zeros((28,28,1))       #поле
              metki[cou-j]=kolceli        #запись метки
              if kolceli!=0:
                     coord=np.random.randint(1, 26, (2, kolceli))
                     i=kolceli
                     while i>0:
                            if a[coord[0][i-1]][coord[1][i-1]] > 1000:   #не попадать двумя целями в одну точку
                                   metki[cou-j]=metki[cou-j]-1
                            else:  #запись целей
                                   kef=random.randint(70,100)/100
                                   """
                                   a[coord[0][i-1]][coord[1][i-1]]=c1*kef
                                   a[coord[0][i-1]+1][coord[1][i-1]]=c2*kef
                                   a[coord[0][i-1]-1][coord[1][i-1]]=c2*kef
                                   a[coord[0][i-1]][coord[1][i-1]+1]=c2*kef
                                   a[coord[0][i-1]][coord[1][i-1]-1]=c2*kef
                                   a[coord[0][i-1]+1][coord[1][i-1]+1]=c3*kef
                                   a[coord[0][i-1]+1][coord[1][i-1]-1]=c3*kef
                                   a[coord[0][i-1]-1][coord[1][i-1]+1]=c3*kef
                                   a[coord[0][i-1]-1][coord[1][i-1]-1]=c3*kef
                                   """
                                   x=0
                                   #симуляция радара

                                   while x<28:
                                          y=0
                                          while y<28:
                                                 t=math.sqrt(((x-coord[0][i-1])**2)+((y-coord[1][i-1])**2)) #расстояние от точки до цели
                                                 if t>10:
                                                        kl=0
                                                 else:
                                                        kl=math.cos(0.5*math.sqrt((0.4*t)**2))    #коэффициент для вычисления интеграла
                                                 a[x][y]=O*kl+a[x][y]*kef #интеграл на кэф расстояния и на рандомный кэф
                                                 y=y+1
                                          x=x+1
                                   
                                                        

                            i=i-1
              np.save(f, a)
              j=j-1
       
              
print('saved')


j=cou
with open('provmetki.npy', 'wb') as f:
       while j>0:
              np.save(f, metki[cou-j])
              j=j-1
print('saved')







""" Генерация точек

aa=np.zeros((1000))

j=1000

with open('provceli.npy', 'wb') as f:
       while j>0:      
              rx=np.zeros((2,5))
              celi=random.randint(0,5)
              a=np.zeros((28,28,1))
              aa[1000-j]=celi
              if celi!=0:
                     coord=np.random.randint(1, 26, (2, celi))
                     i=celi
                     while i>0:
                            a[coord[0][i-1]][coord[1][i-1]]=random.randint(70,100)+a[coord[0][i-1]][coord[1][i-1]]
                            a[coord[0][i-1]+1][coord[1][i-1]]=a[coord[0][i-1]][coord[1][i-1]]*0.5+a[coord[0][i-1]+1][coord[1][i-1]]
                            a[coord[0][i-1]-1][coord[1][i-1]]=a[coord[0][i-1]][coord[1][i-1]]*0.5+a[coord[0][i-1]-1][coord[1][i-1]]
                            a[coord[0][i-1]][coord[1][i-1]+1]=a[coord[0][i-1]][coord[1][i-1]]*0.5+a[coord[0][i-1]][coord[1][i-1]+1]
                            a[coord[0][i-1]][coord[1][i-1]-1]=a[coord[0][i-1]][coord[1][i-1]]*0.5+a[coord[0][i-1]][coord[1][i-1]-1]
                            a[coord[0][i-1]+1][coord[1][i-1]+1]=a[coord[0][i-1]][coord[1][i-1]]*0.25+a[coord[0][i-1]+1][coord[1][i-1]+1]
                            a[coord[0][i-1]+1][coord[1][i-1]-1]=a[coord[0][i-1]][coord[1][i-1]]*0.25+a[coord[0][i-1]+1][coord[1][i-1]-1]
                            a[coord[0][i-1]-1][coord[1][i-1]+1]=a[coord[0][i-1]][coord[1][i-1]]*0.25+a[coord[0][i-1]-1][coord[1][i-1]+1]
                            a[coord[0][i-1]-1][coord[1][i-1]-1]=a[coord[0][i-1]][coord[1][i-1]]*0.25+a[coord[0][i-1]-1][coord[1][i-1]-1]
                            i=i-1
              b=np.random.randint(1, 30, (28,28,1))
              if celi!=0:
                     b=b+a
              np.save(f, b)
              j = j - 1
print('saved')
j=1000
with open('provmetki.npy', 'wb') as f:
       while j>0:
              np.save(f, aa[1000-j])
              j=j-1
print('saved')



plt.figure()
plt.imshow(np.squeeze(b, 2))
plt.colorbar()
plt.grid(False)
plt.show()
       
"""       
       



