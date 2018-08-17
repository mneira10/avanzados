import datetime as dt
then = dt.datetime(1996,1,11,0,0,0)
now = dt.datetime.now()
delta = now-then
print((delta).days*24*60*60 + delta.seconds)

goldenRatio = ((1+5**0.5)/2)

miLista  = [x for x in range(1,11)]
print(miLista[5:])

segundaLista = [(-1)**(x) for x in range(100)]
print(segundaLista)

terceraLista = [1]
for i in range(15):
    terceraLista.insert(0,0)
    terceraLista.append(0)
print(terceraLista)


a = [21, 48, 79, 60, 77, 
    15, 43, 90,  5,  49, 
    15, 52, 20, 70, 55,  
    4, 86, 49, 87, 59]
def mediana(lista):
    lista = sorted(lista)
    if(len(a)%2==1):
        return lista[len(lista)//2]
    else:
        suma = 0
        suma += lista[len(lista)//2]
        suma += lista[len(lista)//2+1]
        return suma/2
mediana(a)

miDict = {0:'a',1:'e',2:'i',3:'o',4:'u'}

activities = {
    'Monday': {'study':4, 'sleep':8, 'party':0},
    'Tuesday': {'study':8, 'sleep':4, 'party':0},
    'Wednesday': {'study':8, 'sleep':4, 'party':0},
    'Thursday': {'study':4, 'sleep':4, 'party':4},
    'Friday': {'study':1, 'sleep':4, 'party':8},
}

horas = {}
for day in activities.keys():
    horasSum = 0
    for activity in activities[day]:
        horasSum += activities[day][activity]
    horas[day] = horasSum
print(horas)

for i in range(5,-1,-1):
    print(i)
    
suma =0
import random
for i in range(100):
    suma += random.random()
print(suma)



def darNumero():
    suma=0
    for i in range(100):
        suma += random.random()
    return suma

def std(N):
    lista = []
    for i in range(N):
        lista.append(darNumero())
    mean = sum(lista)/len(lista)
    std = 0
    for i in lista:
        std += (i-mean)**2
    std /= len(lista)
    std = std **0.5 
    return std

print("para N = 100:")
print(std(100))
print("para N = 200:")
print(std(200))
print("para N = 300:")
print(std(300))


from random import randint
numTimes = 0 
for i in range(1000):
    points = 0
    counter = 0 


    while(points<100):
        points+=randint(1,6)
        counter +=1
        
    numTimes +=counter
    
print(numTimes/1000)

def distance(a,b):
    temp = 0
    for i in range(len(a)):
        temp += (a[i]-b[i])**2
    temp = temp**0.5
    return temp


class Circle:
    def __init__(self, radius):
        self.radius = radius #all attributes must be preceded by "self."
    def area(self):
        import math
        return math.pi * self.radius * self.radius
    def perimeter(self):
        import math
        return math.pi*2*self.radius


class Vector3D:
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
    def dot(self,other):
        return self.x*other.x + self.y+other.y + self.z+other.z


import random
lista = [x for x in range(10)]
print(lista)
random.shuffle(lista)
print(lista)
