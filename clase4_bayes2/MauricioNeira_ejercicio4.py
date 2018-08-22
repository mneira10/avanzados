import numpy as np
import matplotlib.pyplot as plt


n = 2
m = 40
npoints = 10000
popsize = 1000
x = np.linspace(0, 200, npoints)


def muestrear():
    emaxl = []
    for emax in range(15, 200):
        for i in range(1000):
            pob = np.random.randint(0, emax+1, 1000)
            par = np.random.choice(pob, 2)
            if(max(par) == 40):
                emaxl.append(emax)
    return emaxl


obs = muestrear()


histograma = plt.hist(obs, bins=30, density=True)
x = histograma[1][:-1]
y = histograma[0]


y /= sum(y)

ex = sum(x*y)
maxp = x[y == max(y)]
sigma = (sum(x**2*y)-ex**2)**0.5


plt.hist(obs, bins=30, density=True)
plt.title("Ex = " + "{:.2f}".format(ex) + ", +prob = " +
          "{:.2f}".format(maxp[0])+", sigma " + "{:.2f}".format(sigma))
plt.plot([maxp]*2, [0, 0.03], linestyle="--", label="+prob")
plt.plot([ex]*2, [0, 0.03], linestyle="--", label="E[x]")
plt.legend()
plt.savefig("edades.pdf")