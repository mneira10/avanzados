import numpy as np 
import matplotlib.pyplot as plt
from scipy import integrate

def invExp(x,lamb):
    return np.exp(-lamb/x)/x**2

lamb = 1
d=0.5
x0 =0.5
xs = [x0]
N=1000000

for i in range(N):
    xi = xs[-1] + d*2*(np.random.random()-0.5)
    while(xi<=0):
        xi = xs[-1] + d*2*(np.random.random()-0.5)
    F = invExp(xi,lamb)/invExp(xs[i],lamb)
    if F>=1:
        xs.append(xi)
    else:
        if np.random.uniform()<F:
            xs.append(xi)
        else:
            xs.append(xs[-1])

xt = np.linspace(0.00001,300,1000)
yt = invExp(xt,lamb)
yt/=(integrate.simps(yt,xt))

histo = plt.hist(xs,bins=200,density=True,label = r"$sampled\ distribution$")

plt.plot(xt,yt,label = r"$\lambda = 1$")
plt.title(r"$f(x;\lambda)=c \frac{e^{-\lambda/x}}{x^2}$")
plt.legend()
plt.xlim(0,30)
plt.savefig("invExp.pdf")

