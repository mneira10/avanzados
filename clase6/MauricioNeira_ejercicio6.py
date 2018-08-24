import numpy as np
import matplotlib.pyplot as plt

x_obs = np.array([-2.0, 1.3, 0.4, 5.0, 0.1, -4.7, 3.0, -3.5, -1.1])
y_obs = np.array([-1.931,   2.38,   1.88,  -24.22,
                  3.31, -21.9,  -5.18, -12.23,   0.822])
sigma_y_obs = ([2.63,  6.23, -1.461, 1.376, -4.72,
                1.313, -4.886, -1.091,  0.8054])


def model(x, a, b, c):
    return a*x**2 + b*x + c


def loglikelihood(x_obs, y_obs, sigma_y_obs, a, b, c):
    d = y_obs - model(x_obs, a, b, c)
    d = d/sigma_y_obs
    d = -0.5 * np.sum(d**2)
    return d


def logprior(a, b, c):
    p = -np.inf
    if a <= 5 and a > -100 and b > -20 and b < 20 and c > -10 and c < 10:
        p = 0.0
    return p


N = 200000
lista_a = [np.random.random()]
lista_b = [np.random.random()]
lista_c = [np.random.random()]
logposterior = [loglikelihood(x_obs, y_obs, sigma_y_obs, lista_a[0],
                              lista_b[0], lista_c[0]) + logprior(lista_a[0], lista_b[0], lista_c[0])]
sigma_delta_a = 1.0
sigma_delta_b = 1.0
sigma_delta_c = 0.5
for i in range(1, N):
    propuesta_a = lista_a[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_a)
    propuesta_b = lista_b[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_b)
    propuesta_c = lista_c[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_c)
    logposterior_viejo = loglikelihood(
        x_obs, y_obs, sigma_y_obs, lista_a[i-1], lista_b[i-1], lista_c[i-1]) + logprior(lista_a[i-1], lista_b[i-1], lista_c[i-1])
    logposterior_nuevo = loglikelihood(x_obs, y_obs, sigma_y_obs, propuesta_a,
                                       propuesta_b, propuesta_c) + logprior(propuesta_a, propuesta_b, propuesta_c)
    r = min(1, np.exp(logposterior_nuevo-logposterior_viejo))
    alpha = np.random.random()
    if(alpha < r):
        lista_a.append(propuesta_a)
        lista_b.append(propuesta_b)
        lista_c.append(propuesta_c)
        logposterior.append(logposterior_nuevo)
    else:
        lista_a.append(lista_a[i-1])
        lista_b.append(lista_b[i-1])
        lista_c.append(lista_c[i-1])
        logposterior.append(logposterior_viejo)
lista_a = np.array(lista_a)
lista_b = np.array(lista_b)
lista_c = np.array(lista_c)
logposterior = np.array(logposterior)


param_a = np.mean(lista_a)
param_b = np.mean(lista_b)
param_c = np.mean(lista_c)
print("a="+str(param_a))
print("b="+str(param_b))
print("c="+str(param_c))

x = np.array([-2.0, 1.3, 0.4, 5.0, 0.1, -4.7, 3.0, -3.5, -1.1])
z = np.array([-1.931,   2.38,   1.88,  -24.22,
              3.31, -21.9,  -5.18, -12.23,   0.822])
sigma_z = ([2.63,  6.23, -1.461, 1.376, -4.72,
            1.313, -4.886, -1.091,  0.8054])
xs = np.linspace(-5, 5, 100)
ys = model(xs, param_a, param_b, param_c)
plt.plot(xs, ys, label=r"$y=ax^2+bx+c$")
plt.legend()
plt.title(r"$a={:.2f}\ b={:.2f}\ c={:.2f}$".format(param_a, param_b, param_c))
plt.errorbar(x, z, yerr=sigma_z, fmt='o')
plt.savefig("results.pdf")
