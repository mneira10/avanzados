import numpy as np 
import matplotlib.pyplot as plt 


x_obs = np.array([-2.0,1.3,0.4,5.0,0.1, -4.7, 3.0, -3.5,-1.1])
y_obs = np.array([ -1.931,   2.38,   1.88,  -24.22,   3.31, -21.9,  -5.18, -12.23,   0.822])
sigma_y_obs = ([ 2.63,  6.23, -1.461, 1.376, -4.72,  1.313, -4.886, -1.091,  0.8054])


ua=-1
ub=0
uc=2
sa=1
sb=1
sc=1

def model(x,a,b,c):
    return a*x**2 + b*x + c

def leapfrog(qa,pa,qb,pb,qc,pc, delta_t=1E-1, niter=5):
    qa_new = qa
    pa_new = pa
    qb_new = qb
    pb_new = pb
    qc_new = qc
    pc_new = pc
    for i in range(niter):
        pa_new = pa_new + 0.5 * delta_t * gradient_log_a(qa_new,qb_new,qc_new) #kick
        qa_new = qa_new + delta_t * pa_new #drift
        pa_new = pa_new + 0.5 * delta_t * gradient_log_a(qa_new,qb_new,qc_new) #kick
        
        pb_new = pb_new + 0.5 * delta_t * gradient_log_b(qa_new,qb_new,qc_new) #kick
        qb_new = qb_new + delta_t * pb_new #drift
        pb_new = pb_new + 0.5 * delta_t * gradient_log_b(qa_new,qb_new,qc_new) #kick
    
        pc_new = pc_new + 0.5 * delta_t * gradient_log_c(qa_new,qb_new,qc_new) #kick
        qc_new = qc_new + delta_t * pc_new #drift
        pc_new = pc_new + 0.5 * delta_t * gradient_log_c(qa_new,qb_new,qc_new) #kick
    return qa_new, pa_new, qb_new, pb_new, qc_new, pc_new

def gradient_log_a(qa,qb,qc):
    return np.log(1/(sa*(2*np.pi)**0.5))*(qa-ua)/sa**2 - sum((y_obs-model(x_obs,qa,qb,qc))*(x_obs**2/sigma_y_obs**2))
    
def gradient_log_a(qa,qb,qc):
    return np.log(1/(sa*(2*np.pi)**0.5))*(qa-ua)/sa**2 - sum((y_obs-model(x_obs,qa,qb,qc))*(x_obs/sigma_y_obs**2))

def gradient_log_a(qa,qb,qc):
    return np.log(1/(sa*(2*np.pi)**0.5))*(qa-ua)/sa**2 - sum((y_obs-model(x_obs,qa,qb,qc))*(1/sigma_y_obs**2))

def log_gausiana(x,u,sigma):
    return -np.log(1/(sigma*(2*np.pi)**0.5))*(x-u)**2/(2*sigma**2)

def log_likelyhood(a,b,c):
    return -0.5*sum(y_obs - model(x_obs,a,b,c)/sigma_y_obs)
    

def H(qa,qb,qc,pa,pb,pc):
    K = (pa**2+pb**2+pc**2)/2
    U = -log_likelyhood(qa,qb,qc)-log_gausiana(qa,ua,sa)-log_gausiana(qb,ub,sb)-log_gausiana(qc,uc,sc)
    return K + U

    

def MCMC(nsteps):
    qa = np.zeros(nsteps)
    pa = np.zeros(nsteps)
    qb = np.zeros(nsteps)
    pb = np.zeros(nsteps)
    qc = np.zeros(nsteps)
    pc = np.zeros(nsteps)
    
    
    pa[0] = np.random.normal(0,1)
    qa[0] = np.random.normal(ua,sa)
    pb[0] = np.random.normal(0,1)
    qb[0] = np.random.normal(ub,sb)
    pc[0] = np.random.normal(0,1)
    qc[0] = np.random.normal(uc,sc)
    
    
    for i in range(1,nsteps):
        pa[i] = np.random.normal(0,1)
        pb[i] = np.random.normal(0,1)
        pc[i] = np.random.normal(0,1)
                            
#         qa_new, pa_new = leapfroga(qa[i-1],pa[i-1]) # la propuesta se hace con leapfrog
#         qb_new, pb_new = leapfrogb(qb[i-1],pb[i-1]) # la propuesta se hace con leapfrog
#         qc_new, pc_new = leapfrogc(qc[i-1],pc[i-1]) # la propuesta se hace con leapfrog
        
        qa_new, pa_new, qb_new, pb_new, qc_new, pc_new = leapfrog(qa[i-1],pa[i-1],qb[i-1],pb[i-1],qc[i-1],pc[i-1])
                            
        pa_new = -pa_new #negamos a p para que la propuesta sea simetrica.
        pb_new = -pb_new #negamos a p para que la propuesta sea simetrica.
        pc_new = -pc_new #negamos a p para que la propuesta sea simetrica.
        
#         E_new = H(q_new, p_new, sigma)# En lugar de evaluar la pdf se evalua la energia.
        E_new = H(qa_new,qb_new,qc_new,pa_new,pb_new,pc_new)# En lugar de evaluar la pdf se evalua la energia.
        E_old = H(q[i-1], p[i-1], sigma)
        E_old = H(qa[i-1],qb[i-1],qc[i-1], pa[i-1],pb[i-1],pc[i-1])
        alpha = min(1.0,np.exp(-(E_new - E_old))) # Se comparan las dos energias
        beta = np.random.random()
        if beta < alpha:
            qa[i] = qa_new
            qb[i] = qb_new
            qc[i] = qc_new
        else:
            qa[i] = qa[i-1]
            qb[i] = qb[i-1]
            qc[i] = qc[i-1]
    return qa,qb,qc

