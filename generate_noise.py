import numpy as np
from math import sqrt,log,sin,cos,pi,exp




def exp_decay(t, A, K):
    return A * np.exp(-K*t)
def coloredNoise(params,length):
    noise=[]
    D=params[0]
    lam=params[1]
    delta_t=1
    n,m = np.random.uniform(0.0,1.0,2)
    E=lambda x:exp(-x*delta_t)
    e_prev=sqrt(-2*D*lam*log(m))*cos(2*pi*n)
    noise.append(e_prev)
    for i in range(length-1):
        a,b = np.random.uniform(0.0,1.0,2)
        h=sqrt(-2*D*lam*(1-E(lam)**2)*log(a))*cos(2*pi*b)
        e_next=e_prev*E(lam)+h
        noise.append(e_next)
        e_prev=e_next
    return np.array(noise)

autocorr_params=[30.0,1.0/30.0]
noise_signal=coloredNoise(autocorr_params, 1000)
f_h=open("noisy_signal"+str(autocorr_params[0])+".dat",'w')
for e in noise_signal:
  f_h.write(str(-70+e))
  f_h.write("\n")
f_h.close
