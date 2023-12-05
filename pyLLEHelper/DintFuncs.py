import numpy as np
import pandas as pd
import scipy
import scipy.constants as const
import matplotlib.pyplot as plt

def parabolic(x,a,b,c):
    return a+b*x+c*x**2

def cubic(x,a,b,c,d):
    return a+b*x+c*x**2+d*x**3

def quartic(x,a,b,c,d,e):
    return a+b*x+c*x**2+d*x**3+e*x**4

def getResAngFreqs(D1,w_0,N_mu,DfitCoeff,fit):
    mus = np.arange(N_mu)
    w_mu = [w_0]
    for mu in mus:
        w_mu.append(w_mu[2*mu] + D1 + 1/2 * calcD2(w_mu[2*mu],D1,DfitCoeff,fit) * (2*mu+1))
        w_mu.insert(0,w_mu[0] - D1 + 1/2 * calcD2(w_mu[0],D1,DfitCoeff,fit) * (2*mu+1))
    return w_mu

def calcD2(w_mu,D1,DfitCoeff,fit):
    D2 = - const.c/2 * D1**2 * beta2(w_mu,DfitCoeff,fit)
    return D2

def beta2(w_mu,DfitCoeff,fit):
    beta2 = - 2*np.pi*const.c/w_mu**2 * D(w_mu,DfitCoeff,fit) 
    return beta2

def D(w_mu,DfitCoeff,fit):
    if fit == 'q':
        return quartic(w_mu, *DfitCoeff)
    elif fit == 'c':
        return cubic(w_mu,*DfitCoeff)
    elif fit == 'p':
        return parabolic(w_mu,*DfitCoeff)

def calcSimDint(omegas, D, D1, w_0, N_mu,fit = 'q',DfitCoeff=None):
    if not DfitCoeff:
        if fit == 'q':
            DfitCoeff, _ = scipy.optimize.curve_fit(quartic, omegas, D)
        elif fit == 'c':
            DfitCoeff, _ = scipy.optimize.curve_fit(cubic, omegas, D)
        elif fit == 'p':
            DfitCoeff, _ = scipy.optimize.curve_fit(parabolic, omegas, D)
        
    angFreqs = getResAngFreqs(D1,w_0,N_mu,DfitCoeff,fit)
    mus = np.arange(-N_mu,N_mu+1)
    Dint = angFreqs - (w_0 + D1*mus)
    return np.array(Dint), np.array(angFreqs)

def calcMeasDint(resFreq,fpmp,D1_manual):
    w = resFreq*2*np.pi
    peak_indexes = np.arange(len(w))
    index_index = np.arange(1,np.size(peak_indexes)+1,dtype="float")

    #calc FSR and remove outliers
    if not D1_manual:
        FSR = np.abs(np.diff(w))
        d = np.abs(FSR - np.median(FSR))
        mdev = np.median(d)
        s = d/mdev if mdev else np.zeros(len(d))
        D1 =  np.mean(FSR[s<4e-1])
    else:
        D1 = D1_manual

    #resonance index from whic to integrate dispersion
    _center_index = np.where(np.abs(resFreq-fpmp) < 0.5*D1/2/np.pi)[0]
    assert _center_index.size == 1, 'Wavelength not found!'
    center_index = _center_index[0]
    print('PUMP IS:',resFreq[center_index])
    mu = index_index-center_index-1
    w0 = w[peak_indexes[center_index]]

    #plot normalized sweep and detected peaks 
    Dint_raw = w[peak_indexes]- (w0 + D1*mu)
    Dint = ((Dint_raw/D1)-np.around(Dint_raw/D1))*D1 # calc distance to nearest grid line in case of missed resonances
    mu_corr = mu + np.around(Dint_raw/D1)

    return Dint, D1, center_index, mu, mu_corr

def updateDint(Dint, D1,resFreq, pmp_idx, mu):
    w = resFreq*2*np.pi
    peak_indexes = np.arange(len(w))

    D1mult = (Dint[-1]-Dint[0])/(w[-1]-w[0])
    D1 = D1*(1+D1mult)

    Dint_raw = w[peak_indexes]- (w[pmp_idx] + D1*mu)
    Dint = ((Dint_raw/D1)-np.around(Dint_raw/D1))*D1 # calc distance to nearest grid line in case of missed resonances
    mu_corr = mu + np.around(Dint_raw/D1)

    return Dint, D1, mu_corr

def connectDint(DintFit,Dint, mu_corr, N_mu):
    connDint = np.copy(DintFit)
    for idx,mu in enumerate(mu_corr):
        connDint[N_mu+int(mu)] = Dint[idx]
    return connDint