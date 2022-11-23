import math

import numpy as np


def garch_n(ve, params):
    '''
    Model:
        y(t) = e(t) = sig(t) * iidN(0,1)
        e(t)^2 = domega + dbeta * sig(t-1)^2 + dalpha * e(t-1)^2
    Data:
        ve
    Parametars:
        domega
        dbeta
        dalpha
    '''
    domega = params[0]
    dbeta = params[1]
    dalpha = params[2]
    ns = len(ve)
    ve2 = ve**2
    vsig2 = np.zeros(ns).reshape(ns, 1)

    if ((dalpha + dbeta) >= 1) or (domega <= 0) or (dbeta < 0) or (dalpha < 0):
        return None, 10**10
    else:
        dini = domega / (1-dbeta-dalpha)
        vsig2[0] = dini
        for t in np.arange(1, ns, 1):
            vsig2[t] = domega + dbeta * vsig2[t-1] + dalpha * ve2[t-1]
        dloglik = -0.5 * np.sum(np.log(vsig2)+ve2/vsig2)
    return vsig2, -dloglik


def garch_skt(ve, params):
    '''
    Model:
        y(t) = e(t) = sig(t) * iidSkew-t(0,1)
        e(t)^2 = domega + dbeta * sig(t-1)^2 + dalpha * e(t-1)^2
    Data:
        ve
    Parametars:
        domega
        dbeta
        dalpha
        eta
        lamd
    '''
    domega = params[0]
    dbeta = params[1]
    dalpha = params[2]
    eta = params[3]
    lamd = params[4]
    ns = len(ve)
    ve2 = ve**2
    vsig2 = np.zeros(ns).reshape(ns, 1)
    logL = np.zeros(ns)

    if ((dalpha+dbeta) >= 1) or (domega <= 0) or (dbeta < 0) or (dalpha < 0):
        return None, 10**10
    elif (eta <= 2) or (lamd <= -1) or (lamd >= 1):
        return None, 10**10
    else:
        dini = domega / (1-dbeta-dalpha)
        vsig2[0] = dini
        logL[0] = np.log(np.sqrt(vsig2[0])) + np.log(hsktpdf(ve[0], eta, lamd))
        for t in np.arange(1, ns, 1):
            vsig2[t] = domega + dbeta * vsig2[t-1] + dalpha * ve2[t-1]
            logL[t] = np.log(np.sqrt(vsig2[t])) + np.log(hsktpdf(ve[t], eta, lamd))
    return vsig2, -np.sum(logL)


def gjr_n(ve, params):
    '''
    Model:
        y(t) = e(t) = sig(t) * iidN(0,1)
        e(t)^2 = domega + dbeta * sig(t-1)^2 + dalpha * e(t-1)^2 + dgamma*I(e(t-1)<0)*e(t-1)^2
    Data:
        ve
    Parametars:
        domega
        dbeta
        dalpha
        dgamma
    '''
    domega = params[0]
    dbeta = params[1]
    dalpha = params[2]
    dgamma = params[3]
    ns = len(ve)
    ve2 = ve**2
    vsig2 = np.zeros(ns).reshape(ns, 1)

    if ((dalpha+dbeta) >= 1) or (domega <= 0) or (dbeta < 0) or (dalpha < 0) or (dgamma < 0):
        return None, 10**10
    else:
        dini = domega / (1-dbeta-dalpha)
        vsig2[0] = dini
        for t in np.arange(1, ns, 1):
            if ve[t-1] >= 0:
                vsig2[t] = domega + dbeta * vsig2[t-1] + dalpha * ve2[t-1]
            else:
                vsig2[t] = domega + dbeta * vsig2[t-1] + (dalpha+dgamma) * ve2[t-1]
        dloglik = -0.5 * np.sum(np.log(vsig2)+ve2/vsig2)
    return vsig2, -dloglik


def hsktpdf(z, eta, lamd):
    '''
    mean 0 and unit variance skew-t Distribution in Hansen(1994).
    z in R,
    2<eta<infty,
    -1<lamd<1.
    '''
    if eta <= 2:
        raise TypeError('eta must be greater than 2.')
    if lamd <= -1 or lamd >= 1:
        raise TypeError('lamd must be in (-1,1).')
    c = math.gamma((eta+1)/2)/(np.sqrt(np.pi*(eta-2))*math.gamma(eta/2))
    a = 4*lamd*c*((eta-2)/(eta-1))
    b = np.sqrt(1+3*(lamd**2)-a**2)
    if z < (-a/b):
        g = b*c*(1+(1/(eta-2))*(((b*z+a)/(1-lamd))**2))**(-(eta+1)/2)
    else:
        g = b*c*(1+(1/(eta-2))*(((b*z+a)/(1+lamd))**2))**(-(eta+1)/2)
    return g
