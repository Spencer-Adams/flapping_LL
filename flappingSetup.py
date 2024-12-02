import numpy as np

def interpolate(x1,x2,x,y1,y2):
    return (y2-y1)/(x2-x1)*(x-x1)+y1

def theta2s(theta):
    return -1.*np.cos(theta)

def s2theta(s):
    return np.arccos(-s)

def omega(th):
    return abs(np.cos(th))

def chi(th):
    s = theta2s(th)
    if s < -sat or abs(s) <= sar or sat < s:
        return 0.
    elif -sat <= s and s < -sar:
        return flapEfficiency(th)
    else:
        return -flapEfficiency(th)

def flapEfficiency(th):
    s = abs(theta2s(th))
    hs = interpolate(sar,sat,s,har,hat)
    c = chord(th)
    te = 0.75*c
    fcf = (te - hs) / c
    thf = np.arccos(2*fcf-1) # thf
    ei = 1 - (thf - np.sin(thf)) / np.pi
    return de*he*ei

def chord(th):
    # s = theta2s(th)
    # cw = 1.
    # cr, ct = 2.*cw/(1.+Rt), 2.*Rt*cw/(1.+Rt)
    # return cr + abs(s) * (ct - cr)
    return (4)/np.pi*np.sin(th)

isElliptic = True      ## flag that can be used to know if the planform is elliptic (useful when creating the C matrix)

Ra = 8.0
bw = Ra         ## implies cw = 1, which means chord(th) should give a cw of 1
cw = 1
Rt = 1.0
CLa = 2*np.pi

aoa_deg     = 8.0
Omega_deg   = 0.0
das_deg      = 0.0
# pbar        = 0.199997482553477502
pbar = 0.275

sar     = 0.2
sat     = 0.4
fcf_ar  = 0.2
fcf_at  = 0.2
de      = 1. # flap efficiency
he      = 0.85 # hinge efficiency

car = chord(s2theta(sar))
cat = chord(s2theta(sat))
har = (0.75-fcf_ar)*car #
hat = (0.75-fcf_at)*cat
