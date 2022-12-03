import numpy as np 
import plotly.graph_objects as go

def R2(E,R,f):
    F = f(R)
    SSR  =  0
    SST = 0
    mean = np.mean(E)
    for x in range(len(E)):
        SSR += (E[x]-F[x])**2
        SST += (E[x]-mean)**2
    R = 1- (SSR/SST)
    return R
mN = 1.1622*10**-26 #reduced mass of N2 in kg
A_to_m = 10**-10
Ha_to_J = 4.35975*10**-18 
c = 299792458
E= np.array([-108.970423343, -108.971565380, -108.971947136,-108.971617061,-108.970662031,-108.969002005]) #Ha
R= np.array([1.05, 1.06,1.07,1.08,1.09,1.1]) #A
R_ex = np.array([0.7,0.725,0.75,0.775,0.8,0.825,0.85,0.875,0.9,0.925,0.95, 0.975,1,1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09,1.1,1.11,1.12,1.13,1.14,1.15,1.175,1.2,1.225,1.25,1.275,1.3,1.325,1.35,1.375,1.4,1.425,1.45]) #A
E_ex = np.array([-107.729100157,-107.966739491,-108.166229590,-108.333034117,-108.471826177,-108.586603065,-108.680785547,-108.757303156,-108.818667121,-108.867032574, -108.904251476708, -108.931917569, -108.951404483, -108.957180357,-108.961907448,-108.965650276,-108.968469741,-108.970423343, -108.971565380,-108.971947136,-108.971617061,-108.970620931,-108.969002005,-108.966801166,-108.964057060,-108.960806214,-108.957083164,-108.952920555,-108.940789385,-108.926537221,-108.910544417,-108.893140970,-108.874612733,-108.855206793,-108.835136130,-108.814583710,-108.793706079,-108.772636566,-108.751488129,-108.730355921]) #Ha
a= np.arange(1.04,1.11,0.001) #ranges for plots 
b= np.arange(0.65, 1.5, 0.001)
fit = np.polyfit(R,E, 2)
fit_ex = np.polyfit(R_ex,E_ex,2)
f_ex = np.poly1d(fit_ex)
f = np.poly1d(fit)
y= f(a)
y_ex = f_ex(b)
fig = go.Figure()
fig.add_trace(go.Scatter(x=R, y=E ,mode = 'markers'))
fig.add_trace(go.Scatter(x=a, y =y, mode = 'lines'))
fig_ex = go.Figure()
fig_ex.add_trace(go.Scatter(x=R_ex, y = E_ex,mode= 'markers'))
fig_ex.add_trace(go.Scatter(x=b,y = y_ex, mode = 'lines'))
fig.show()
fig_ex.show()
print(fit)
print(R2(E,R,f))
second_order_derivative = np.polyder(fit,2)
print(fit_ex)
print(R2(E_ex,R_ex,f_ex))
print(second_order_derivative) # positive thus minima 
k = second_order_derivative*(Ha_to_J/(A_to_m**2)) #converting to N/m 
v = (1/(2*np.pi)) * ((k/mN)**0.5)
wavelength = c/v
wavenumber = 2*np.pi/wavelength
print(wavelength)
print(wavenumber)
print(k)
print(v)
