import numpy as np 
import matplotlib.pyplot as plt 
import plotly.graph_objects as go

def r30(p):
    return np.exp(-p/3)*(3-2*p+(2*p**2)/9)
def r31(p):
    return np.exp(-p/3)*(2*p*((4-2*(p/3))/3))
def r32(p):
    return np.exp(-p/3)*((4*p**2)/9)

def theta20(theta, N = 1):
    return  (np.sqrt(5)/(4*np.sqrt(np.pi)))*((3*np.cos(theta)**2)-1) * N
def theta21(theta,N= 1):
    return  (np.sqrt(15)/(2*np.sqrt(2*np.pi)))*(np.cos(theta)*np.sin(theta))*N
def theta22(theta,N=1):
    return (np.sqrt(15)/(4*np.sqrt(2*np.pi)))*(np.sin(theta)**2) * N 
def phi20(phi):
    return 1
def phi21(phi):
    return 2*np.cos(phi)
def phi2n1(phi):
    return 2*np.sin(phi)
def phi22(phi):
    return 2*np.cos(2*phi)
def phi2n2(phi):
    return 2*np.sin(2*phi)

r = np.arange(0,15,0.01)
zero = np.zeros(16)
x = np.arange(0,16,1)
R30 = r30(r)
R31 = r31(r)
R32 = r32(r)
plt.plot(r,R30,label = 'R30')
plt.plot(r,R31,label = 'R31')
plt.plot(r,R32,label = 'R32')
plt.plot(x,zero)
plt.xlabel('p')
plt.ylabel('R(p)')
plt.legend()
plt.show()

a = np.arange(-15,15,0.2)
X, Y, Z = np.meshgrid(a,a,a)
X_zoom, Y_zoom, Z_zoom = np.mgrid[-15:15:100j,-15:15:100j,-15:15:100j]

imax = 0.7
imin = -0.7

surface_count = 3
opacity = 0.6


p = np.sqrt(X**2 + Y**2 + Z**2)
phi = np.arctan2(Y,X)
theta = np.arctan2(np.sqrt(X**2 + Y**2),Z)
p_zoom = np.sqrt(X_zoom**2 + Y_zoom**2+ Z_zoom**2)
phi_zoom = np.arctan2(Y_zoom,X_zoom)
theta_zoom = np.arctan2(np.sqrt(X_zoom**2 + Y_zoom**2),Z_zoom)
psi_z2 = r32(p) * phi20(phi) * theta20(theta)
psi_xz = r32(p) * phi21(phi) * theta21(theta)
psi_yz = r32(p) * phi2n1(phi) * theta21(theta)
psi_x2_y2 = r32(p) * phi22(phi) * theta22(theta)
psi_xy = r32(p) * phi2n2(phi) * theta22(theta)

th = np.arange(0,2*np.pi,0.01)
ph = 0
r = phi20(ph) * theta20(th)
x = r*np.sin(th)*np.cos(ph)
z = r*np.cos(th)
fig = go.Figure(data = [go.Scatter(x= x, y = z, mode = 'lines')])
fig.show()
r = phi21(ph) * theta21(th)
x = r*np.sin(th)*np.cos(ph)
z = r*np.cos(th)
fig = go.Figure(data = [go.Scatter(x= x, y = z, mode = 'lines')])
fig.show()
ph = np.pi/4
r = phi2n1(ph) * theta21(th)
x = r*np.sin(th)*np.cos(ph)
z = r*np.cos(th)
fig = go.Figure(data = [go.Scatter(x= x, y = z, mode = 'lines')])
fig.show()
ph = 0
r = phi22(ph) * theta22(th)
x = r*np.sin(th)*np.cos(ph)
z = r*np.cos(th)
fig = go.Figure(data = [go.Scatter(x= x, y = z, mode = 'lines')])
fig.show()
ph = np.pi/4
r = phi2n2(ph) * theta22(th)
x = r*np.sin(th)*np.cos(ph)
z = r*np.cos(th)
fig = go.Figure(data = [go.Scatter(x= x, y = z, mode = 'lines')])
fig.show()

plot_psi_z2 = go.Figure(data = go.Isosurface(
    x = X.flatten(), y = Y.flatten(), z = Z.flatten(),
    value = psi_z2.flatten(),
    isomin = -0.68,
    isomax = imax,
    surface_count= 3,
    opacity= opacity,
    caps=dict(x_show=False, y_show=False, z_show = False)

    ))

plot_psi_z2.show()



plot_psi_xz = go.Figure(data = go.Isosurface(
    x = X.flatten(), y = Y.flatten(), z = Z.flatten(),
    value = psi_xz.flatten(),
    isomin = imin,
    isomax = imax,
    surface_count= surface_count,
    opacity= opacity,
    caps=dict(x_show=False, y_show=False, z_show = False)
    ))
plot_psi_xz.show()


plot_psi_yz = go.Figure(data = go.Isosurface(
    x = X.flatten(), y = Y.flatten(), z = Z.flatten(),
    value = psi_yz.flatten(),
    isomin = imin,
    isomax = imax,
    surface_count= surface_count,
    opacity= opacity,
    caps=dict(x_show=False, y_show=False, z_show = False)
    ))
plot_psi_yz.show()


plot_psi_x2_y2 = go.Figure(data = go.Isosurface(
    x = X.flatten(), y = Y.flatten(), z = Z.flatten(),
    value = psi_x2_y2.flatten(),
    isomin = imin,
    isomax = imax,
    surface_count= surface_count,
    opacity= opacity,
    caps=dict(x_show=False, y_show=False, z_show = False)
    ))
plot_psi_x2_y2.show()


plot_psi_xy = go.Figure(data = go.Isosurface(
    x = X.flatten(), y = Y.flatten(), z = Z.flatten(),
    value = psi_xy.flatten(),
    isomin = imin,
    isomax = imax,
    surface_count= surface_count,
    opacity= opacity,
    caps=dict(x_show=False, y_show=False, z_show = False)
    ))
plot_psi_xy.show()




