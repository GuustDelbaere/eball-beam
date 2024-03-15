from scipy.integrate import solve_ivp
from sympy import Function, Matrix,symbols,solve,pprint,simplify,eye,Poly
import numpy as np
from scipy import signal
from CAS_lib import minreal
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

stable = 1

def ODEFUN(t, y, J, m, Rb, g, l, M, JM, d, R, L, Kb, Kt, j, B, rg, V):
    r, th, q, rdot, thdot, qdot = y
    
    dX = np.array([[rdot],
                  [(1/(J/(Rb**2)+m))*((m*r*((d/L)*thdot)**2)-m*g*(d/l)*th)], 
                  [thdot], 
                  [(1/((JM+m*r**2)*(d/l)*(rg**2)+j))*(qdot*Kt*rg-B*thdot-(2*m*r*rdot*(d/l)*thdot*(m*g*r+0.5*M*g*L)*(1-(d**2/(2*l**2))*th**2)))],
                  [qdot], 
                  [V-((Kb/rg)*thdot+R*qdot)/L]])
    
    return dX.flatten()


#constants
J = 9.9*10**(-6)
m = 0.1
Rb = 0.015
g = 9.81
l = 0.3
M = 0.5
JM = 0.006
d = 0.04

if stable == 1:
    #Stable motor coeff
    Vmax = 24 
    R = 2
    L = 0.5
    Kb = 0.1
    Kt = 0.1
    j = 0.01
    B = 0.02
    rg = 0.1
    V = 10 #input voltage
else:
    #Unstable motor coeff
    Vmax = 40 
    R = 1.6
    L = 0.001
    Kb = 0.26
    Kt = 0.26
    j = 0.0002
    B = 0.001
    rg = 0.1
    V = 10 #input voltage

#equilibrium punt en start vw

y0 = np.array([0, 1, 0, 0, 0, 0]) # r, th, q, rdot, thdot, qdot

sol = solve_ivp(ODEFUN, [0, 10], y0, args=(J, m, Rb, g, l, M, JM, d, R, L, Kb, Kt, j, B, rg, V), dense_output=True)

t = np.linspace(0, 10, 60*10)

y = sol.sol(t)

r, th, q, rdot, thdot, qdot = y


#plots
fig = plt.figure()

ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(t, r, 'r')
ax1.set_title('Variabelen')
ax1.set_xlabel('tijd')
ax1.set_ylabel('r')

ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(t, th, 'b')
ax2.set_xlabel('tijd')
ax2.set_ylabel('th')

ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(t, V*np.ones(600), 'g')
ax3.set_xlabel('tijd')
ax3.set_ylabel('V')


#animation
def beamPos(th, l):
    alpha = d*th/l
    return [0, 0, l*np.cos(alpha), l*np.sin(alpha)] #[xBottom, yBottom, xTop, yTop]

def ballPos(th, l, r):
    alpha = d*th/l
    return [(l-r)*np.cos(alpha), (l-r)*np.sin(alpha)] #[x, y]

ax4 = fig.add_subplot(2, 1, 2, aspect='equal', xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
ax4.set_title('Animatie')

x0Bottom, y0Bottom, x0Top, y0Top = beamPos(y0[1], l) #initial position of beam
x0, y0 = ballPos(y0[1], l, y0[0]) #initial position of ball

beam, = ax4.plot([x0Bottom, x0Top], [y0Bottom, y0Top], linewidth=2, color='k')
ball = ax4.add_patch(plt.Circle((x0, y0), 0.05, color='k'))


def animate(i):
    xBottom, yBottom, xTop, yTop = beamPos(th[i], l)
    beam.set_data([xBottom, xTop], [yBottom, yTop])

    x, y = ballPos(th[i], l, r[i])
    ball.set_center((x, y))

ani = animation.FuncAnimation(fig, animate, frames=range(len(t)), blit=False, interval=1000/60, repeat=False, save_count=len(t))


plt.tight_layout()
plt.show()








