#------------------------------------------------------------------------------
# Solve and animate the double compound pendulum
# Run:
# python pendpend.py <theta1> <theta2> (in degrees)
#------------------------------------------------------------------------------
import sys
import numpy as np
import matplotlib.pyplot as plt
import celluloid
import scipy
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')

### Physical parameters used throughout
earth_g = 9.8 # m/s^2 (acceleration due to gravity)
pend_el = 0.25 # m (pendulum arm length)


### IVP solver(s) ###

### The equations ###
def single_pendulum_ODE(t,y):
    """A single simple pendulum."""
    omega = y[0]
    theta = y[1]
    return [-earth_g/pend_el * np.sin(theta), omega]


def double_compound_pendulum_ODE(t,y):
    """Double compound pendulum with identical rods."""
    
    # Unpack y0 input
    theta1, theta1dot, theta2, theta2dot = y

    # theta1 = angle of the first pendulum
    # theta1dot = angular velocity of the first pendulum
    # theta2 = angle of the second pendulum
    # theta2dot = angular velocity of the first pendulum

    # Define some shorthand
    deltateta = theta2-theta1
    a = pend_el*np.sin(deltateta)*np.cos(deltateta)
    b = earth_g*np.cos(deltateta)
    c = pend_el*np.sin(deltateta)
    d = pend_el*(2-np.cos(deltateta)**2)
    g = earth_g

    w1 = theta1dot
    w2 = theta2dot
    w1dot = (a*w1**2+(b*np.sin(theta2)+c*w2**2)-2*g*np.sin(theta1))/d
    w2dot = (-a*w2**2+2*(b*np.sin(theta1)-c*w1**2)-2*g*np.sin(theta2))/d
    
    return [w1, w1dot, w2, w2dot]

### The solvers ###
def solve_simple_pendulum(teta0):
    """Call the ivp driver on the simple pendulum ode."""


    t = np.arange(0,1000)
    teta = np.zeros(t.size)
    tetadot = np.zeros(t.size)
    sol = scipy.integrate.solve_ivp(single_pendulum_ODE, [0,1000], [0, teta0], method='RK45', t_eval=t)
    t = sol.t
    teta = sol.y[1]
    tetadot = sol.y[0]
    return (t, teta, tetadot)

def solve_double_pendulum(teta10,teta20):
    """Call the ivp driver on the double pendulum ode."""

    times = np.linspace(0,10,num=1000)      # Integration steps (make end greater for longer animation, make num greater for higher resolution)
    
    resolution = times.size/max(times)       # Time between steps

    teta = np.zeros((times.size,2))
    tetadot = np.zeros((times.size,2))
    y0 = [teta10, 0, teta20, 0]
    results = scipy.integrate.solve_ivp(double_compound_pendulum_ODE, [0,max(times)], y0, method='RK45', t_eval=times, max_step=resolution)

    # results.y[0,:] = theta1 angles
    # results.y[2,:] = theta2 angles
    # results.y[1,:] = omega1 values
    # results.y[3,:] = omega2 values

    teta[:,0] = results.y[0,:]          # Store theta 1 angles for plotting/animation
    teta[:,1] = results.y[2,:]          # Store theta 2 angles for plotting/animation
    tetadot[:,0] = results.y[1,:]       # Store omega 1 values to verify energy conservation
    tetadot[:,1] = results.y[3,:]       # Store omega 2 values to verify energy conservation

    return (times, teta, tetadot)


### The plotters/animators ###
def plot_simple_pendulum(t, teta, tetadot, name=''):
    # Plot the solutions
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    plt.plot(t, teta, label=name)
    plt.xlabel("$t$ [sec]", fontsize=14)
    plt.ylabel(r"$\theta$", fontsize=14)
    if name:
        plt.legend(fontsize=12)

    # And energy conservation
    plt.subplot(1,2,2)
    K = 0.5*(pend_el*tetadot)**2
    U = earth_g*(pend_el - pend_el*np.cos(teta))
    E = K + U
    plt.plot(t, np.abs((E - E[0])/(E[0]+1e-10)), label=name)
    plt.xlabel("$t$ [sec]", fontsize=14)
    plt.ylabel("Mechanical energy $|E - E(t=0)|/E(t=0)$", fontsize=14)
    if name:
        plt.legend()
    plt.show()
    return

def plot_double_pendulum(t, teta, tetadot, name=''):
    tet1 = teta[:,0] % 2*np.pi
    tet2 = teta[:,1] % 2*np.pi
    phi = tet1 - tet2
    t1dot = tetadot[:,0]
    t2dot = tetadot[:,1]
    T = (1/6)*pend_el**2*(
        4*t1dot**2 + t2dot**2 + 3*np.cos(phi)*t1dot*t2dot)
    V = -(1/2)*earth_g*pend_el*(
        3*np.cos(tet1) + np.cos(tet2))
    E = T + V

    # Plot the solutions
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    plt.plot(t, tet1, label=r'$\theta_1$')
    plt.plot(t, tet2, label=r'$\theta_2$')
    plt.xlabel("$t$ [sec]", fontsize=14)
    plt.ylabel(r"$\theta$", fontsize=14)
    plt.legend(fontsize=12)

    # And energy conservation
    plt.subplot(1,2,2)
    plt.plot(t, np.abs((E - E[0])/E[0]))
    plt.xlabel("$t$ [sec]", fontsize=14)
    plt.ylabel("Mechanical energy $|E - E(t=0)|/E(t=0)$", fontsize=14)

    plt.show()
    return

def animate_simple_pendulum(t, teta, fps_mul=1.0):
    dt = t[1] - t[0]
    el = pend_el
    x = el*np.sin(teta.squeeze())
    y = -el*np.cos(teta.squeeze())

    fig = plt.figure(figsize=(8,6))
    plt.xlim((-1.1*el, 1.1*el))
    plt.axis('equal')
    cam = celluloid.Camera(fig)
    for k in range(t.size):
        plt.plot([0,x[k]],[0,y[k]],'k-',linewidth=4)
        plt.plot(x[k],y[k],'o', markersize=14,color='r')
        cam.snap()
    anim = cam.animate(interval=dt*fps_mul,repeat=False)
    plt.show()
    return anim

def animate_double_pendulum(t, teta, fps_mul=1.0):
    """Animate the solution (found by solve_double_pendulum())."""
    tet1 = teta[:,0]
    tet2 = teta[:,1]
    dt = t[1] - t[0]
    el = pend_el
    x1 = el*np.sin(tet1)
    y1 = -el*np.cos(tet1)
    x2 = x1 + el*np.sin(tet2)
    y2 = y1 - el*np.cos(tet2)

    fig = plt.figure(figsize=(8,6))
    plt.xlim((-2.1*el, 2.1*el))
    plt.axis('equal')
    cam = celluloid.Camera(fig)
    for k in range(t.size):
        plt.plot([0,x1[k]],[0,y1[k]],'b-',linewidth=4)
        plt.plot([x1[k],x2[k]],[y1[k],y2[k]],'r-',linewidth=4)
        cam.snap()
    anim = cam.animate(interval=dt*fps_mul,repeat=False)
    plt.show()
    return anim

if __name__ == '__main__':
    teta1 = 2*np.pi*float(sys.argv[1])/360
    teta2 = 2*np.pi*float(sys.argv[2])/360
    t, teta, tetadot = solve_double_pendulum(teta1, teta2)
    plot_double_pendulum(t,teta,tetadot)
    animate_double_pendulum(t,teta)
    
""" To plot simple pendulum, use below lines instead"""
    #teta0 = 2*np.pi*float(sys.argv[1])/360
    #t, teta, tetadot = solve_simple_pendulum(teta0)
    #plot_simple_pendulum(t,teta,tetadot)
    #animate_simple_pendulum(t,teta)