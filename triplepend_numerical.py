import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.integrate import solve_ivp
import sympy as sp

# ================================================================
#   PART 1: SYMBOLIC DERIVATION OF PHYSICS (using SymPy)
# ================================================================
# This part is complex, but it's how we get the correct physics.
# We will "cache" the result in a file to avoid re-running
# this slow step every time.

def get_triple_pendulum_equations(use_cache=True):
    """
    Uses SymPy to derive the equations of motion for a triple pendulum.
    Returns a fast, numerical function for the ODE solver.
    """
    cache_file = 'triple_pendulum_eom.pkl'
    
    if use_cache:
        try:
            import pickle
            with open(cache_file, 'rb') as f:
                print("Loading cached EOM function...")
                return pickle.load(f)
        except FileNotFoundError:
            print("Cached file not found, re-deriving EOM...")
    
    # --- Define symbolic variables ---
    # t = time, g = gravity
    t, g = sp.symbols('t g')
    
    # L1, L2, L3 = lengths; m1, m2, m3 = masses
    L1, L2, L3 = sp.symbols('L1 L2 L3')
    m1, m2, m3 = sp.symbols('m1 m2 m3')
    
    # th1, th2, th3 = angles (functions of time)
    th1, th2, th3 = sp.symbols('theta1 theta2 theta3', cls=sp.Function)
    
    # Differentiate angles to get angular velocities and accelerations
    # th1_d = omega1, th1_dd = alpha1
    th1_d = th1(t).diff(t)
    th2_d = th2(t).diff(t)
    th3_d = th3(t).diff(t)
    th1_dd = th1(t).diff(t, 2)
    th2_dd = th2(t).diff(t, 2)
    th3_dd = th3(t).diff(t, 2)

    # --- Define coordinates ---
    # (x1, y1) ... (x3, y3) are the Cartesian positions of the masses
    x1 = L1 * sp.sin(th1(t))
    y1 = -L1 * sp.cos(th1(t))
    
    x2 = x1 + L2 * sp.sin(th2(t))
    y2 = y1 - L2 * sp.cos(th2(t))
    
    x3 = x2 + L3 * sp.sin(th3(t))
    y3 = y2 - L3 * sp.cos(th3(t))
    
    # --- Kinetic Energy (T) ---
    # T = 0.5 * m * v^2
    # v1^2 = (dx1/dt)^2 + (dy1/dt)^2
    v1_sq = x1.diff(t)**2 + y1.diff(t)**2
    v2_sq = x2.diff(t)**2 + y2.diff(t)**2
    v3_sq = x3.diff(t)**2 + y3.diff(t)**2
    
    T = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq + 0.5 * m3 * v3_sq
    
    # --- Potential Energy (V) ---
    # V = m * g * h
    V = m1 * g * y1 + m2 * g * y2 + m3 * g * y3
    
    # --- Lagrangian (L = T - V) ---
    L = T - V
    
    # --- Lagrange's Equations ---
    # This is the "magic" step that derives the equations of motion
    # d/dt(dL/d(th_d)) - dL/d(th) = 0
    from sympy.physics.mechanics import LagrangesMethod, Lagrangian
    from sympy.physics.mechanics import dynamicsymbols
    
    # We must use SymPy's 'dynamicsymbols' for this method
    th1_dyn, th2_dyn, th3_dyn = dynamicsymbols('theta1 theta2 theta3')
    th1d_dyn, th2d_dyn, th3d_dyn = dynamicsymbols('theta1 theta2 theta3', 1)
    
    # Create a map to substitute our simple functions with dynamic symbols
    # This is a bit of a SymPy-specific quirk
    sub_map = {
        th1(t): th1_dyn, th2(t): th2_dyn, th3(t): th3_dyn,
        th1_d: th1d_dyn, th2_d: th2d_dyn, th3_d: th3d_dyn,
    }
    
    # Substitute into the Lagrangian
    L_dyn = L.subs(sub_map)
    
    # Create the LagrangianMethod object
    LM = LagrangesMethod(L_dyn, [th1_dyn, th2_dyn, th3_dyn])
    
    # Get the equations of motion
    eom = LM.form_lagranges_equations()
    
    # --- Solve for accelerations (th1_dd, th2_dd, th3_dd) ---
    # The EOM are of the form M(q) * q_dd + B(q, q_d) = 0
    # We need to solve for q_dd (the accelerations)
    
    # This gets the 'Mass Matrix' M and the 'Forcing Vector' F
    M = LM.mass_matrix_full
    F = LM.forcing_full
    
    # We solve M * q_dd = -F
    # Note: SymPy's forcing_full is -B, so we solve M * q_dd = F
    q_dd = M.inv() * F
    
    # --- Create a numerical function ---
    # Now we convert these giant symbolic equations into fast NumPy functions.
    # The 'state' vector will be [th1, th2, th3, w1, w2, w3]
    # where w = omega = angular velocity
    
    # Define the symbols for the state vector
    th1_s, th2_s, th3_s = sp.symbols('th1_s th2_s th3_s')
    w1_s, w2_s, w3_s = sp.symbols('w1_s w2_s w3_s')
    
    # Create a substitution map for the lambdify step
    # We replace the 'dynamic symbols' with the simple state symbols
    eom_sub_map = {
        th1_dyn: th1_s, th2_dyn: th2_s, th3_dyn: th3_s,
        th1d_dyn: w1_s, th2d_dyn: w2_s, th3d_dyn: w3_s
    }
    
    # Substitute into the acceleration equations
    a1_expr = q_dd[0].subs(eom_sub_map)
    a2_expr = q_dd[1].subs(eom_sub_map)
    a3_expr = q_dd[2].subs(eom_sub_map)
    
    # Define the inputs to our final function
    # (g, L's, m's) and (state vector)
    params = [g, L1, L2, L3, m1, m2, m3]
    state_vars = [th1_s, th2_s, th3_s, w1_s, w2_s, w3_s]
    
    # Use sp.lambdify to create the fast functions
    # This 'lambdifies' the symbolic expressions into numpy functions
    # for acceleration (alpha)
    a1_func = sp.lambdify(params + state_vars, a1_expr, 'numpy')
    a2_func = sp.lambdify(params + state_vars, a2_expr, 'numpy')
    a3_func = sp.lambdify(params + state_vars, a3_expr, 'numpy')

    # --- Define the final ODE function for the solver ---
    def ode_system(t, y, g_val, L1_val, L2_val, L3_val, m1_val, m2_val, m3_val):
        """
        The function required by solve_ivp.
        y = [th1, th2, th3, w1, w2, w3]
        returns dy/dt = [w1, w2, w3, a1, a2, a3]
        """
        th1, th2, th3, w1, w2, w3 = y
        
        # Create the args list
        args = [g_val, L1_val, L2_val, L3_val, m1_val, m2_val, m3_val, 
                th1, th2, th3, w1, w2, w3]
        
        # Calculate accelerations
        a1 = a1_func(*args)
        a2 = a2_func(*args)
        a3 = a3_func(*args)
        
        # Return the derivatives
        return [w1, w2, w3, a1, a2, a3]

    if use_cache:
        try:
            import pickle
            with open(cache_file, 'wb') as f:
                print("Caching EOM function...")
                pickle.dump(ode_system, f)
        except Exception as e:
            print(f"Warning: Could not cache EOM function. {e}")
            
    return ode_system


# ================================================================
#   PART 2: NUMERICAL SIMULATION
# ================================================================

# --- Get the ODE function ---
# This will either load from the cache or run the (slow)
# symbolic derivation one time.
ode_func = get_triple_pendulum_equations(use_cache=True)

# --- Physical parameters ---
g = 9.81
L1, L2, L3 = 1.0, 1.0, 1.0
m1, m2, m3 = 1.0, 1.0, 1.0
phys_params = (g, L1, L2, L3, m1, m2, m3)

# --- Numerical parameters ---
T = 30.0       # total simulation time (s)
fps = 24       # frames per second for animation
dt = 1.0 / fps # time step for animation output

# --- Initial conditions ---
# y0 = [th1, th2, th3, w1, w2, w3]
th1_0 = np.deg2rad(120.0)
th2_0 = np.deg2rad(120.0)
th3_0 = np.deg2rad(120.0)
w1_0, w2_0, w3_0 = 0.0, 0.0, 0.0

y0 = [th1_0, th2_0, th3_0, w1_0, w2_0, w3_0]

# --- Run the solver ---
print("Solving ODE...")
# We ask the solver to evaluate the solution at specific time points
# (t_eval) so that our output matches our desired animation FPS.
t_eval = np.arange(0, T, dt)

sol = solve_ivp(
    fun=ode_func,
    t_span=[0, T],
    y0=y0,
    t_eval=t_eval,
    args=phys_params,
    method='RK45'  # A good, standard solver
)
print("Solver finished.")

# --- Post-processing: Convert angles to (x, y) coordinates ---
# The solution object 'sol' has the state at all times.
# sol.y[0] = all th1 values
# sol.y[1] = all th2 values
# ...
# sol.y[5] = all w3 values

th1_vals = sol.y[0]
th2_vals = sol.y[1]
th3_vals = sol.y[2]

x1 = L1 * np.sin(th1_vals)
y1 = -L1 * np.cos(th1_vals)

x2 = x1 + L2 * np.sin(th2_vals)
y2 = y1 - L2 * np.cos(th2_vals)

x3 = x2 + L3 * np.sin(th3_vals)
y3 = y2 - L3 * np.cos(th3_vals)

# 'pos' will store all (x,y) coords for the trail
pos = np.stack((x3, y3), axis=1)
Nt = len(t_eval) # Number of time steps

# ================================================================
#   PART 3: ANIMATION (This part is very similar to your code)
# ================================================================

print("Setting up animation...")

# --- Figure setup ---
fig, ax = plt.subplots(figsize=(8,8))
ax.set_aspect('equal', 'box')
# Set plot limits
total_L = L1 + L2 + L3
margin = total_L * 1.1
ax.set_xlim(-margin, margin)
ax.set_ylim(-margin, margin)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Triple Pendulum', fontsize=15)

# --- Graphic elements ---
# Rods (lines)
rod1, = ax.plot([], [], 'o-', lw=2, markersize=3, color='gray') # Pivot to mass 1
rod2, = ax.plot([], [], 'o-', lw=2, markersize=3, color='gray') # Mass 1 to mass 2
rod3, = ax.plot([], [], 'o-', lw=2, markersize=3, color='gray') # Mass 2 to mass 3

# Masses (patches)
mass_radius = 0.05 * total_L
mass1 = patches.Circle((0, 0), radius=mass_radius, fc='C0')
mass2 = patches.Circle((0, 0), radius=mass_radius, fc='C0')
mass3 = patches.Circle((0, 0), radius=mass_radius, fc='C0')
ax.add_patch(mass1)
ax.add_patch(mass2)
ax.add_patch(mass3)

# Pivot
pivot_dot, = ax.plot([0], [0], 'ko', markersize=5)

# Trail
trail_line, = ax.plot([], [], lw=1, color='orange', alpha=0.8)

# Gravity vector (static)
arrow_x, arrow_y = -margin * 0.9, margin * 0.9
ax.arrow(arrow_x, arrow_y, 0, -0.15*total_L,
         head_width=0.08, head_length=0.12,
         fc='red', ec='red', lw=2)
ax.text(arrow_x * 1.05, arrow_y - 0.07*total_L, 'g', color='red', fontsize=13)


def init():
    rod1.set_data([], [])
    rod2.set_data([], [])
    rod3.set_data([], [])
    mass1.center = (x1[0], y1[0])
    mass2.center = (x2[0], y2[0])
    mass3.center = (x3[0], y3[0])
    trail_line.set_data([], [])
    return rod1, rod2, rod3, mass1, mass2, mass3, pivot_dot, trail_line

def update(frame):
    # Get coordinates for the current frame
    frame_x1, frame_y1 = x1[frame], y1[frame]
    frame_x2, frame_y2 = x2[frame], y2[frame]
    frame_x3, frame_y3 = x3[frame], y3[frame]
    
    # Update rods
    rod1.set_data([0, frame_x1], [0, frame_y1])
    rod2.set_data([frame_x1, frame_x2], [frame_y1, frame_y2])
    rod3.set_data([frame_x2, frame_x3], [frame_y2, frame_y3])
    
    # Update masses
    mass1.center = (frame_x1, frame_y1)
    mass2.center = (frame_x2, frame_y2)
    mass3.center = (frame_x3, frame_y3)
    
    # Update trail
    trail_line.set_data(pos[:frame, 0], pos[:frame, 1])
    
    return rod1, rod2, rod3, mass1, mass2, mass3, pivot_dot, trail_line

# --- Create and save animation ---
# 'frames' will just be the index of the time step
frames_idx = range(Nt)
anim = FuncAnimation(fig, update, frames=frames_idx, init_func=init, blit=True)

# --- Save as MP4 ---
output = 'triple_pendulum.mp4'
print(f"Saving animation to {output}...")
writer = FFMpegWriter(fps=fps, bitrate=1800)
anim.save(output, writer=writer)
plt.close(fig)

print("Done.")