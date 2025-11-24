import numpy as np
import matplotlib.pyplot as plt
import mujoco

# ================================================================
#   PART 1: THE XML MODEL (Unchanged)
# ================================================================
triple_pendulum_xml = """
<mujoco model="triple_pendulum"> 
  <option gravity="0 0 -9.81" timestep="0.001" />

  <worldbody>
    <site name="pivot" pos="0 0 0" type="sphere" size="0.05" rgba="1 1 1 1"/>
    <body name="body1" pos="0 0 0">
      <inertial pos="0 0 -1" mass="1" diaginertia="0.01 0.01 0.01" />
      <joint name="joint1" type="hinge" axis="0 1 0" pos="0 0 0" />
      <geom name="rod1" type="capsule" fromto="0 0 0 0 0 -1" size="0.03" rgba="0.8 0.2 0.2 1" />
      
      <body name="body2" pos="0 0.05 -1">
        <inertial pos="0 0 -1" mass="1" diaginertia="0.01 0.01 0.01" />
        <joint name="joint2" type="hinge" axis="0 1 0" pos="0 0 0" />
        <geom name="rod2" type="capsule" fromto="0 0 0 0 0 -1" size="0.03" rgba="0.2 0.8 0.2 1" />
        
        <body name="body3" pos="0 0.05 -1">
          <inertial pos="0 0 -1" mass="1" diaginertia="0.01 0.01 0.01" />
          <joint name="joint3" type="hinge" axis="0 1 0" pos="0 0 0" />
          <geom name="rod3" type="capsule" fromto="0 0 0 0 0 -1" size="0.03" rgba="0.2 0.2 0.8 1" />
          <site name="tip" pos="0 0 -1" type="sphere" size="0.05" rgba="0.2 0.2 0.8 1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# ================================================================
#   PART 2: RUN MULTIPLE SIMULATIONS
# ================================================================

# --- 1. DEFINE YOUR INITIAL CONDITIONS ---
# Define a list of initial conditions (ICs) to test.
# Each IC is a 3-element array [qpos0, qpos1, qpos2] in DEGREES.
ic_list = [
    ("ic1",   [10.0, 0.0, 0.0]),
    ("ic2",   [30.0, 0.0, 0.0]),
    ("ic3",   [60.0, 0.0, 0.0]), 
    ("ic4",   [90.0, 0.0, 0.0]),
]

# --- 2. SET UP SIMULATION ---
T = 100.0       # total simulation time (s)
sim_dt = 0.001 # simulation timestep (from XML)

print("Loading model from XML...")
model = mujoco.MjModel.from_xml_string(triple_pendulum_xml)
data = mujoco.MjData(model)

# This list will store the final, wrapped trajectories
all_trajectories = []

# --- 3. LOOP AND RUN ALL SIMULATIONS ---
print("Running simulations for all initial conditions...")

for name, ic_degrees in ic_list:
    print(f"  Running: {name}")
    
    # Reset the simulation to its initial state
    mujoco.mj_resetData(model, data)
    
    # Set the initial conditions
    data.qpos[0] = np.deg2rad(ic_degrees[0])
    data.qpos[1] = np.deg2rad(ic_degrees[1])
    data.qpos[2] = np.deg2rad(ic_degrees[2])
    
    # Prepare to log data for *this* run
    qpos_log = []
    
    # Run the simulation loop
    while data.time < T:
        mujoco.mj_step(model, data)
        qpos_log.append(np.copy(data.qpos))
        
    # Convert this run's log to a numpy array
    qpos_history = np.array(qpos_log)
    
    # Wrap the angles for this run
    theta1_wrapped = np.mod(qpos_history[:, 0] + np.pi, 2 * np.pi) - np.pi
    theta2_wrapped = np.mod(qpos_history[:, 1] + np.pi, 2 * np.pi) - np.pi
    theta3_wrapped = np.mod(qpos_history[:, 2] + np.pi, 2 * np.pi) - np.pi

    # Store the wrapped trajectory
    all_trajectories.append((name, theta1_wrapped, theta2_wrapped, theta3_wrapped))

print("All simulations finished. Generating 3D plot...")

# ================================================================
#   PART 3: 3D PLOTTING (Modified to show all runs)
# ================================================================

# Create a 3D figure
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Loop through our results and plot each one with a different color
for i, (name, theta1, theta2, theta3) in enumerate(all_trajectories):
    ax.plot(theta1, theta2, theta3, lw=0.5, alpha=0.8, label=name)

# Set labels, title, and legend
ax.set_title('3D Configuration Space (Multiple Initial Conditions)', fontsize=16)
ax.set_xlabel('$\theta_1$ (rad)', fontsize=12)
ax.set_ylabel('$\theta_2$ (rad)', fontsize=12)
ax.set_zlabel('$\theta_3$ (rad)', fontsize=12)
ax.legend() # Add a legend to show which line is which

# --- Save and Show Plot ---
plt.tight_layout()
#plt.savefig('fixed_pendulum_3D_multirun.png')
#print("3D plot saved to 'fixed_pendto_3D_multirun.png'")
plt.show()