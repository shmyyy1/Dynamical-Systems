import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
import mujoco

# ================================================================
#   PART 1: THE XML MODEL
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
#   PART 2: MUJOCO SIMULATION
# ================================================================

# --- Numerical parameters ---
T = 100.0       # total simulation time (s)
fps = 24       # frames per second for animation
sim_dt = 0.001 # simulation timestep (from XML)
anim_dt = 1.0 / fps # time between animation frames

# --- Load Model ---
print("Loading model from XML...")
model = mujoco.MjModel.from_xml_string(triple_pendulum_xml)
data = mujoco.MjData(model)

# --- Initial conditions ---
data.qpos[0] = np.deg2rad(120)
data.qpos[1] = 0.0
data.qpos[2] = 0.0

# --- Data Logging (Corrected) ---
# We will store the (x, z) coordinates of the rod ENDS
j2_log, j3_log, tip_log = [], [], [] # Joint 2, Joint 3, and Tip
t_log = []

# --- Run the Simulation Loop (Corrected) ---
print("Running MuJoCo simulation...")
frame_counter = 0
while data.time < T:
    mujoco.mj_step(model, data)

    # Save data only at the animation framerate
    if data.time >= frame_counter * anim_dt:
        # Get the 3D world position of the JOINTS/ENDS
        # j2 is the end of rod 1 (origin of body2)
        j2_pos = data.body("body2").xpos
        # j3 is the end of rod 2 (origin of body3)
        j3_pos = data.body("body3").xpos
        # tip is the end of rod 3 (site defined in body3)
        tip_pos = data.site("tip").xpos
        
        # Store the (x, z) coordinates for our 2D plot
        j2_log.append([j2_pos[0], j2_pos[2]])
        j3_log.append([j3_pos[0], j3_pos[2]])
        tip_log.append([tip_pos[0], tip_pos[2]])
        t_log.append(data.time)
        
        frame_counter += 1

print("Simulation finished. Processing animation...")

# Convert lists to numpy arrays for animation (Corrected)
pos_data = {
    'j2': np.array(j2_log), # Position of Joint 2
    'j3': np.array(j3_log), # Position of Joint 3
    'tip': np.array(tip_log), # Position of the tip
}
trail_data = pos_data['tip'] # Trail of the *actual* tip
Nt = len(t_log)

#================================================================
 # PART 3: ANIMATION 
#================================================================

# --- Figure setup ---
fig, ax = plt.subplots(figsize=(8,8))
ax.set_aspect('equal', 'box')
total_L = 3.0 # L1+L2+L3
margin = total_L * 1.1
ax.set_xlim(-margin, margin)
ax.set_ylim(-margin, margin) # Note: Y-axis is now MuJoCo's Z-axis
ax.set_xlabel('x (m)')
ax.set_ylabel('z (m)') # We plot x vs z
ax.set_title('Triple Pendulum Dynamics', fontsize=15)

# --- Graphic elements ---
rod1, = ax.plot([], [], 'o-', lw=2, markersize=3, color='C0') # Pivot to mass 1
rod2, = ax.plot([], [], 'o-', lw=2, markersize=3, color='C1') # Mass 1 to mass 2
rod3, = ax.plot([], [], 'o-', lw=2, markersize=3, color='C2') # Mass 2 to mass 3
trail_line, = ax.plot([], [], lw=1, color='orange', alpha=0.8)
pivot_dot, = ax.plot([0], [0], 'ko', markersize=5) # Pivot at (0,0)

def init():
    rod1.set_data([], [])
    rod2.set_data([], [])
    rod3.set_data([], [])
    trail_line.set_data([], [])
    return rod1, rod2, rod3, trail_line, pivot_dot

def update(frame):
    # Get coordinates for the current frame (Corrected)
    frame_x1, frame_y1 = pos_data['j2'][frame]  # End of rod 1
    frame_x2, frame_y2 = pos_data['j3'][frame]  # End of rod 2
    frame_x3, frame_y3 = pos_data['tip'][frame] # End of rod 3
    
    # Update rods (Corrected)
    # rod1 (blue) plots from Pivot to end of rod 1
    rod1.set_data([0, frame_x1], [0, frame_y1])
    # rod2 (orange) plots from end of rod 1 to end of rod 2
    rod2.set_data([frame_x1, frame_x2], [frame_y1, frame_y2])
    # rod3 (green) plots from end of rod 2 to end of rod 3
    rod3.set_data([frame_x2, frame_x3], [frame_y2, frame_y3])
    
    # Update trail
    trail_line.set_data(trail_data[:frame, 0], trail_data[:frame, 1])
    
    return rod1, rod2, rod3, trail_line, pivot_dot

# --- Create and save animation ---
frames_idx = range(Nt)
anim = FuncAnimation(fig, update, frames=frames_idx, init_func=init, blit=True)

output = 'triple_pendulum_mujoco.mp4'
print(f"Saving animation to {output}...")
writer = FFMpegWriter(fps=fps, bitrate=1800)
anim.save(output, writer=writer)
plt.close(fig)

print("Done.")


