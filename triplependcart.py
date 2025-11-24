import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
import mujoco

# ================================================================
#   PART 1: THE XML MODEL (MODIFIED)
# ================================================================
triple_pendulum_xml = """
<mujoco model="triple_pendulum_cart"> 
  <option gravity="0 0 -9.81" timestep="0.001" />

  <worldbody>
    <geom name="floor" type="box" pos="0 0 -0.1" size="20 10 0.1" rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0" />
    <light name="top" pos="0 0 5" />

    <body name="cart" pos="0 0 0">
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01" />
      <joint name="cart_slider" type="slide" axis="1 0 0" pos="0 0 0" />
      <geom name="cart_geom" type="box" size="0.2 0.1 0.1" rgba="0.2 0.2 0.8 1" />
      
      <body name="body1" pos="0 0 0">
        <inertial pos="0 0 0.5" mass="1" diaginertia="0.333 0.333 0.001" />
        <joint name="joint1" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.1" />
        <geom name="rod1" type="capsule" fromto="0 0 0 0 0 1" size="0.03" rgba="0.8 0.2 0.2 1" />

        <body name="body2" pos="0 0.05 1">
          <inertial pos="0 0 0.5" mass="1" diaginertia="0.333 0.333 0.001" />
          <joint name="joint2" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.1" />
          <geom name="rod2" type="capsule" fromto="0 0 0 0 0 1" size="0.03" rgba="0.2 0.8 0.2 1" />

          <body name="body3" pos="0 0.05 1">
            <inertial pos="0 0 0.5" mass="1" diaginertia="0.333 0.333 0.001" />
            <joint name="joint3" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.1" />
            <geom name="rod3" type="capsule" fromto="0 0 0 0 0 1" size="0.03" rgba="0.2 0.2 0.8 1" />
            <site name="tip" pos="0 0 1" type="sphere" size="0.05" rgba="0.2 0.2 0.8 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="slide_motor" joint="cart_slider" ctrlrange="-100 100"/>
  </actuator>
</mujoco>
"""

# ================================================================
#   PART 2: MUJOCO SIMULATION
# ================================================================

# --- Numerical parameters ---
T = 15.0       # Total simulation time (s)
fps = 30       
sim_dt = 0.001 
anim_dt = 1.0 / fps 

# --- Load Model ---
print("Loading model from XML...")
model = mujoco.MjModel.from_xml_string(triple_pendulum_xml)
data = mujoco.MjData(model)

# --- Initial conditions (MODIFIED) ---
# qpos map: [cart_pos, theta1, theta2, theta3]
# We set them to 0 (Upright) with a small disturbance to test controller
data.qpos[0] = 0.0    # Cart at 0
data.qpos[1] = 0.05   # Small initial disturbance on rod 1
data.qpos[2] = 0.02   # Tiny disturbance on rod 2
data.qpos[3] = 0.0    # Rod 3 upright

# --- PID Controller Gains ---
# PD gains for each pendulum angle (Kp, Kd)
Kp1, Kd1 = 100.0, 50.0   # Joint 1 (closest to cart)
Kp2, Kd2 = 80.0, 40.0    # Joint 2 (middle)
Kp3, Kd3 = 60.0, 30.0    # Joint 3 (tip)
# PD gains for cart position
Kp_cart, Kd_cart = 5.0, 15.0

# --- Data Logging (MODIFIED) ---
cart_log = [] # New log for cart
j2_log, j3_log, tip_log = [], [], []
t_log = []

# --- Run the Simulation Loop ---
print("Running MuJoCo simulation...")
frame_counter = 0
while data.time < T:
    # PID Control to keep pendulum upright
    # State: qpos = [cart_pos, theta1, theta2, theta3]
    #        qvel = [cart_vel, theta1_dot, theta2_dot, theta3_dot]

    # For inverted pendulum: positive angle feedback (move cart toward tilt)
    # with derivative damping
    u = (Kp1 * data.qpos[1] - Kd1 * data.qvel[1]) \
        + (Kp2 * data.qpos[2] - Kd2 * data.qvel[2]) \
        + (Kp3 * data.qpos[3] - Kd3 * data.qvel[3]) \
        - (Kp_cart * data.qpos[0] + Kd_cart * data.qvel[0])

    # Saturate control (limit to motor range)
    u = np.clip(u, -100, 100)
    data.ctrl[0] = u
    
    mujoco.mj_step(model, data)

    if data.time >= frame_counter * anim_dt:
        # Log Cart Position
        cart_log.append(data.body("cart").xpos)
        
        # Log Rod Ends
        j2_pos = data.body("body2").xpos
        j3_pos = data.body("body3").xpos
        tip_pos = data.site("tip").xpos
        
        j2_log.append([j2_pos[0], j2_pos[2]])
        j3_log.append([j3_pos[0], j3_pos[2]])
        tip_log.append([tip_pos[0], tip_pos[2]])
        t_log.append(data.time)
        
        frame_counter += 1

print("Simulation finished. Processing animation...")

pos_data = {
    'cart': np.array(cart_log),
    'j2': np.array(j2_log),
    'j3': np.array(j3_log),
    'tip': np.array(tip_log),
}
trail_data = pos_data['tip']
Nt = len(t_log)

#================================================================
 # PART 3: ANIMATION (MODIFIED)
#================================================================

# --- Figure setup ---
fig, ax = plt.subplots(figsize=(10,6))
ax.set_aspect('equal', 'box')
ax.set_ylim(-4, 4) # Adjusted for upright pendulum
ax.set_xlabel('x (m)')
ax.set_ylabel('z (m)')
ax.set_title('Triple Inverted Pendulum on Cart', fontsize=15)

# --- Graphic elements ---
rod1, = ax.plot([], [], 'o-', lw=2, markersize=3, color='C0')
rod2, = ax.plot([], [], 'o-', lw=2, markersize=3, color='C1')
rod3, = ax.plot([], [], 'o-', lw=2, markersize=3, color='C2')
trail_line, = ax.plot([], [], lw=1, color='orange', alpha=0.8)
floor_line, = ax.plot([-100, 100], [-0.1, -0.1], 'k-', lw=2) # Floor

# Cart Rectangle
cart_width, cart_height = 0.4, 0.2
cart_patch = patches.Rectangle((0, 0), cart_width, cart_height, fc='blue', alpha=0.8)
ax.add_patch(cart_patch)

def init():
    rod1.set_data([], [])
    rod2.set_data([], [])
    rod3.set_data([], [])
    trail_line.set_data([], [])
    cart_patch.set_xy((-cart_width/2, -cart_height/2))
    return rod1, rod2, rod3, trail_line, cart_patch

def update(frame):
    # Get positions
    cart_pos = pos_data['cart'][frame] # [x, y, z]
    cx, cz = cart_pos[0], cart_pos[2]
    
    p1 = pos_data['j2'][frame]
    p2 = pos_data['j3'][frame]
    p3 = pos_data['tip'][frame]
    
    # Update Cart
    cart_patch.set_xy((cx - cart_width/2, cz - cart_height/2))
    
    # Update Rods
    # Rod 1 goes from Cart -> J2
    rod1.set_data([cx, p1[0]], [cz, p1[1]])
    rod2.set_data([p1[0], p2[0]], [p1[1], p2[1]])
    rod3.set_data([p2[0], p3[0]], [p2[1], p3[1]])
    
    # Update Trail
    trail_line.set_data(trail_data[:frame, 0], trail_data[:frame, 1])
    
    # CAMERA FOLLOW: Center the X-axis on the cart
    window = 4.0
    ax.set_xlim(cx - window, cx + window)
    
    return rod1, rod2, rod3, trail_line, cart_patch

# --- Create and save animation ---
frames_idx = range(Nt)
anim = FuncAnimation(fig, update, frames=frames_idx, init_func=init, blit=True)

output = 'triple_cart_pendulum.mp4'
print(f"Saving animation to {output}...")
writer = FFMpegWriter(fps=fps, bitrate=1800)
anim.save(output, writer=writer)
plt.close(fig)

print("Done.")