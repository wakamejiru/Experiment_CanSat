import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

# --- 0. シミュレーション・物理パラメータ ---
DT = 0.01  # シミュレーション時間ステップ (s)

# CanSat パラメータ
MASS_CANSAT = 0.0641250  # kg
# 寸法 (m) (既存の length, width, height を使う)
CAN_LENGTH = 0.018 # X軸方向の長さと仮定 (慣性テンソル計算用)
CAN_WIDTH = 0.020   # Y軸方向
CAN_HEIGHT = 0.095  # Z軸方向

def transform_points(points, rotation_matrix, translation_vector):
    """点群を回転・平行移動 (points: Nx3 配列)"""
    # 1. 回転を適用
    #    points.T は (3,N) 配列になり、これに (3,3) の回転行列を左から掛ける
    rotated_points_transposed = rotation_matrix @ points.T
    # 結果は (3,N) 配列なので、再び転置して (N,3) に戻す
    rotated_points = rotated_points_transposed.T
    
    # 2. 平行移動を適用
    #    各点に平行移動ベクトルを加える
    #    (translation_vector が (3,) の場合、ブロードキャストされて各行に加算される)
    return rotated_points + translation_vector

"""
テンソアを計算
"""
def get_inertia_tensor_cuboid(mass, l, w, h):
    I_xx = (1/12) * mass * (w**2 + h**2)
    I_yy = (1/12) * mass * (l**2 + h**2)
    I_zz = (1/12) * mass * (l**2 + w**2)
    return np.diag([I_xx, I_yy, I_zz])


# --- 0. シミュレーション・物理パラメータ ---
DT = 0.01  # シミュレーション時間ステップ (s)
# (CanSat, CMGの物理パラメータ定義は変更なし)
MASS_CANSAT = 0.0641250; CAN_LENGTH = 0.018; CAN_WIDTH = 0.020; CAN_HEIGHT = 0.095
I_CANSAT_BODY = get_inertia_tensor_cuboid(MASS_CANSAT, CAN_LENGTH, CAN_WIDTH, CAN_HEIGHT)
I_CANSAT_BODY_INV = np.linalg.inv(I_CANSAT_BODY) # 今回は使わないが定義は残す
I_WHEEL = 8.48e-7; OMEGA_WHEEL_NOMINAL_RPM = 6000
OMEGA_WHEEL_NOMINAL_RAD_S = OMEGA_WHEEL_NOMINAL_RPM * (2 * np.pi) / 60
H_WHEEL_SCALAR = I_WHEEL * OMEGA_WHEEL_NOMINAL_RAD_S
GIMBAL_RATE_MAX_RAD_S = np.deg2rad(90)
KP_OMEGA = 0.005 # 制御ゲイン (調整が必要)
OUTER_GIMBAL_AXIS_BODY = np.array([0., 0., 1.])
INNER_GIMBAL_AXIS_IN_OUTER_FRAME = np.array([0., 1., 0.])
WHEEL_SPIN_AXIS_IN_INNER_FRAME = np.array([1., 0., 0.])
NUM_GIMBALS = 2

# --- 1. 初期状態とデータ ---
# 落下データ (dps) - これを仮想的なCanSatの角速度として使う
gyro_datas_x = [0.841, 0.833, 0.852, 0.848, 0.791, 0.781, 0.891, 0.802, 0.884, 0.843, 0.875, 0.734, 0.769, 0.829, 0.786, 0.919, 0.889, 0.807, 0.834, 0.927, 0.886, 0.963, 1.007, 0.892, 0.855, 0.765, 0.479, 0.424, 0.633, 0.674, 0.577, 0.496, 0.572, 0.648, 0.587, 0.371, 0.448, 0.487, 0.599, 0.622, 0.655, 0.455, 0.992, 0.644, 0.99, 1.021, 0.877, 0.821, 0.805, 0.865, 0.802, 0.796, 0.813, 0.716, 0.799, 0.802, 0.855, 0.913, 0.834, 0.839, 0.806, 0.819, 0.783, 0.834, 0.877, 0.854, 0.726, 0.746, 0.785, 0.771, 0.786, 0.787, 0.817, 0.701, 0.725, 0.645, 0.441, 0.57, 0.262, 0.184, 0.29, 0.399, 0.321, 0.263, 0.251, 0.306, 0.274, 0.303, 0.342, 0.304, 0.296, 0.316, 0.348, 0.335, 0.336, 0.372, 0.37, 0.372, 0.362, 0.342, 0.337, 0.345, 0.337, 0.318, 0.329, 0.303, 0.309, 0.318, 0.316, 0.322, 0.315, 0.325, 0.321, 0.309, 0.313, 0.332, 0.346, 0.31, 0.297, 0.278, 0.294, 0.317, 0.293, 0.277, 0.274, 0.271, 0.257, 0.278, 0.292, 0.303, 0.257, 0.289, 0.347, 0.375, 0.379, 0.315, 0.404, 0.465, 0.519, 0.494, 0.544, 0.526, 0.514, 0.52, 0.501, 0.531, 0.519, 0.599, 0.597, 0.576, 0.601, 0.574, 0.548, 0.543, 0.53, 0.566, 0.567, 0.637, 0.087, 0.137, 0.341, 0.457, 0.425, 0.532, 0.565, 0.505, 0.836, 0.717, 0.572, 0.349, 0.421, 0.542, 0.745, 0.76, 0.656, 0.586, 0.515, 0.382, 0.149, -0.022, 0.442, 0.516, -0.021, -0.553, -1, -0.268, -0.287, -0.219, -0.228, -0.375, -0.255, -0.25, -0.136, -0.204, -0.08, 0.059, -0.083, -0.041, -0.005, -0.016, -0.079, -0.056, 0.084, 0.038, 0.122, 0.098, 0.053, 0.005, -0.008, 0.148, 0.204, 0.268, 0.26, 0.234, 0.216, 0.249, 0.285, 0.152, 0.209, 0.37, 0.091, 0.419, 0.217, 0.334, 0.267, 0.339, 0.226, 0.307, 0.295]
gyro_datas_y = [-0.037, -0.041, -0.021, -0.168, 0.082, 0.094, 0.013, 0.119, 0.191, 0.134, 0.168, 0.168, 0.181, 0.181, 0.155, 0.16, 0.198, 0.354, 0.299, 0.443, 0.32, 0.229, -0.077, -0.103, -0.343, -0.419, -0.604, -0.685, -0.799, -0.767, -0.677, -0.654, -0.7, -0.82, -0.724, -0.132, -0.682, -0.729, -0.696, -0.49, -0.611, -0.141, -0.386, -0.132, -0.386, -0.263, -0.342, -0.332, -0.338, -0.341, -0.32, -0.358, -0.398, -0.422, -0.492, -0.425, -0.38, -0.438, -0.389, -0.358, -0.372, -0.333, -0.335, -0.365, -0.416, -0.42, -0.423, -0.468, -0.491, -0.46, -0.482, -0.466, -0.485, -0.382, -0.429, -0.337, -0.226, -0.61, -0.321, -0.329, -0.372, -0.472, -0.401, -0.34, -0.423, -0.451, -0.442, -0.446, -0.387, -0.362, -0.33, -0.382, -0.414, -0.404, -0.396, -0.396, -0.387, -0.407, -0.4, -0.377, -0.381, -0.377, -0.378, -0.381, -0.393, -0.372, -0.381, -0.377, -0.386, -0.397, -0.393, -0.403, -0.386, -0.386, -0.396, -0.429, -0.405, -0.44, -0.398, -0.395, -0.447, -0.502, -0.502, -0.485, -0.463, -0.456, -0.445, -0.466, -0.421, -0.409, -0.425, -0.503, -0.478, -0.479, -0.607, -0.479, -0.545, -0.515, -0.625, -0.445, -0.495, -0.382, -0.329, -0.362, -0.359, -0.396, -0.385, -0.411, -0.388, -0.352, -0.365, -0.357, -0.319, -0.332, -0.312, -0.31, -0.29, -0.297, -0.039, -0.064, -0.162, -0.325, -0.291, -0.373, -0.3, -0.139, -0.478, -0.261, -0.475, -0.412, -0.516, -0.517, -0.724, -0.706, -0.525, -0.669, -0.663, -0.384, -0.45, -0.929, -0.387, -0.301, -5.189, -0.387, -1.412, -0.163, -0.353, -0.575, -0.495, -0.298, -0.388, -0.491, -0.437, -0.397, -0.5, -0.48, -0.546, -0.377, -0.421, -0.449, -0.406, -0.521, -0.504, -0.531, -0.338, -0.284, -0.317, -0.273, -0.189, -0.316, -0.219, -0.153, -0.022, -0.165, -0.145, -0.042, 0.033, -0.049, -0.019, -0.007, -0.034, -0.057, -0.254, -0.232, -0.39, -0.266, -0.316, -0.068, -0.146]
gyro_datas_z = [-0.609,-0.585,-0.604,-0.622,-0.567,-0.607,-0.665,-0.596,-0.662,-0.645,-0.665,-0.58,-0.606,-0.647,-0.606,-0.688,-0.675,-0.571,-0.438,-0.42,-0.331,-0.332,-0.257,-0.153,-0.271,-0.525,-0.561,-0.491,-0.375,-0.342,-0.509,-0.503,-0.445,-0.385,-0.281,-0.413,-0.937,-0.759,-0.703,-0.636,-0.66,-0.433,-0.907,-0.557,-0.663,-0.632,-0.566,-0.481,-0.457,-0.498,-0.476,-0.481,-0.451,-0.424,-0.482,-0.47,-0.51,-0.545,-0.492,-0.49,-0.448,-0.443,-0.425,-0.464,-0.562,-0.514,-0.479,-0.513,-0.559,-0.541,-0.556,-0.553,-0.585,-0.535,-0.578,-0.553,-0.454,-1.005,-0.917,-0.906,-0.892,-1.016,-0.896,-0.826,-0.962,-0.908,-0.893,-0.987,-1.023,-0.888,-0.875,-0.914,-0.998,-0.909,-0.895,-0.957,-0.948,-0.999,-0.913,-0.889,-0.898,-0.929,-0.923,-0.908,-0.919,-0.891,-0.928,-0.949,-0.927,-0.918,-0.897,-0.916,-0.91,-0.885,-0.917,-0.958,-0.936,-0.883,-0.844,-0.874,-0.885,-0.952,-0.917,-0.887,-0.935,-0.879,-0.86,-0.913,-0.921,-0.92,-0.829,-0.857,-0.75,-0.742,-0.895,-0.779,-0.937,-0.807,-0.847,-0.743,-0.838,-0.808,-0.762,-0.755,-0.781,-0.815,-0.787,-0.871,-0.856,-0.826,-0.872,-0.839,-0.782,-0.781,-0.745,-0.775,-0.758,-0.838,-0.135,-0.272,-0.296,-0.435,-0.39,-0.478,-0.63,-0.583,-0.605,-0.833,-0.787,-0.75,-1.035,-1.058,-0.988,-0.891,-0.895,-1.134,-0.953,-0.635,-0.725,-1.155,-0.602,-1.457,7.93,1.188,-1.413,-0.945,-0.962,-0.753,-0.701,-0.892,-0.803,-0.795,-0.901,-0.839,-0.926,-0.919,-1.081,-0.89,-0.883,-0.906,-0.885,-0.961,-1.133,-1.193,-0.998,-0.84,-0.955,-0.993,-1.011,-1.072,-1.112,-1.021,-1.031,-0.983,-0.884,-1.005,-1.124,-1.123,-0.994,-0.851,-0.941,-1.175,-1.043,-0.967,-0.983,-0.94,-1.009,-0.969,-0.982]
num_steps = len(gyro_datas_x)
time_hist = np.linspace(0, (num_steps - 1) * DT, num_steps)

# ★ CanSatの初期姿勢と角速度は固定 (静止、基準姿勢)
initial_omega_body_rad_s = np.array([0.0, 0.0, 0.0])
initial_orientation = R.from_euler('xyz', [0,0,0], degrees=True)

print(f"Number of simulation steps: {num_steps}")
print(f"Initial CanSat angular velocity (rad/s): {initial_omega_body_rad_s} (FIXED)")
print(f"Control Gain KP_OMEGA: {KP_OMEGA}")
print(f"H_WHEEL_SCALAR: {H_WHEEL_SCALAR:.2e} Nms")

# --- 2. 状態変数 (CanSatの状態は固定だが、履歴は一応用意) ---
omega_body_hist_rad_s = np.zeros((num_steps, 3))
orientation_hist_quat = np.zeros((num_steps, 4))
euler_hist_deg = np.zeros((num_steps, 3))
gimbal_angles_rad_hist = np.zeros((num_steps, NUM_GIMBALS))
gimbal_rates_rad_s_hist = np.zeros((num_steps, NUM_GIMBALS))
cmg_torque_hist_body = np.zeros((num_steps, 3)) # CMGが発生しようとするトルク

# 初期値設定
# ★ CanSatの現在の角速度と姿勢は常に初期値 (固定)
current_omega_body_rad_s_cansat_fixed = initial_omega_body_rad_s.copy() # CanSat自体は動かない
current_orientation_cansat_fixed = initial_orientation # CanSat自体は動かない

current_gimbal_angles_rad = np.zeros(NUM_GIMBALS)

# 履歴の初期値
omega_body_hist_rad_s[0, :] = current_omega_body_rad_s_cansat_fixed
orientation_hist_quat[0, :] = current_orientation_cansat_fixed.as_quat()
euler_hist_deg[0, :] = current_orientation_cansat_fixed.as_euler('xyz', degrees=True)
gimbal_angles_rad_hist[0, :] = current_gimbal_angles_rad

# --- 3. シミュレーションループ ---
print("Starting simulation loop (CanSat fixed, observing CMG reaction)...")
for i in range(1, num_steps):
    t = time_hist[i]

    # --- ★ 3a. CMG制御への入力となる「仮想的な」CanSatの角速度 ---
    # 落下データから、この時刻のCanSatの角速度(rad/s)を取得
    # (データ点数がシミュレーションステップと一致することを前提)
    virtual_omega_x_rad_s = np.deg2rad(gyro_datas_x[i])
    virtual_omega_y_rad_s = np.deg2rad(gyro_datas_y[i])
    virtual_omega_z_rad_s = np.deg2rad(gyro_datas_z[i])
    virtual_cansat_omega_body = np.array([virtual_omega_x_rad_s, virtual_omega_y_rad_s, virtual_omega_z_rad_s])

    # CMGはこの仮想的な角速度を打ち消そうとする
    target_torque_body = -KP_OMEGA * virtual_cansat_omega_body

    # --- CMGのジンバル角度とホイール角運動量の計算 (変更なし) ---
    outer_angle = current_gimbal_angles_rad[0]
    inner_angle = current_gimbal_angles_rad[1]
    R_o_from_b = R.from_rotvec(OUTER_GIMBAL_AXIS_BODY * outer_angle)
    R_i_from_o_frame = R.from_rotvec(INNER_GIMBAL_AXIS_IN_OUTER_FRAME * inner_angle)
    R_wheel_orientation_in_body = R_o_from_b * R_i_from_o_frame
    h_wheel_at_rest_in_inner_coord = WHEEL_SPIN_AXIS_IN_INNER_FRAME * H_WHEEL_SCALAR
    current_h_wheel_body = R_wheel_orientation_in_body.apply(h_wheel_at_rest_in_inner_coord)

    A_jacobian = np.zeros((3, NUM_GIMBALS))
    A_jacobian[:, 0] = np.cross(OUTER_GIMBAL_AXIS_BODY, current_h_wheel_body)
    current_inner_gimbal_axis_in_body_for_jacobian = R_o_from_b.apply(INNER_GIMBAL_AXIS_IN_OUTER_FRAME) # ★ Jac.用に再計算
    A_jacobian[:, 1] = np.cross(current_inner_gimbal_axis_in_body_for_jacobian, current_h_wheel_body)

    try:
        lambda_sr = 1e-4 # 正則化項を調整
        A_pinv = A_jacobian.T @ np.linalg.inv(A_jacobian @ A_jacobian.T + lambda_sr * np.eye(3))
        commanded_gimbal_rates_rad_s = A_pinv @ target_torque_body
    except np.linalg.LinAlgError:
        commanded_gimbal_rates_rad_s = np.zeros(NUM_GIMBALS)
    
    actual_gimbal_rates_rad_s = np.clip(commanded_gimbal_rates_rad_s, -GIMBAL_RATE_MAX_RAD_S, GIMBAL_RATE_MAX_RAD_S)

    # --- ★ 3b. CMGが発生しようとするトルク (CanSatには加えない) ---
    simulated_cmg_torque_body = A_jacobian @ actual_gimbal_rates_rad_s

    # --- ★ 3c. CanSatの運動方程式は計算しない (CanSatは固定) ---
    # omega_dot_body = ... (この部分は実行しない)

    # --- ★ 3d. 数値積分 (CMGのジンバル角度のみ更新) ---
    # current_omega_body_rad_s は更新しない (常に initial_omega_body_rad_s)
    # current_orientation は更新しない (常に initial_orientation)
    current_gimbal_angles_rad += actual_gimbal_rates_rad_s * DT

    # --- 3e. データ記録 ---
    omega_body_hist_rad_s[i, :] = virtual_cansat_omega_body # 仮想的な角速度を記録
    orientation_hist_quat[i, :] = initial_orientation.as_quat() # 固定された姿勢を記録
    euler_hist_deg[i, :] = initial_orientation.as_euler('xyz', degrees=True) # 固定された姿勢を記録
    gimbal_angles_rad_hist[i, :] = current_gimbal_angles_rad
    gimbal_rates_rad_s_hist[i, :] = actual_gimbal_rates_rad_s
    cmg_torque_hist_body[i, :] = simulated_cmg_torque_body # CMGが発生しようとしたトルク

print("Simulation loop finished.")

# --- 4. 結果のプロット ---
# (グラフ描画部分はほぼ同じだが、タイトルやラベルを調整)
fig_plots, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
axs[0].plot(time_hist, np.rad2deg(omega_body_hist_rad_s[:,0]), label=r'$\omega_x$ (Virtual CanSat)')
axs[0].plot(time_hist, np.rad2deg(omega_body_hist_rad_s[:,1]), label=r'$\omega_y$ (Virtual CanSat)')
axs[0].plot(time_hist, np.rad2deg(omega_body_hist_rad_s[:,2]), label=r'$\omega_z$ (Virtual CanSat)')
axs[0].set_ylabel('Virtual Ang. Vel. (deg/s)'); axs[0].legend(); axs[0].grid(True)
axs[0].set_title('CMG Reaction to Virtual CanSat Angular Velocity (CanSat Fixed)')

axs[1].plot(time_hist, euler_hist_deg[:,0], label='Roll (Fixed CanSat)')
axs[1].plot(time_hist, euler_hist_deg[:,1], label='Pitch (Fixed CanSat)')
axs[1].plot(time_hist, euler_hist_deg[:,2], label='Yaw (Fixed CanSat)')
axs[1].set_ylabel('Fixed Euler Angles (deg)'); axs[1].legend(); axs[1].grid(True)

axs[2].plot(time_hist, np.rad2deg(gimbal_angles_rad_hist[:, 0]), label='Outer Gimbal Angle')
axs[2].plot(time_hist, np.rad2deg(gimbal_angles_rad_hist[:, 1]), label='Inner Gimbal Angle')
axs[2].set_ylabel('Gimbal Angles (deg)'); axs[2].legend(); axs[2].grid(True)

axs[3].plot(time_hist, cmg_torque_hist_body[:,0], label='$T_{x}$ (CMG Output)')
axs[3].plot(time_hist, cmg_torque_hist_body[:,1], label='$T_{y}$ (CMG Output)')
axs[3].plot(time_hist, cmg_torque_hist_body[:,2], label='$T_{z}$ (CMG Output)')
axs[3].set_ylabel('Simulated CMG Torque (Nm)'); axs[3].set_xlabel('Time (s)'); axs[3].legend(); axs[3].grid(True)
plt.tight_layout(); plt.savefig("cansat_fixed_cmg_reaction_plots.png"); plt.show()


# --- 5. アニメーション (CanSat固定、CMGの動きのみ) ---
vertices_body_frame = np.array([
    [-CAN_LENGTH/2, -CAN_WIDTH/2, -CAN_HEIGHT/2], [ CAN_LENGTH/2, -CAN_WIDTH/2, -CAN_HEIGHT/2],
    [ CAN_LENGTH/2,  CAN_WIDTH/2, -CAN_HEIGHT/2], [-CAN_LENGTH/2,  CAN_WIDTH/2, -CAN_HEIGHT/2],
    [-CAN_LENGTH/2, -CAN_WIDTH/2,  CAN_HEIGHT/2], [ CAN_LENGTH/2, -CAN_WIDTH/2,  CAN_HEIGHT/2],
    [ CAN_LENGTH/2,  CAN_WIDTH/2,  CAN_HEIGHT/2], [-CAN_LENGTH/2,  CAN_WIDTH/2,  CAN_HEIGHT/2]
])
WHEEL_RADIUS = 0.015; WHEEL_THICKNESS = 0.003; SPOKE_COLOR = 'red'; WHEEL_EDGE_COLOR = 'darkgray'; WHEEL_FACE_COLOR = 'gold'
fig_anim_fixed_cansat = plt.figure(figsize=(8, 8))

def create_cylinder_faces(radius, height, n_segments=12): # セグメント数を減らして描画負荷軽減
    theta = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    x = radius * np.cos(theta); y = radius * np.sin(theta)
    z_top = height / 2; z_bottom = -height / 2
    bottom_verts = np.vstack((x, y, np.full(n_segments, z_bottom))).T
    top_verts = np.vstack((x, y, np.full(n_segments, z_top))).T
    faces = []
    for i in range(n_segments):
        i_next = (i + 1) % n_segments
        faces.append([bottom_verts[i], bottom_verts[i_next], top_verts[i_next], top_verts[i]])
    faces.append(bottom_verts[::-1]); faces.append(top_verts)
    return faces

base_wheel_faces = create_cylinder_faces(WHEEL_RADIUS, WHEEL_THICKNESS)
base_spoke_x = np.array([[0,0,0],[WHEEL_RADIUS*0.9,0,0]]) # XY平面上のX軸方向の線分
base_spoke_y = np.array([[0,0,0],[0,WHEEL_RADIUS*0.9,0]]) # XY平面上のY軸方向の線分
fig_anim_dgcmg = plt.figure(figsize=(8, 8)); ax_anim_dgcmg = fig_anim_dgcmg.add_subplot(111, projection='3d')
ax_anim_dgcmg.view_init(elev=30, azim=-60)

def update_animation_dgcmg(frame):
    ax_anim_dgcmg.cla()
    ax_anim_dgcmg.set_xlim([-0.05, 0.05]); ax_anim_dgcmg.set_ylim([-0.05, 0.05]); ax_anim_dgcmg.set_zlim([-0.07, 0.07])
    ax_anim_dgcmg.set_xlabel("X_i"); ax_anim_dgcmg.set_ylabel("Y_i"); ax_anim_dgcmg.set_zlabel("Z_i")
    ax_anim_dgcmg.set_title(f"CanSat & DGCMG, T: {time_hist[frame]:.2f}s")

    q_cansat = orientation_hist_quat[frame, :]
    R_cansat_inertial_mat = R.from_quat(q_cansat).as_matrix()
    rotated_cansat_verts = (R_cansat_inertial_mat @ vertices_body_frame.T).T
    

fig_anim_fixed_cansat = plt.figure(figsize=(8, 8))
ax_anim_fixed_cansat = fig_anim_fixed_cansat.add_subplot(111, projection='3d')
ax_anim_fixed_cansat.view_init(elev=30, azim=-60)

def update_animation_fixed_cansat_cmg_only(frame):
    ax_anim_fixed_cansat.cla()
    ax_anim_fixed_cansat.set_xlim([-0.06, 0.06]); ax_anim_fixed_cansat.set_ylim([-0.06, 0.06]); ax_anim_fixed_cansat.set_zlim([-0.08, 0.08])
    ax_anim_fixed_cansat.set_xlabel("X_body/inertial"); ax_anim_fixed_cansat.set_ylabel("Y_body/inertial"); ax_anim_fixed_cansat.set_zlabel("Z_body/inertial")
    ax_anim_fixed_cansat.set_title(f"CMG Reaction (CanSat Fixed), Time: {time_hist[frame]:.2f}s")

    # CanSat本体は常に初期姿勢で描画
    R_cansat_fixed_inertial_mat = initial_orientation.as_matrix() # initial_orientation はグローバルスコープで定義された固定姿勢
    rotated_cansat_vertices = (R_cansat_fixed_inertial_mat @ vertices_body_frame.T).T
    cansat_faces_def = [
        [rotated_cansat_vertices[0], rotated_cansat_vertices[1], rotated_cansat_vertices[2], rotated_cansat_vertices[3]],
        [rotated_cansat_vertices[4], rotated_cansat_vertices[5], rotated_cansat_vertices[6], rotated_cansat_vertices[7]],
        [rotated_cansat_vertices[0], rotated_cansat_vertices[1], rotated_cansat_vertices[5], rotated_cansat_vertices[4]],
        [rotated_cansat_vertices[2], rotated_cansat_vertices[3], rotated_cansat_vertices[7], rotated_cansat_vertices[6]],
        [rotated_cansat_vertices[1], rotated_cansat_vertices[2], rotated_cansat_vertices[6], rotated_cansat_vertices[5]],
        [rotated_cansat_vertices[0], rotated_cansat_vertices[3], rotated_cansat_vertices[7], rotated_cansat_vertices[4]]
    ]
    cansat_poly3d = Poly3DCollection(cansat_faces_def, facecolors='lightskyblue', linewidths=0.5, edgecolors='darkblue', alpha=0.1)
    ax_anim_fixed_cansat.add_collection3d(cansat_poly3d)

    outer_ang = gimbal_angles_rad_hist[frame, 0]
    inner_ang = gimbal_angles_rad_hist[frame, 1]

    R_o_from_b = R.from_rotvec(OUTER_GIMBAL_AXIS_BODY * outer_ang)
    R_i_from_o_frame = R.from_rotvec(INNER_GIMBAL_AXIS_IN_OUTER_FRAME * inner_ang)

    R_align_stdZ_to_wheelspin_mat = np.eye(3) # (R_align_stdZ_to_wheelspin の計算は前回のものを流用)
    target_spin_axis_in_i = WHEEL_SPIN_AXIS_IN_INNER_FRAME / np.linalg.norm(WHEEL_SPIN_AXIS_IN_INNER_FRAME)
    if not np.allclose([0,0,1], target_spin_axis_in_i):
        if np.allclose([0,0,1], -target_spin_axis_in_i):
            ortho_ax = np.cross([0,0,1], np.array([1.,0.,0.]));
            if np.linalg.norm(ortho_ax) < 1e-6: ortho_ax = np.cross([0,0,1], np.array([0.,1.,0.]))
            if np.linalg.norm(ortho_ax) > 1e-6: R_align_stdZ_to_wheelspin_mat = R.from_rotvec(np.pi * ortho_ax / np.linalg.norm(ortho_ax)).as_matrix()
            else: R_align_stdZ_to_wheelspin_mat = np.diag([-1.,-1.,1.])
        else:
            rot_obj_align, _ = R.align_vectors(target_spin_axis_in_i[np.newaxis,:], np.array([[0.,0.,1.]]))
            R_align_stdZ_to_wheelspin_mat = rot_obj_align.as_matrix()
    
    # ホイールの向き (ボディ座標系内で)
    R_wheel_orientation_in_body_obj = R_o_from_b * R_i_from_o_frame * R.from_matrix(R_align_stdZ_to_wheelspin_mat)

    # ★★★ R_wheel_in_inertial_obj と cmg_origin_inertial の定義 ★★★
    # CanSatは固定なので、ボディ座標系での向きがそのまま慣性系での向きになる (CanSatの基準姿勢が慣性系と一致する場合)
    # initial_orientation はCanSatの固定姿勢を表す scipy.Rotation オブジェクト
    R_wheel_in_inertial_obj = initial_orientation * R_wheel_orientation_in_body_obj # ボディ内での向きをさらにCanSatの固定姿勢で回転
    
    cmg_origin_body_coord = np.array([0.,0.,0.]) # CMGはCanSat中心と仮定
    cmg_origin_inertial = initial_orientation.apply(cmg_origin_body_coord) # CanSat固定姿勢でのCMG原点

    # ホイールの面を描画
    wheel_faces_inertial_plot = []
    for face_verts_base in base_wheel_faces:
        wheel_faces_inertial_plot.append(transform_points(np.array(face_verts_base), R_wheel_in_inertial_obj, cmg_origin_inertial))
    
    ax_anim_fixed_cansat.add_collection3d(Poly3DCollection(wheel_faces_inertial_plot[:-2], facecolors=WHEEL_FACE_COLOR, edgecolor=WHEEL_EDGE_COLOR, alpha=0.9, linewidths=0.3))
    ax_anim_fixed_cansat.add_collection3d(Poly3DCollection([wheel_faces_inertial_plot[-2]], facecolors=WHEEL_FACE_COLOR, edgecolor=WHEEL_EDGE_COLOR, alpha=0.9, linewidths=0.3))
    ax_anim_fixed_cansat.add_collection3d(Poly3DCollection([wheel_faces_inertial_plot[-1]], facecolors=WHEEL_FACE_COLOR, edgecolor=WHEEL_EDGE_COLOR, alpha=0.9, linewidths=0.3))

    # スポークの描画
    spin_vis_factor = 0.1
    spin_angle_vis = (frame * DT * OMEGA_WHEEL_NOMINAL_RAD_S * spin_vis_factor) % (2*np.pi)
    R_spoke_spin_local = R.from_rotvec(np.array([0,0,1]) * spin_angle_vis)

    for base_spk in [base_spoke_x, base_spoke_y]:
        spoke_spun_local = R_spoke_spin_local.apply(base_spk)
        spoke_inertial_plot = transform_points(spoke_spun_local, R_wheel_in_inertial_obj, cmg_origin_inertial)
        ax_anim_fixed_cansat.plot(spoke_inertial_plot[:,0], spoke_inertial_plot[:,1], spoke_inertial_plot[:,2], color=SPOKE_COLOR, linewidth=1.2)
    return []

print("Creating animation (CanSat fixed)...")
ani_fixed_cansat = FuncAnimation(fig_anim_fixed_cansat, update_animation_fixed_cansat_cmg_only, frames=num_steps, blit=False, interval=max(1, int(DT*1000)))
# try:
#     writer = FFMpegWriter(fps=max(1,int(1/DT)))
#     ani_fixed_cansat.save("cansat_fixed_cmg_reaction_animation.mp4", writer=writer)
#     print("Animation saved to cansat_fixed_cmg_reaction_animation.mp4")
# except Exception as e:
#     print(f"Could not save animation: {e}")
plt.show()
print("Animation part finished.")