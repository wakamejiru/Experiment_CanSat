import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter # 既存
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # 既存
from scipy.spatial.transform import Rotation as R

# --- 0. シミュレーション・物理パラメータ ---
DT = 0.01  # シミュレーション時間ステップ (s)
TOTAL_TIME = 10.0  # 総シミュレーション時間 (s)

# CanSat パラメータ (例)
MASS_CANSAT = 0.0641250  # kg
# 寸法 (m) (既存の length, width, height を使う)
CAN_LENGTH = 0.018 # X軸方向の長さと仮定 (慣性テンソル計算用)
CAN_WIDTH = 0.020   # Y軸方向
CAN_HEIGHT = 0.095  # Z軸方向



def get_inertia_tensor_cuboid(mass, l, w, h):
    I_xx = (1/12) * mass * (w**2 + h**2)
    I_yy = (1/12) * mass * (l**2 + h**2)
    I_zz = (1/12) * mass * (l**2 + w**2)
    return np.diag([I_xx, I_yy, I_zz])

I_CANSAT_BODY = get_inertia_tensor_cuboid(MASS_CANSAT, CAN_LENGTH, CAN_WIDTH, CAN_HEIGHT)
I_CANSAT_BODY_INV = np.linalg.inv(I_CANSAT_BODY)

# CMG パラメータ (例 - 3基共通とする)
I_WHEEL = 2e-5  # ホイールの慣性モーメント (kg*m^2)
OMEGA_WHEEL_NOMINAL_RPM = 6000  # ホイール定格回転数 (rpm)
OMEGA_WHEEL_NOMINAL_RAD_S = OMEGA_WHEEL_NOMINAL_RPM * (2 * np.pi) / 60
H_WHEEL_SCALAR = I_WHEEL * OMEGA_WHEEL_NOMINAL_RAD_S # ホイールの角運動量の大きさ
GIMBAL_RATE_MAX_DPS = 90  # ジンバル最大角速度 (deg/s)
GIMBAL_RATE_MAX_RAD_S = np.deg2rad(GIMBAL_RATE_MAX_DPS)

# 制御ゲイン
KP_OMEGA = 0.01 # 角速度制御の比例ゲイン (要調整)

# CMG配置: [ジンバル軸(物体座標系), ジンバル角0度でのホイールスピン軸(物体座標系)]
CMG_CONFIG = [
    {'gimbal_axis': np.array([1., 0., 0.]), 'initial_spin_axis': np.array([0., 1., 0.])}, # CMG1: Xジンバル, Yスピン
    {'gimbal_axis': np.array([0., 1., 0.]), 'initial_spin_axis': np.array([0., 0., 1.])}, # CMG2: Yジンバル, Zスピン
    {'gimbal_axis': np.array([0., 0., 1.]), 'initial_spin_axis': np.array([1., 0., 0.])}, # CMG3: Zジンバル, Xスピン
]
NUM_CMGS = len(CMG_CONFIG)

# 観測された角速度データ (dps単位のリスト gyro_datas_x, gyro_datas_y, gyro_datas_z があると仮定)
gyro_datas_x = [0.841, 0.833, 0.852, 0.848, 0.791, 0.781, 0.891, 0.802, 0.884, 0.843, 0.875, 0.734, 0.769, 0.829, 0.786, 0.919, 0.889, 0.807, 0.834, 0.927, 0.886, 0.963, 1.007, 0.892, 0.855, 0.765, 0.479, 0.424, 0.633, 0.674, 0.577, 0.496, 0.572, 0.648, 0.587, 0.371, 0.448, 0.487, 0.599, 0.622, 0.655, 0.455, 0.992, 0.644, 0.99, 1.021, 0.877, 0.821, 0.805, 0.865, 0.802, 0.796, 0.813, 0.716, 0.799, 0.802, 0.855, 0.913, 0.834, 0.839, 0.806, 0.819, 0.783, 0.834, 0.877, 0.854, 0.726, 0.746, 0.785, 0.771, 0.786, 0.787, 0.817, 0.701, 0.725, 0.645, 0.441, 0.57, 0.262, 0.184, 0.29, 0.399, 0.321, 0.263, 0.251, 0.306, 0.274, 0.303, 0.342, 0.304, 0.296, 0.316, 0.348, 0.335, 0.336, 0.372, 0.37, 0.372, 0.362, 0.342, 0.337, 0.345, 0.337, 0.318, 0.329, 0.303, 0.309, 0.318, 0.316, 0.322, 0.315, 0.325, 0.321, 0.309, 0.313, 0.332, 0.346, 0.31, 0.297, 0.278, 0.294, 0.317, 0.293, 0.277, 0.274, 0.271, 0.257, 0.278, 0.292, 0.303, 0.257, 0.289, 0.347, 0.375, 0.379, 0.315, 0.404, 0.465, 0.519, 0.494, 0.544, 0.526, 0.514, 0.52, 0.501, 0.531, 0.519, 0.599, 0.597, 0.576, 0.601, 0.574, 0.548, 0.543, 0.53, 0.566, 0.567, 0.637, 0.087, 0.137, 0.341, 0.457, 0.425, 0.532, 0.565, 0.505, 0.836, 0.717, 0.572, 0.349, 0.421, 0.542, 0.745, 0.76, 0.656, 0.586, 0.515, 0.382, 0.149, -0.022, 0.442, 0.516, -0.021, -0.553, -1, -0.268, -0.287, -0.219, -0.228, -0.375, -0.255, -0.25, -0.136, -0.204, -0.08, 0.059, -0.083, -0.041, -0.005, -0.016, -0.079, -0.056, 0.084, 0.038, 0.122, 0.098, 0.053, 0.005, -0.008, 0.148, 0.204, 0.268, 0.26, 0.234, 0.216, 0.249, 0.285, 0.152, 0.209, 0.37, 0.091, 0.419, 0.217, 0.334, 0.267, 0.339, 0.226, 0.307, 0.295]
gyro_datas_y = [-0.037, -0.041, -0.021, -0.168, 0.082, 0.094, 0.013, 0.119, 0.191, 0.134, 0.168, 0.168, 0.181, 0.181, 0.155, 0.16, 0.198, 0.354, 0.299, 0.443, 0.32, 0.229, -0.077, -0.103, -0.343, -0.419, -0.604, -0.685, -0.799, -0.767, -0.677, -0.654, -0.7, -0.82, -0.724, -0.132, -0.682, -0.729, -0.696, -0.49, -0.611, -0.141, -0.386, -0.132, -0.386, -0.263, -0.342, -0.332, -0.338, -0.341, -0.32, -0.358, -0.398, -0.422, -0.492, -0.425, -0.38, -0.438, -0.389, -0.358, -0.372, -0.333, -0.335, -0.365, -0.416, -0.42, -0.423, -0.468, -0.491, -0.46, -0.482, -0.466, -0.485, -0.382, -0.429, -0.337, -0.226, -0.61, -0.321, -0.329, -0.372, -0.472, -0.401, -0.34, -0.423, -0.451, -0.442, -0.446, -0.387, -0.362, -0.33, -0.382, -0.414, -0.404, -0.396, -0.396, -0.387, -0.407, -0.4, -0.377, -0.381, -0.377, -0.378, -0.381, -0.393, -0.372, -0.381, -0.377, -0.386, -0.397, -0.393, -0.403, -0.386, -0.386, -0.396, -0.429, -0.405, -0.44, -0.398, -0.395, -0.447, -0.502, -0.502, -0.485, -0.463, -0.456, -0.445, -0.466, -0.421, -0.409, -0.425, -0.503, -0.478, -0.479, -0.607, -0.479, -0.545, -0.515, -0.625, -0.445, -0.495, -0.382, -0.329, -0.362, -0.359, -0.396, -0.385, -0.411, -0.388, -0.352, -0.365, -0.357, -0.319, -0.332, -0.312, -0.31, -0.29, -0.297, -0.039, -0.064, -0.162, -0.325, -0.291, -0.373, -0.3, -0.139, -0.478, -0.261, -0.475, -0.412, -0.516, -0.517, -0.724, -0.706, -0.525, -0.669, -0.663, -0.384, -0.45, -0.929, -0.387, -0.301, -5.189, -0.387, -1.412, -0.163, -0.353, -0.575, -0.495, -0.298, -0.388, -0.491, -0.437, -0.397, -0.5, -0.48, -0.546, -0.377, -0.421, -0.449, -0.406, -0.521, -0.504, -0.531, -0.338, -0.284, -0.317, -0.273, -0.189, -0.316, -0.219, -0.153, -0.022, -0.165, -0.145, -0.042, 0.033, -0.049, -0.019, -0.007, -0.034, -0.057, -0.254, -0.232, -0.39, -0.266, -0.316, -0.068, -0.146]
gyro_datas_z = [-0.609,-0.585,-0.604,-0.622,-0.567,-0.607,-0.665,-0.596,-0.662,-0.645,-0.665,-0.58,-0.606,-0.647,-0.606,-0.688,-0.675,-0.571,-0.438,-0.42,-0.331,-0.332,-0.257,-0.153,-0.271,-0.525,-0.561,-0.491,-0.375,-0.342,-0.509,-0.503,-0.445,-0.385,-0.281,-0.413,-0.937,-0.759,-0.703,-0.636,-0.66,-0.433,-0.907,-0.557,-0.663,-0.632,-0.566,-0.481,-0.457,-0.498,-0.476,-0.481,-0.451,-0.424,-0.482,-0.47,-0.51,-0.545,-0.492,-0.49,-0.448,-0.443,-0.425,-0.464,-0.562,-0.514,-0.479,-0.513,-0.559,-0.541,-0.556,-0.553,-0.585,-0.535,-0.578,-0.553,-0.454,-1.005,-0.917,-0.906,-0.892,-1.016,-0.896,-0.826,-0.962,-0.908,-0.893,-0.987,-1.023,-0.888,-0.875,-0.914,-0.998,-0.909,-0.895,-0.957,-0.948,-0.999,-0.913,-0.889,-0.898,-0.929,-0.923,-0.908,-0.919,-0.891,-0.928,-0.949,-0.927,-0.918,-0.897,-0.916,-0.91,-0.885,-0.917,-0.958,-0.936,-0.883,-0.844,-0.874,-0.885,-0.952,-0.917,-0.887,-0.935,-0.879,-0.86,-0.913,-0.921,-0.92,-0.829,-0.857,-0.75,-0.742,-0.895,-0.779,-0.937,-0.807,-0.847,-0.743,-0.838,-0.808,-0.762,-0.755,-0.781,-0.815,-0.787,-0.871,-0.856,-0.826,-0.872,-0.839,-0.782,-0.781,-0.745,-0.775,-0.758,-0.838,-0.135,-0.272,-0.296,-0.435,-0.39,-0.478,-0.63,-0.583,-0.605,-0.833,-0.787,-0.75,-1.035,-1.058,-0.988,-0.891,-0.895,-1.134,-0.953,-0.635,-0.725,-1.155,-0.602,-1.457,7.93,1.188,-1.413,-0.945,-0.962,-0.753,-0.701,-0.892,-0.803,-0.795,-0.901,-0.839,-0.926,-0.919,-1.081,-0.89,-0.883,-0.906,-0.885,-0.961,-1.133,-1.193,-0.998,-0.84,-0.955,-0.993,-1.011,-1.072,-1.112,-1.021,-1.031,-0.983,-0.884,-1.005,-1.124,-1.123,-0.994,-0.851,-0.941,-1.175,-1.043,-0.967,-0.983,-0.94,-1.009,-0.969,-0.982]
	
# ※これらのリストが同じ長さで、対応する時間軸があることを想定しています。
# ※もし時間軸データがない場合は、サンプリングレートからDTを決定する必要があります。

if gyro_datas_x and gyro_datas_y and gyro_datas_z: # データが存在するか確認
    # 落下データの最初の値をシミュレーションの初期角速度として使用 (dpsからrad/sへ変換)
    initial_omega_x_rad_s = np.deg2rad(gyro_datas_x[0])
    initial_omega_y_rad_s = np.deg2rad(gyro_datas_y[0])
    initial_omega_z_rad_s = np.deg2rad(gyro_datas_z[0])
    initial_omega_body_rad_s = np.array([initial_omega_x_rad_s, initial_omega_y_rad_s, initial_omega_z_rad_s])
    print(f"Using initial angular velocity from gyro data (rad/s): {initial_omega_body_rad_s}")
else:
    # データがない場合のフォールバック (以前のダミーデータ)
    print("Gyro data not found or empty. Using default initial angular velocity.")
    initial_omega_body_rad_s = np.deg2rad(np.array([10.0, -20.0, 5.0])) # 例: [rad/s]

initial_orientation_quat = R.from_euler('xyz', [0,0,0], degrees=True) # 初期姿勢 (クォータニオン)

# --- 2. 状態変数 ---
time_hist = np.arange(0, TOTAL_TIME, DT)
num_steps = len(time_hist)

# CanSatの状態履歴
omega_body_hist_rad_s = np.zeros((num_steps, 3))
orientation_hist_quat = np.zeros((num_steps, 4)) # scipy.Rotation.as_quat() は [x,y,z,w] 順
euler_hist_deg = np.zeros((num_steps, 3)) # 確認用

# CMGの状態履歴
gimbal_angles_rad_hist = np.zeros((num_steps, NUM_CMGS))
gimbal_rates_rad_s_hist = np.zeros((num_steps, NUM_CMGS))
cmg_torque_hist_body = np.zeros((num_steps, 3)) # CMGが発生した合計トルク(物体座標系)

# 初期値設定
current_omega_body_rad_s = initial_omega_body_rad_s.copy()
current_orientation = initial_orientation_quat # Scipy Rotation object
current_gimbal_angles_rad = np.zeros(NUM_CMGS) # 初期ジンバル角0度

omega_body_hist_rad_s[0, :] = current_omega_body_rad_s
orientation_hist_quat[0, :] = current_orientation.as_quat()
euler_hist_deg[0, :] = current_orientation.as_euler('xyz', degrees=True)
gimbal_angles_rad_hist[0, :] = current_gimbal_angles_rad

# --- 3. シミュレーションループ ---
for i in range(1, num_steps):
    t = time_hist[i]

    # --- 3a. 制御ロジック ---
    # 目標角速度はゼロ
    error_omega_body = current_omega_body_rad_s # (current_omega - 0)
    target_torque_body = -KP_OMEGA * error_omega_body

    # 現在の各CMGのホイール角運動量ベクトル (物体座標系)
    h_wheels_body = []
    for k_cmg in range(NUM_CMGS):
        # ジンバル回転を考慮した現在のホイールスピン軸
        # R_gimbal_k = R.from_rotvec(CMG_CONFIG[k_cmg]['gimbal_axis'] * current_gimbal_angles_rad[k_cmg])
        # スピン軸はジンバル軸に直交しているので、ロドリゲスの回転公式などで正確に計算
        u_g = CMG_CONFIG[k_cmg]['gimbal_axis']
        theta_g = current_gimbal_angles_rad[k_cmg]
        h0_spin = CMG_CONFIG[k_cmg]['initial_spin_axis'] * H_WHEEL_SCALAR
        # 簡単な回転 (ジンバル軸が座標軸と一致している場合)
        if np.allclose(u_g, [1,0,0]): R_g_mat = R.from_euler('x', theta_g).as_matrix()
        elif np.allclose(u_g, [0,1,0]): R_g_mat = R.from_euler('y', theta_g).as_matrix()
        elif np.allclose(u_g, [0,0,1]): R_g_mat = R.from_euler('z', theta_g).as_matrix()
        else: # 一般の軸 (ロドリゲスの公式)
            K = np.array([[0, -u_g[2], u_g[1]], [u_g[2], 0, -u_g[0]], [-u_g[1], u_g[0], 0]])
            R_g_mat = np.eye(3) + np.sin(theta_g) * K + (1 - np.cos(theta_g)) * (K @ K)
        
        current_h_k_body = R_g_mat @ h0_spin
        h_wheels_body.append(current_h_k_body)

    # トルクヤコビアン A の計算: A_ij = (gimbal_axis_i x h_wheel_i)_j
    A_jacobian = np.zeros((3, NUM_CMGS))
    for k_cmg in range(NUM_CMGS):
        torque_per_unit_gimbal_rate = np.cross(CMG_CONFIG[k_cmg]['gimbal_axis'], h_wheels_body[k_cmg])
        A_jacobian[:, k_cmg] = torque_per_unit_gimbal_rate

    # ジンバルレート指令の計算 (擬似逆行列を使用)
    try:
        # A_pinv = np.linalg.pinv(A_jacobian) # 通常の擬似逆行列
        # 特異点対策 (SR-Inverse: Singularity Robust Inverse)
        lambda_sr = 0.01 # SR-Inverseの正則化パラメータ (小さいほど通常のpinvに近い)
        A_pinv = A_jacobian.T @ np.linalg.inv(A_jacobian @ A_jacobian.T + lambda_sr * np.eye(3))

        commanded_gimbal_rates_rad_s = A_pinv @ target_torque_body
    except np.linalg.LinAlgError: # 特異点などで逆行列が計算できない場合
        print(f"Warning: Jacobian pseudo-inverse failed at t={t:.2f}s. Setting gimbal rates to zero.")
        commanded_gimbal_rates_rad_s = np.zeros(NUM_CMGS)

    # ジンバルレートの制限
    actual_gimbal_rates_rad_s = np.clip(commanded_gimbal_rates_rad_s, -GIMBAL_RATE_MAX_RAD_S, GIMBAL_RATE_MAX_RAD_S)

    # --- 3b. CMGが発生する実際の合計トルク ---
    actual_total_cmg_torque_body = A_jacobian @ actual_gimbal_rates_rad_s # 線形性を仮定

    # --- 3c. CanSatの運動方程式 (オイラー方程式) ---
    omega_dot_body = I_CANSAT_BODY_INV @ (actual_total_cmg_torque_body - np.cross(current_omega_body_rad_s, I_CANSAT_BODY @ current_omega_body_rad_s))

    # --- 3d. 数値積分 (オイラー法またはより高次の方法) ---
    # 角速度の更新
    current_omega_body_rad_s += omega_dot_body * DT

    # 姿勢の更新 (クォータニオン)
    # Scipy Rotation オブジェクトで回転を表現し、角速度ベクトルでインクリメンタルに回転させる
    rotation_increment = R.from_rotvec(current_omega_body_rad_s * DT)
    current_orientation = rotation_increment * current_orientation # 回転を合成
    # current_orientation.normalize() # ScipyのRotationは通常内部で正規化されるが、念のため確認

    # ジンバル角度の更新
    current_gimbal_angles_rad += actual_gimbal_rates_rad_s * DT
    # ジンバル角度の範囲制限 (必要なら -pi から pi など)
    # current_gimbal_angles_rad = (current_gimbal_angles_rad + np.pi) % (2 * np.pi) - np.pi


    # --- 3e. データ記録 ---
    omega_body_hist_rad_s[i, :] = current_omega_body_rad_s
    orientation_hist_quat[i, :] = current_orientation.as_quat() # [x,y,z,w]
    euler_hist_deg[i, :] = current_orientation.as_euler('xyz', degrees=True)
    gimbal_angles_rad_hist[i, :] = current_gimbal_angles_rad
    gimbal_rates_rad_s_hist[i, :] = actual_gimbal_rates_rad_s
    cmg_torque_hist_body[i, :] = actual_total_cmg_torque_body

# --- 4. 結果のプロット ---
fig_plots, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# 角速度
axs[0].plot(time_hist, np.rad2deg(omega_body_hist_rad_s[:,0]), label='$\omega_x$ (body)')
axs[0].plot(time_hist, np.rad2deg(omega_body_hist_rad_s[:,1]), label='$\omega_y$ (body)')
axs[0].plot(time_hist, np.rad2deg(omega_body_hist_rad_s[:,2]), label='$\omega_z$ (body)')
axs[0].set_ylabel('Ang. Vel. (deg/s)')
axs[0].legend()
axs[0].grid(True)
axs[0].set_title('CanSat Angular Velocity with CMG Control')

# オイラー角 (確認用)
axs[1].plot(time_hist, euler_hist_deg[:,0], label='Roll (X)')
axs[1].plot(time_hist, euler_hist_deg[:,1], label='Pitch (Y)')
axs[1].plot(time_hist, euler_hist_deg[:,2], label='Yaw (Z)')
axs[1].set_ylabel('Euler Angles (deg)')
axs[1].legend()
axs[1].grid(True)

# ジンバル角度
for k_cmg in range(NUM_CMGS):
    axs[2].plot(time_hist, np.rad2deg(gimbal_angles_rad_hist[:, k_cmg]), label=f'Gimbal {k_cmg+1} Angle')
axs[2].set_ylabel('Gimbal Angles (deg)')
axs[2].legend()
axs[2].grid(True)

# CMG発生トルク
axs[3].plot(time_hist, cmg_torque_hist_body[:,0], label='$T_{CMG,x}$')
axs[3].plot(time_hist, cmg_torque_hist_body[:,1], label='$T_{CMG,y}$')
axs[3].plot(time_hist, cmg_torque_hist_body[:,2], label='$T_{CMG,z}$')
axs[3].set_ylabel('CMG Torque (Nm)')
axs[3].set_xlabel('Time (s)')
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.savefig("cansat_cmg_simulation_plots.png")
plt.show()

# --- 5. アニメーション (既存のコードを参考に作成) ---
# ここでは省略しますが、orientation_hist_quat を使って物体の姿勢をアニメーション化します。
# 物体の頂点座標 (既存のものを流用)
vertices_body_frame = np.array([
    [-CAN_LENGTH/2, -CAN_WIDTH/2, -CAN_HEIGHT/2], [ CAN_LENGTH/2, -CAN_WIDTH/2, -CAN_HEIGHT/2],
    [ CAN_LENGTH/2,  CAN_WIDTH/2, -CAN_HEIGHT/2], [-CAN_LENGTH/2,  CAN_WIDTH/2, -CAN_HEIGHT/2],
    [-CAN_LENGTH/2, -CAN_WIDTH/2,  CAN_HEIGHT/2], [ CAN_LENGTH/2, -CAN_WIDTH/2,  CAN_HEIGHT/2],
    [ CAN_LENGTH/2,  CAN_WIDTH/2,  CAN_HEIGHT/2], [-CAN_LENGTH/2,  CAN_WIDTH/2,  CAN_HEIGHT/2]
])

fig_anim = plt.figure(figsize=(8,8))
ax_anim = fig_anim.add_subplot(111, projection='3d')

def update_animation(frame):
    ax_anim.cla()
    ax_anim.set_xlim([-0.1, 0.1]); ax_anim.set_ylim([-0.1, 0.1]); ax_anim.set_zlim([-0.1, 0.1]) # スケール調整
    ax_anim.set_xlabel("X"); ax_anim.set_ylabel("Y"); ax_anim.set_zlabel("Z")
    ax_anim.set_title(f"CanSat Attitude, Time: {time_hist[frame]:.2f} s")

    # Scipyのクォータニオンは [x,y,z,w] なので、Rotationオブジェクトに渡す際はそのまま
    q_current_scipy = orientation_hist_quat[frame, :]
    R_body_to_inertial = R.from_quat(q_current_scipy).as_matrix()

    rotated_vertices = (R_body_to_inertial @ vertices_body_frame.T).T

    faces = [
        [rotated_vertices[0], rotated_vertices[1], rotated_vertices[2], rotated_vertices[3]],
        [rotated_vertices[4], rotated_vertices[5], rotated_vertices[6], rotated_vertices[7]],
        [rotated_vertices[0], rotated_vertices[1], rotated_vertices[5], rotated_vertices[4]],
        [rotated_vertices[2], rotated_vertices[3], rotated_vertices[7], rotated_vertices[6]],
        [rotated_vertices[1], rotated_vertices[2], rotated_vertices[6], rotated_vertices[5]],
        [rotated_vertices[0], rotated_vertices[3], rotated_vertices[7], rotated_vertices[4]]
    ]
    poly3d = Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='k', alpha=0.7)
    ax_anim.add_collection3d(poly3d)
    ax_anim.view_init(elev=20, azim=30 + frame*0.5) # カメラアングル
    return []

ani = FuncAnimation(fig_anim, update_animation, frames=num_steps, blit=False, interval=DT*1000)
# ani.save("cansat_cmg_animation.mp4", writer=FFMpegWriter(fps=int(1/DT))) # 保存する場合
plt.show()