import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter # 既存
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # 既存
from scipy.spatial.transform import Rotation as R

# --- 0. シミュレーション・物理パラメータ ---
DT = 0.01  # シミュレーション時間ステップ (s)

# CanSat パラメータ (例)
MASS_CANSAT = 0.0641250  # kg
# 寸法 (m) (既存の length, width, height を使う)
CAN_LENGTH = 0.018 # X軸方向の長さと仮定 (慣性テンソル計算用)
CAN_WIDTH = 0.020   # Y軸方向
CAN_HEIGHT = 0.095  # Z軸方向


"""
テンソアを計算
"""
def get_inertia_tensor_cuboid(mass, l, w, h):
    I_xx = (1/12) * mass * (w**2 + h**2)
    I_yy = (1/12) * mass * (l**2 + h**2)
    I_zz = (1/12) * mass * (l**2 + w**2)
    return np.diag([I_xx, I_yy, I_zz])

I_CANSAT_BODY = get_inertia_tensor_cuboid(MASS_CANSAT, CAN_LENGTH, CAN_WIDTH, CAN_HEIGHT)
I_CANSAT_BODY_INV = np.linalg.inv(I_CANSAT_BODY)

# CMG パラメータ (例 - 3基共通とする)
I_WHEEL = 1.357e-5  # ホイールの慣性モーメント (kg*m^2)　タングステン合金を想定
OMEGA_WHEEL_NOMINAL_RPM = 1000  # ホイール定格回転数 (rpm)
OMEGA_WHEEL_NOMINAL_RAD_S = OMEGA_WHEEL_NOMINAL_RPM * (2 * np.pi) / 60
H_WHEEL_SCALAR = I_WHEEL * OMEGA_WHEEL_NOMINAL_RAD_S # ホイールの角運動量の大きさ
GIMBAL_RATE_MAX_DPS = 90  # ジンバル最大角速度 (deg/s)
GIMBAL_RATE_MAX_RAD_S = np.deg2rad(GIMBAL_RATE_MAX_DPS)

# 制御ゲイン
KP_OMEGA = 0.01 # 角速度制御の比例ゲイン でかくすると振動する

# CMGの想定は1つ2つのジンバルで制御


# CMG配置: [ジンバル軸(物体座標系), ジンバル角0度でのホイールスピン軸(物体座標系)]
OUTER_GIMBAL_AXIS_BODY = np.array([0., 0., 1.]) # 外側ジンバルの回転軸
INNER_GIMBAL_AXIS_BODY_RELATIVE = np.array([0., 1., 0.]) # 内側ジンバルの回転軸 (外側ジンバルに対する相対軸)
INITIAL_SPIN_AXIS_BODY_RELATIVE = np.array([1., 0., 0.]) # ホイールのスピン軸 (内側ジンバルに対する相対軸, 角度0時)
NUM_GIMBALS = 2 # ジンバルの数

# 観測された角速度データ [dps]
gyro_datas_x = [0.841, 0.833, 0.852, 0.848, 0.791, 0.781, 0.891, 0.802, 0.884, 0.843, 0.875, 0.734, 0.769, 0.829, 0.786, 0.919, 0.889, 0.807, 0.834, 0.927, 0.886, 0.963, 1.007, 0.892, 0.855, 0.765, 0.479, 0.424, 0.633, 0.674, 0.577, 0.496, 0.572, 0.648, 0.587, 0.371, 0.448, 0.487, 0.599, 0.622, 0.655, 0.455, 0.992, 0.644, 0.99, 1.021, 0.877, 0.821, 0.805, 0.865, 0.802, 0.796, 0.813, 0.716, 0.799, 0.802, 0.855, 0.913, 0.834, 0.839, 0.806, 0.819, 0.783, 0.834, 0.877, 0.854, 0.726, 0.746, 0.785, 0.771, 0.786, 0.787, 0.817, 0.701, 0.725, 0.645, 0.441, 0.57, 0.262, 0.184, 0.29, 0.399, 0.321, 0.263, 0.251, 0.306, 0.274, 0.303, 0.342, 0.304, 0.296, 0.316, 0.348, 0.335, 0.336, 0.372, 0.37, 0.372, 0.362, 0.342, 0.337, 0.345, 0.337, 0.318, 0.329, 0.303, 0.309, 0.318, 0.316, 0.322, 0.315, 0.325, 0.321, 0.309, 0.313, 0.332, 0.346, 0.31, 0.297, 0.278, 0.294, 0.317, 0.293, 0.277, 0.274, 0.271, 0.257, 0.278, 0.292, 0.303, 0.257, 0.289, 0.347, 0.375, 0.379, 0.315, 0.404, 0.465, 0.519, 0.494, 0.544, 0.526, 0.514, 0.52, 0.501, 0.531, 0.519, 0.599, 0.597, 0.576, 0.601, 0.574, 0.548, 0.543, 0.53, 0.566, 0.567, 0.637, 0.087, 0.137, 0.341, 0.457, 0.425, 0.532, 0.565, 0.505, 0.836, 0.717, 0.572, 0.349, 0.421, 0.542, 0.745, 0.76, 0.656, 0.586, 0.515, 0.382, 0.149, -0.022, 0.442, 0.516, -0.021, -0.553, -1, -0.268, -0.287, -0.219, -0.228, -0.375, -0.255, -0.25, -0.136, -0.204, -0.08, 0.059, -0.083, -0.041, -0.005, -0.016, -0.079, -0.056, 0.084, 0.038, 0.122, 0.098, 0.053, 0.005, -0.008, 0.148, 0.204, 0.268, 0.26, 0.234, 0.216, 0.249, 0.285, 0.152, 0.209, 0.37, 0.091, 0.419, 0.217, 0.334, 0.267, 0.339, 0.226, 0.307, 0.295]
gyro_datas_y = [-0.037, -0.041, -0.021, -0.168, 0.082, 0.094, 0.013, 0.119, 0.191, 0.134, 0.168, 0.168, 0.181, 0.181, 0.155, 0.16, 0.198, 0.354, 0.299, 0.443, 0.32, 0.229, -0.077, -0.103, -0.343, -0.419, -0.604, -0.685, -0.799, -0.767, -0.677, -0.654, -0.7, -0.82, -0.724, -0.132, -0.682, -0.729, -0.696, -0.49, -0.611, -0.141, -0.386, -0.132, -0.386, -0.263, -0.342, -0.332, -0.338, -0.341, -0.32, -0.358, -0.398, -0.422, -0.492, -0.425, -0.38, -0.438, -0.389, -0.358, -0.372, -0.333, -0.335, -0.365, -0.416, -0.42, -0.423, -0.468, -0.491, -0.46, -0.482, -0.466, -0.485, -0.382, -0.429, -0.337, -0.226, -0.61, -0.321, -0.329, -0.372, -0.472, -0.401, -0.34, -0.423, -0.451, -0.442, -0.446, -0.387, -0.362, -0.33, -0.382, -0.414, -0.404, -0.396, -0.396, -0.387, -0.407, -0.4, -0.377, -0.381, -0.377, -0.378, -0.381, -0.393, -0.372, -0.381, -0.377, -0.386, -0.397, -0.393, -0.403, -0.386, -0.386, -0.396, -0.429, -0.405, -0.44, -0.398, -0.395, -0.447, -0.502, -0.502, -0.485, -0.463, -0.456, -0.445, -0.466, -0.421, -0.409, -0.425, -0.503, -0.478, -0.479, -0.607, -0.479, -0.545, -0.515, -0.625, -0.445, -0.495, -0.382, -0.329, -0.362, -0.359, -0.396, -0.385, -0.411, -0.388, -0.352, -0.365, -0.357, -0.319, -0.332, -0.312, -0.31, -0.29, -0.297, -0.039, -0.064, -0.162, -0.325, -0.291, -0.373, -0.3, -0.139, -0.478, -0.261, -0.475, -0.412, -0.516, -0.517, -0.724, -0.706, -0.525, -0.669, -0.663, -0.384, -0.45, -0.929, -0.387, -0.301, -5.189, -0.387, -1.412, -0.163, -0.353, -0.575, -0.495, -0.298, -0.388, -0.491, -0.437, -0.397, -0.5, -0.48, -0.546, -0.377, -0.421, -0.449, -0.406, -0.521, -0.504, -0.531, -0.338, -0.284, -0.317, -0.273, -0.189, -0.316, -0.219, -0.153, -0.022, -0.165, -0.145, -0.042, 0.033, -0.049, -0.019, -0.007, -0.034, -0.057, -0.254, -0.232, -0.39, -0.266, -0.316, -0.068, -0.146]
gyro_datas_z = [-0.609,-0.585,-0.604,-0.622,-0.567,-0.607,-0.665,-0.596,-0.662,-0.645,-0.665,-0.58,-0.606,-0.647,-0.606,-0.688,-0.675,-0.571,-0.438,-0.42,-0.331,-0.332,-0.257,-0.153,-0.271,-0.525,-0.561,-0.491,-0.375,-0.342,-0.509,-0.503,-0.445,-0.385,-0.281,-0.413,-0.937,-0.759,-0.703,-0.636,-0.66,-0.433,-0.907,-0.557,-0.663,-0.632,-0.566,-0.481,-0.457,-0.498,-0.476,-0.481,-0.451,-0.424,-0.482,-0.47,-0.51,-0.545,-0.492,-0.49,-0.448,-0.443,-0.425,-0.464,-0.562,-0.514,-0.479,-0.513,-0.559,-0.541,-0.556,-0.553,-0.585,-0.535,-0.578,-0.553,-0.454,-1.005,-0.917,-0.906,-0.892,-1.016,-0.896,-0.826,-0.962,-0.908,-0.893,-0.987,-1.023,-0.888,-0.875,-0.914,-0.998,-0.909,-0.895,-0.957,-0.948,-0.999,-0.913,-0.889,-0.898,-0.929,-0.923,-0.908,-0.919,-0.891,-0.928,-0.949,-0.927,-0.918,-0.897,-0.916,-0.91,-0.885,-0.917,-0.958,-0.936,-0.883,-0.844,-0.874,-0.885,-0.952,-0.917,-0.887,-0.935,-0.879,-0.86,-0.913,-0.921,-0.92,-0.829,-0.857,-0.75,-0.742,-0.895,-0.779,-0.937,-0.807,-0.847,-0.743,-0.838,-0.808,-0.762,-0.755,-0.781,-0.815,-0.787,-0.871,-0.856,-0.826,-0.872,-0.839,-0.782,-0.781,-0.745,-0.775,-0.758,-0.838,-0.135,-0.272,-0.296,-0.435,-0.39,-0.478,-0.63,-0.583,-0.605,-0.833,-0.787,-0.75,-1.035,-1.058,-0.988,-0.891,-0.895,-1.134,-0.953,-0.635,-0.725,-1.155,-0.602,-1.457,7.93,1.188,-1.413,-0.945,-0.962,-0.753,-0.701,-0.892,-0.803,-0.795,-0.901,-0.839,-0.926,-0.919,-1.081,-0.89,-0.883,-0.906,-0.885,-0.961,-1.133,-1.193,-0.998,-0.84,-0.955,-0.993,-1.011,-1.072,-1.112,-1.021,-1.031,-0.983,-0.884,-1.005,-1.124,-1.123,-0.994,-0.851,-0.941,-1.175,-1.043,-0.967,-0.983,-0.94,-1.009,-0.969,-0.982]


if not (gyro_datas_x and gyro_datas_y and gyro_datas_z and \
        len(gyro_datas_x) == len(gyro_datas_y) == len(gyro_datas_z) and \
        len(gyro_datas_x) > 0):
    print("Error: Gyro data is missing, empty, or inconsistent in length.")
    print("Please provide valid gyro_datas_x, gyro_datas_y, gyro_datas_z lists.")
    exit() # エラー終了

# シミュレーションのステップ数を角速度データの長さに合わせる
num_steps = len(gyro_datas_x)
print(f"Number of simulation steps will be: {num_steps} (based on gyro data length)")

# シミュレーションの時間軸を生成
time_hist = np.arange(0, num_steps * DT, DT)
# arangeの仕様上、ステップ数が一つずれる可能性があるので、長さをnum_stepsに合わせる
if len(time_hist) > num_steps:
    time_hist = time_hist[:num_steps]
elif len(time_hist) < num_steps: # DTが大きすぎたりした場合の稀なケース
    time_hist = np.linspace(0, (num_steps -1) * DT, num_steps)


# 初期角速度をデータから取得
initial_omega_x_rad_s = np.deg2rad(gyro_datas_x[0])
initial_omega_y_rad_s = np.deg2rad(gyro_datas_y[0])
initial_omega_z_rad_s = np.deg2rad(gyro_datas_z[0])
initial_omega_body_rad_s = np.array([initial_omega_x_rad_s, initial_omega_y_rad_s, initial_omega_z_rad_s])
print(f"Using initial angular velocity from gyro data (rad/s): {initial_omega_body_rad_s}")

initial_orientation = R.from_euler('xyz', [0,0,0], degrees=True) # 初期姿勢

# --- 2. 状態変数 ---

# CanSatの状態履歴
omega_body_hist_rad_s = np.zeros((num_steps, 3))
orientation_hist_quat = np.zeros((num_steps, 4)) # scipy.Rotation.as_quat() は [x,y,z,w] 順
euler_hist_deg = np.zeros((num_steps, 3)) # 確認用

# CMGの状態履歴
# --- ★ 単一2ジンバルCMGの状態履歴に変更 ---
gimbal_angles_rad_hist = np.zeros((num_steps, NUM_GIMBALS)) # (num_steps, 2)
gimbal_rates_rad_s_hist = np.zeros((num_steps, NUM_GIMBALS)) # (num_steps, 2)
cmg_torque_hist_body = np.zeros((num_steps, 3))

# 初期値設定
current_omega_body_rad_s = initial_omega_body_rad_s.copy()
current_orientation = initial_orientation # Scipy Rotation object
current_gimbal_angles_rad = np.zeros(NUM_GIMBALS) # [outer_angle, inner_angle]

omega_body_hist_rad_s[0, :] = current_omega_body_rad_s
orientation_hist_quat[0, :] = current_orientation.as_quat()
euler_hist_deg[0, :] = current_orientation.as_euler('xyz', degrees=True)
gimbal_angles_rad_hist[0, :] = current_gimbal_angles_rad

# --- 3. シミュレーションループ ---
"""
CMGの挙動をシミュレーションする
"""

for i in range(1, num_steps):
    t = time_hist[i]

    # --- 3a. 制御ロジック ---
    # 目標角速度はゼロ

    error_omega_body = current_omega_body_rad_s
    target_torque_body = -KP_OMEGA * error_omega_body

    # --- ★ 現在のホイール角運動量ベクトルの計算 (2ジンバル) ---
    outer_angle = current_gimbal_angles_rad[0]
    inner_angle = current_gimbal_angles_rad[1]

    # 外側ジンバル回転
    R_outer_gimbal = R.from_rotvec(OUTER_GIMBAL_AXIS_BODY * outer_angle).as_matrix()
    # 外側ジンバル回転後の内側ジンバル軸 (ボディ座標系)
    current_inner_gimbal_axis_body = R_outer_gimbal @ INNER_GIMBAL_AXIS_BODY_RELATIVE
    # 内側ジンバル回転 (現在の内側軸周り)
    R_inner_gimbal = R.from_rotvec(current_inner_gimbal_axis_body * inner_angle).as_matrix()

    # ホイールの初期スピン軸ベクトル (大きさ H を含む)
    h_wheel_initial_vector_body = INITIAL_SPIN_AXIS_BODY_RELATIVE * H_WHEEL_SCALAR
    # 現在のホイール角運動量ベクトル (ボディ座標系)
    current_h_wheel_body = R_inner_gimbal @ R_outer_gimbal @ h_wheel_initial_vector_body
    # --- ★ ---

    # --- ★ トルクヤコビアン A (3x2行列) の計算 ---
    A_jacobian = np.zeros((3, NUM_GIMBALS))
    # 外側ジンバルレートによるトルク (∂τ/∂α_o_dot)
    A_jacobian[:, 0] = np.cross(OUTER_GIMBAL_AXIS_BODY, current_h_wheel_body)
    # 内側ジンバルレートによるトルク (∂τ/∂α_i_dot)
    A_jacobian[:, 1] = np.cross(current_inner_gimbal_axis_body, current_h_wheel_body)
    # --- ★ ---

    # --- ★ ジンバルレート指令の計算 (2ジンバル用) ---
    try:
        lambda_sr = 0.01 # または別の特異点回避策
        # (A^T A + λI)⁻¹ A^T を使う方法もある (より安定)
        A_pinv = np.linalg.pinv(A_jacobian, rcond=lambda_sr) # rcondで調整も可能
        # または SR-Inverse
        # A_pinv = A_jacobian.T @ np.linalg.inv(A_jacobian @ A_jacobian.T + lambda_sr * np.eye(3))

        commanded_gimbal_rates_rad_s = A_pinv @ target_torque_body # (2,) 配列
    except np.linalg.LinAlgError:
        print(f"Warning: Jacobian pseudo-inverse failed at t={t:.2f}s. Setting gimbal rates to zero.")
        commanded_gimbal_rates_rad_s = np.zeros(NUM_GIMBALS)

    # ジンバルレートの制限
    actual_gimbal_rates_rad_s = np.clip(commanded_gimbal_rates_rad_s, -GIMBAL_RATE_MAX_RAD_S, GIMBAL_RATE_MAX_RAD_S)
    # --- ★ ---

    # --- 3b. CMGが発生する実際の合計トルク ---
    actual_total_cmg_torque_body = A_jacobian @ actual_gimbal_rates_rad_s # (3,) 配列

    # --- 3c. CanSatの運動方程式 (変更なし) ---
    omega_dot_body = I_CANSAT_BODY_INV @ (actual_total_cmg_torque_body - np.cross(current_omega_body_rad_s, I_CANSAT_BODY @ current_omega_body_rad_s))

    # --- 3d. 数値積分 (更新は同じだが、ジンバル角度の対象が変わる) ---
    current_omega_body_rad_s += omega_dot_body * DT
    rotation_increment = R.from_rotvec(current_omega_body_rad_s * DT)
    current_orientation = rotation_increment * current_orientation
    # --- ★ 2つのジンバル角度を更新 ---
    current_gimbal_angles_rad += actual_gimbal_rates_rad_s * DT
    # --- ★ ---

    # --- 3e. データ記録 (記録配列の次元が変わっていることに注意) ---
    omega_body_hist_rad_s[i, :] = current_omega_body_rad_s
    orientation_hist_quat[i, :] = current_orientation.as_quat()
    euler_hist_deg[i, :] = current_orientation.as_euler('xyz', degrees=True)
    gimbal_angles_rad_hist[i, :] = current_gimbal_angles_rad # (2,)
    gimbal_rates_rad_s_hist[i, :] = actual_gimbal_rates_rad_s # (2,)
    cmg_torque_hist_body[i, :] = actual_total_cmg_torque_body # (3,)

# --- 4. 結果のプロット ---
fig_plots, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# --- 4. 結果のプロット (ジンバル関連のループを変更) ---
fig_plots, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
# 角速度プロット (変更なし)
axs[0].plot(time_hist, np.rad2deg(omega_body_hist_rad_s[:,0]), label='$\omega_x$'); axs[0].plot(time_hist, np.rad2deg(omega_body_hist_rad_s[:,1]), label='$\omega_y$'); axs[0].plot(time_hist, np.rad2deg(omega_body_hist_rad_s[:,2]), label='$\omega_z$')
axs[0].set_ylabel('Ang. Vel. (deg/s)'); axs[0].legend(); axs[0].grid(True); axs[0].set_title('CanSat with Single 2-Gimbal CMG')
# オイラー角プロット (変更なし)
axs[1].plot(time_hist, euler_hist_deg[:,0], label='Roll'); axs[1].plot(time_hist, euler_hist_deg[:,1], label='Pitch'); axs[1].plot(time_hist, euler_hist_deg[:,2], label='Yaw')
axs[1].set_ylabel('Euler Angles (deg)'); axs[1].legend(); axs[1].grid(True)
# --- ★ ジンバル角度プロット (2つに) ---
axs[2].plot(time_hist, np.rad2deg(gimbal_angles_rad_hist[:, 0]), label=f'Outer Gimbal Angle')
axs[2].plot(time_hist, np.rad2deg(gimbal_angles_rad_hist[:, 1]), label=f'Inner Gimbal Angle')
axs[2].set_ylabel('Gimbal Angles (deg)'); axs[2].legend(); axs[2].grid(True)
# --- ★ ---
# CMG発生トルクプロット (変更なし)
axs[3].plot(time_hist, cmg_torque_hist_body[:,0], label='$T_{x}$'); axs[3].plot(time_hist, cmg_torque_hist_body[:,1], label='$T_{y}$'); axs[3].plot(time_hist, cmg_torque_hist_body[:,2], label='$T_{z}$')
axs[3].set_ylabel('CMG Torque (Nm)'); axs[3].set_xlabel('Time (s)'); axs[3].legend(); axs[3].grid(True)
plt.tight_layout(); plt.savefig("cansat_single_dgcmg_plots.png"); plt.show()


# --- 5. アニメーション ---
# --- 5. アニメーション (単一2ジンバルCMG用) ---

# CanSat本体の頂点定義
vertices_body_frame = np.array([
    [-CAN_LENGTH/2, -CAN_WIDTH/2, -CAN_HEIGHT/2], [ CAN_LENGTH/2, -CAN_WIDTH/2, -CAN_HEIGHT/2],
    [ CAN_LENGTH/2,  CAN_WIDTH/2, -CAN_HEIGHT/2], [-CAN_LENGTH/2,  CAN_WIDTH/2, -CAN_HEIGHT/2],
    [-CAN_LENGTH/2, -CAN_WIDTH/2,  CAN_HEIGHT/2], [ CAN_LENGTH/2, -CAN_WIDTH/2,  CAN_HEIGHT/2],
    [ CAN_LENGTH/2,  CAN_WIDTH/2,  CAN_HEIGHT/2], [-CAN_LENGTH/2,  CAN_WIDTH/2,  CAN_HEIGHT/2]
])

# CMGホイール描画用のパラメータ (確認のため再掲)
WHEEL_RADIUS = 0.015  # ホイールの半径 (m)
WHEEL_THICKNESS = 0.003 # ホイールの厚み (m)
SPOKE_COLOR = 'red'
WHEEL_EDGE_COLOR = 'darkgray'
WHEEL_FACE_COLOR = 'lightgray'
GIMBAL_AXIS_COLOR = 'blue'
INNER_GIMBAL_AXIS_COLOR = 'green' # 内側ジンバル軸の色を変える場合

# --- ヘルパー関数 ---
def create_cylinder_faces(radius, height, n_segments=20):
    """円柱の側面、上面、底面のポリゴン頂点リスト(3D座標のリスト)を生成"""
    theta = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z_top = height / 2
    z_bottom = -height / 2

    # 底面の頂点
    bottom_verts = np.vstack((x, y, np.full(n_segments, z_bottom))).T
    # 上面の頂点
    top_verts = np.vstack((x, y, np.full(n_segments, z_top))).T

    faces = []
    # 側面
    for i in range(n_segments):
        i_next = (i + 1) % n_segments
        faces.append([bottom_verts[i], bottom_verts[i_next], top_verts[i_next], top_verts[i]])
    # 底面 (反時計回りになるように頂点順を調整)
    faces.append(bottom_verts[::-1])
    # 上面
    faces.append(top_verts)
    return faces # 面ごとの頂点リストのリスト

def create_disk_points(radius, n_points=50):
    """指定された半径の円周上の点を生成 (XY平面上、Z=0、 Nx3 配列)"""
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros(n_points)
    return np.vstack((x, y, z)).T

def transform_points(points, rotation_matrix, translation_vector):
    """点群を回転・平行移動 (points: Nx3 配列)"""
    rotated_points = (rotation_matrix @ points.T).T
    return rotated_points + translation_vector

# --- アニメーション設定 ---
fig_anim_dgcmg = plt.figure(figsize=(10, 10))
ax_anim_dgcmg = fig_anim_dgcmg.add_subplot(111, projection='3d')
ax_anim_dgcmg.view_init(elev=25, azim=45) # 固定のカメラ視点

# 基準となるCMG部品の形状データを生成
base_wheel_faces = create_cylinder_faces(WHEEL_RADIUS, WHEEL_THICKNESS)
base_spoke = np.array([[0, 0, 0], [WHEEL_RADIUS * 0.9, 0, 0]]) # XY平面上の線分

# アニメーション更新関数
def update_animation_dgcmg(frame):
    ax_anim_dgcmg.cla()
    # 軸範囲とラベルを毎回設定
    ax_anim_dgcmg.set_xlim([-0.06, 0.06]); ax_anim_dgcmg.set_ylim([-0.06, 0.06]); ax_anim_dgcmg.set_zlim([-0.08, 0.08])
    ax_anim_dgcmg.set_xlabel("X_inertial"); ax_anim_dgcmg.set_ylabel("Y_inertial"); ax_anim_dgcmg.set_zlabel("Z_inertial")
    ax_anim_dgcmg.set_title(f"CanSat Attitude & 2-Gimbal CMG, Time: {time_hist[frame]:.2f} s")

    # --- 1. CanSat本体の描画 (半透明スタイル) ---
    q_cansat_current = orientation_hist_quat[frame, :]
    R_cansat_to_inertial_mat = R.from_quat(q_cansat_current).as_matrix()
    rotated_cansat_vertices = (R_cansat_to_inertial_mat @ vertices_body_frame.T).T
    cansat_faces_def = [
        [rotated_cansat_vertices[0], rotated_cansat_vertices[1], rotated_cansat_vertices[2], rotated_cansat_vertices[3]],
        [rotated_cansat_vertices[4], rotated_cansat_vertices[5], rotated_cansat_vertices[6], rotated_cansat_vertices[7]],
        [rotated_cansat_vertices[0], rotated_cansat_vertices[1], rotated_cansat_vertices[5], rotated_cansat_vertices[4]],
        [rotated_cansat_vertices[2], rotated_cansat_vertices[3], rotated_cansat_vertices[7], rotated_cansat_vertices[6]],
        [rotated_cansat_vertices[1], rotated_cansat_vertices[2], rotated_cansat_vertices[6], rotated_cansat_vertices[5]],
        [rotated_cansat_vertices[0], rotated_cansat_vertices[3], rotated_cansat_vertices[7], rotated_cansat_vertices[4]]
    ]
    cansat_poly3d = Poly3DCollection(cansat_faces_def, facecolors='cyan', linewidths=0.5, edgecolors='gray', alpha=0.15)
    ax_anim_dgcmg.add_collection3d(cansat_poly3d)

    # --- 2. 単一2ジンバルCMGの描画 ---
    outer_gimbal_angle = gimbal_angles_rad_hist[frame, 0]
    inner_gimbal_angle = gimbal_angles_rad_hist[frame, 1]

    # a. ジンバル回転行列の計算
    R_outer = R.from_rotvec(OUTER_GIMBAL_AXIS_BODY * outer_gimbal_angle).as_matrix()
    current_inner_axis_body = R_outer @ INNER_GIMBAL_AXIS_BODY_RELATIVE
    R_inner = R.from_rotvec(current_inner_axis_body * inner_gimbal_angle).as_matrix()

    # b. ホイールの最終的な向きを決定する回転 (ボディ座標系内)
    #    初期スピン軸をまず内側ジンバルで回転し、次に外側ジンバルで回転
    R_wheel_orientation_in_body = R_outer @ R_inner

    # c. CMGの搭載位置 (ボディ座標系 - 例: 原点)
    cmg_origin_body = np.array([0., 0., 0.])
    # 慣性系でのCMG原点
    cmg_origin_inertial = R_cansat_to_inertial_mat @ cmg_origin_body # (今回は原点なのでゼロ)

    # d. ホイールの描画
    #    d1. ホイール基準形状 (スピン軸=初期スピン軸) を定義
    #        INITIAL_SPIN_AXIS_BODY_RELATIVE をZ軸に向ける回転を求める
    vec_from_wheel_local = np.array([0.,0.,1.])
    vec_to_wheel_local = INITIAL_SPIN_AXIS_BODY_RELATIVE / np.linalg.norm(INITIAL_SPIN_AXIS_BODY_RELATIVE)
    R_align_wheel_to_initial_spin = np.eye(3)
    if not np.allclose(vec_from_wheel_local, vec_to_wheel_local):
        if np.allclose(vec_from_wheel_local, -vec_to_wheel_local):
            ortho_axis = np.cross(vec_from_wheel_local, np.array([1.,0.,0.]))
            if np.linalg.norm(ortho_axis) < 1e-6: ortho_axis = np.cross(vec_from_wheel_local, np.array([0.,1.,0.]))
            if np.linalg.norm(ortho_axis) > 1e-6: R_align_wheel_to_initial_spin = R.from_rotvec(np.pi * ortho_axis / np.linalg.norm(ortho_axis)).as_matrix()
            else: R_align_wheel_to_initial_spin = np.diag([-1.,-1.,1.]) # X軸を初期スピン軸とするならY軸周り180度
        else:
            rot_obj, _ = R.align_vectors(vec_to_wheel_local[np.newaxis,:], vec_from_wheel_local[np.newaxis,:])
            R_align_wheel_to_initial_spin = rot_obj.as_matrix()

    #    d2. ホイールの最終的な回転行列 (慣性系)
    R_wheel_total_inertial = R_cansat_to_inertial_mat @ R_wheel_orientation_in_body @ R_align_wheel_to_initial_spin

    #    d3. ホイールの面を描画
    wheel_faces_inertial = []
    for face_verts_base in base_wheel_faces:
        face_verts_inertial = transform_points(np.array(face_verts_base), R_wheel_total_inertial, cmg_origin_inertial)
        wheel_faces_inertial.append(face_verts_inertial)

    # Poly3DCollectionで側面、底面、上面を描画
    wheel_poly3d_sides = Poly3DCollection(wheel_faces_inertial[:-2], facecolors=WHEEL_FACE_COLOR, edgecolors=WHEEL_EDGE_COLOR, alpha=0.8, linewidths=0.5)
    wheel_poly3d_bottom = Poly3DCollection([wheel_faces_inertial[-2]], facecolors=WHEEL_FACE_COLOR, edgecolors=WHEEL_EDGE_COLOR, alpha=0.8, linewidths=0.5)
    wheel_poly3d_top = Poly3DCollection([wheel_faces_inertial[-1]], facecolors=WHEEL_FACE_COLOR, edgecolors=WHEEL_EDGE_COLOR, alpha=0.8, linewidths=0.5)
    ax_anim_dgcmg.add_collection3d(wheel_poly3d_sides)
    ax_anim_dgcmg.add_collection3d(wheel_poly3d_bottom)
    ax_anim_dgcmg.add_collection3d(wheel_poly3d_top)


    # e. ホイールのスピンを示す「スポーク」
    visual_spin_rate_factor = 0.03 # スピンの見た目の速さを調整
    wheel_spin_angle_visual = (frame * DT * OMEGA_WHEEL_NOMINAL_RAD_S * visual_spin_rate_factor) % (2 * np.pi)
    # スポークはホイールの初期スピン軸周りに回転させると仮定
    R_spoke_visual = R.from_rotvec(INITIAL_SPIN_AXIS_BODY_RELATIVE * wheel_spin_angle_visual).as_matrix()
    # スポークの基準形状を回転させ、ホイール全体の回転と位置オフセット、CanSat回転を適用
    spoke_rotated_base = R_spoke_visual @ base_spoke.T
    spoke_inertial = transform_points(spoke_rotated_base.T, R_wheel_total_inertial, cmg_origin_inertial)

    ax_anim_dgcmg.plot(spoke_inertial[:,0], spoke_inertial[:,1], spoke_inertial[:,2], color=SPOKE_COLOR, linewidth=1.5)

    # f. (オプション) ジンバル軸を描画
    axis_viz_length = WHEEL_RADIUS * 1.5
    # 外側ジンバル軸 (慣性系)
    outer_gimbal_axis_inertial = R_cansat_to_inertial_mat @ OUTER_GIMBAL_AXIS_BODY
    og_start = cmg_origin_inertial - outer_gimbal_axis_inertial * axis_viz_length / 2
    og_end   = cmg_origin_inertial + outer_gimbal_axis_inertial * axis_viz_length / 2
    ax_anim_dgcmg.plot([og_start[0], og_end[0]], [og_start[1], og_end[1]], [og_start[2], og_end[2]],
                     color=GIMBAL_AXIS_COLOR, linestyle='--', linewidth=1, alpha=0.7, label='Outer Gimbal Axis')

    # 内側ジンバル軸 (慣性系) - 外側ジンバルの回転も考慮
    inner_gimbal_axis_inertial = R_cansat_to_inertial_mat @ current_inner_axis_body
    ig_start = cmg_origin_inertial - inner_gimbal_axis_inertial * axis_viz_length / 2
    ig_end   = cmg_origin_inertial + inner_gimbal_axis_inertial * axis_viz_length / 2
    ax_anim_dgcmg.plot([ig_start[0], ig_end[0]], [ig_start[1], ig_end[1]], [ig_start[2], ig_end[2]],
                     color=INNER_GIMBAL_AXIS_COLOR, linestyle=':', linewidth=1, alpha=0.7, label='Inner Gimbal Axis')

    # 一度だけ凡例を表示
    if frame == 0:
       ax_anim_dgcmg.legend(fontsize='small')

    return []

# --- アニメーションの実行と表示/保存 ---
# (num_steps と DT はシミュレーションループから引き継ぐ)
print("Creating animation...")
ani_dgcmg = FuncAnimation(fig_anim_dgcmg, update_animation_dgcmg, frames=num_steps,
                        blit=False, interval=max(1, int(DT*1000))) # intervalはミリ秒、最低1ms

# アニメーションをMP4として保存する場合 (FFMpegWriterが必要)
try:
    writer = FFMpegWriter(fps=max(1, int(1/DT)), metadata=dict(artist='Me'), bitrate=1800)
    #ani_dgcmg.save("cansat_dgcmg_animation.mp4", writer=writer)
    print("Animation saved to cansat_dgcmg_animation.mp4")
except Exception as e:
    print(f"Could not save animation: {e}")
    print("Showing animation instead...")
    plt.show()

# 保存しない場合は単に表示
plt.show()

print("Animation part finished.")
