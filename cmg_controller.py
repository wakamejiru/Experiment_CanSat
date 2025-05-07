import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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
I_WHEEL = 1.375e-5  # ホイールの慣性モーメント (kg*m^2)　タングステン合金を想定
OMEGA_WHEEL_NOMINAL_RPM = 4500  # ホイール定格回転数 (rpm)
OMEGA_WHEEL_NOMINAL_RAD_S = OMEGA_WHEEL_NOMINAL_RPM * (2 * np.pi) / 60
H_WHEEL_SCALAR = I_WHEEL * OMEGA_WHEEL_NOMINAL_RAD_S # ホイールの角運動量の大きさ
GIMBAL_RATE_MAX_DPS = 90  # ジンバル最大角速度 (deg/s)
GIMBAL_RATE_MAX_RAD_S =  np.deg2rad(GIMBAL_RATE_MAX_DPS)

# 制御ゲイン
KP_OMEGA = 0.15 # 角速度制御の比例ゲイン でかくすると振動する


# CMG配置: [ジンバル軸(物体座標系), ジンバル角0度でのホイールスピン軸(物体座標系)]
CMG_CONFIG = [
    {'gimbal_axis': np.array([1., 0., 0.]), 'initial_spin_axis': np.array([0., 1., 0.])}, # CMG1: Xジンバル, Yスピン
    {'gimbal_axis': np.array([0., 1., 0.]), 'initial_spin_axis': np.array([0., 0., 1.])}, # CMG2: Yジンバル, Zスピン
    {'gimbal_axis': np.array([0., 0., 1.]), 'initial_spin_axis': np.array([1., 0., 0.])}, # CMG3: Zジンバル, Xスピン
]

NUM_CMGS = len(CMG_CONFIG)

# 観測された角速度データ [dps]
gyro_datas_x = [-158.264,-107.208,119.385,-11.047,175.171,533.691,552.979,107.666,395.233,-1.526,-244.202,-118.744,269.318,102.57,-147.797,-8.82,123.535,137.207,178.68,89.63,-78.918,-9.399,183.594,88.867,61.829,103.21,38.086]
gyro_datas_y = [37.476,55.786,77.942,10.803,-132.782,-284.698,-421.326,-533.234,-644.348,-657.104,-482.544,-223.022,-366.302,-572.662,-451.447,-321.991,-296.844,-349.884,-458.679,-426.941,-173.218,-140.808,-656.982,-567.444,-5.402,162.262,129.944]
gyro_datas_z = [98.541,127.93,72.052,-59.692,-174.652,-217.102,-220.306,-346.832,-318.787,-189.789,-399.17,-669.708,-730.042,-565.43,-557.22,-543.61,-441.01,-327.209,-213.013,-232.239,-466.156,-575.134,-947.266,-59.326,-153.839,-49.774,-127.686]
	

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

initial_orientation_quat = R.from_euler('xyz', [0,0,0], degrees=True) # 初期姿勢 (クォータニオン)

# --- 2. 状態変数 ---

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
gimbal_labels = ['X', 'Y', 'Z'] # ジンバル軸のラベルを定義
for k_cmg in range(NUM_CMGS):
    axs[2].plot(time_hist, np.rad2deg(gimbal_angles_rad_hist[:, k_cmg]), label=f'Gimbal {gimbal_labels[k_cmg]} Angle')
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


# --- 5. アニメーション ---

# CanSat本体の頂点定義
vertices_body_frame = np.array([
    [-CAN_LENGTH/2, -CAN_WIDTH/2, -CAN_HEIGHT/2], [ CAN_LENGTH/2, -CAN_WIDTH/2, -CAN_HEIGHT/2],
    [ CAN_LENGTH/2,  CAN_WIDTH/2, -CAN_HEIGHT/2], [-CAN_LENGTH/2,  CAN_WIDTH/2, -CAN_HEIGHT/2],
    [-CAN_LENGTH/2, -CAN_WIDTH/2,  CAN_HEIGHT/2], [ CAN_LENGTH/2, -CAN_WIDTH/2,  CAN_HEIGHT/2],
    [ CAN_LENGTH/2,  CAN_WIDTH/2,  CAN_HEIGHT/2], [-CAN_LENGTH/2,  CAN_WIDTH/2,  CAN_HEIGHT/2]
])

# CMGホイール描画用のパラメータ
WHEEL_RADIUS = 0.015  # ホイールの半径 (m) - CanSatのサイズに合わせて調整
WHEEL_THICKNESS = 0.003 # ホイールの厚み (m) - 視覚的なもの
SPOKE_COLOR = 'red'
WHEEL_EDGE_COLOR = 'darkgray'
WHEEL_FACE_COLOR = 'lightgray'

def create_cylinder_mesh(radius, height, n_segments=20):
    """円柱のメッシュデータを生成 (中心が原点、軸がZ軸方向)"""
    # 底面と上面の円周上の点
    theta = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # 底面の頂点 (z = -height/2)
    bottom_verts = np.vstack((x, y, -height/2 * np.ones(n_segments))).T
    # 上面の頂点 (z = height/2)
    top_verts = np.vstack((x, y, height/2 * np.ones(n_segments))).T

    all_verts = np.vstack((bottom_verts, top_verts))

    faces = []
    # 側面
    for i in range(n_segments):
        i_next = (i + 1) % n_segments
        faces.append([bottom_verts[i], bottom_verts[i_next], top_verts[i_next], top_verts[i]])
    # 底面 (ポリゴンとして描画)
    # faces.append(Poly3DCollection([bottom_verts], facecolors=WHEEL_FACE_COLOR, edgecolors=WHEEL_EDGE_COLOR, alpha=0.8))
    #上面
    # faces.append(Poly3DCollection([top_verts], facecolors=WHEEL_FACE_COLOR, edgecolors=WHEEL_EDGE_COLOR, alpha=0.8))
    return all_verts, faces # ここでは側面の面定義のみ返す (Poly3DCollectionで使うため)

def create_disk_points(radius, n_points=50):
    """指定された半径の円周上の点を生成 (XY平面上、Z=0)"""
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros(n_points)
    return np.vstack((x, y, z)).T # Nx3 配列


fig_anim_cmg = plt.figure(figsize=(10,10)) # ウィンドウサイズ調整
ax_anim_cmg = fig_anim_cmg.add_subplot(111, projection='3d')
ax_anim_cmg.view_init(elev=25, azim=45) # 固定のカメラ視点

# アニメーション更新関数
def update_animation_with_cmg(frame):
    ax_anim_cmg.cla()
    ax_anim_cmg.set_xlim([-0.05, 0.05]); ax_anim_cmg.set_ylim([-0.05, 0.05]); ax_anim_cmg.set_zlim([-0.07, 0.07]) # 描画範囲調整
    ax_anim_cmg.set_xlabel("X_inertial"); ax_anim_cmg.set_ylabel("Y_inertial"); ax_anim_cmg.set_zlabel("Z_inertial")
    ax_anim_cmg.set_title(f"CanSat Attitude & CMG, Time: {time_hist[frame]:.2f} s")

    # --- 1. CanSat本体の描画 ---
    q_cansat_current = orientation_hist_quat[frame, :]
    R_cansat_to_inertial_mat = R.from_quat(q_cansat_current).as_matrix()
    rotated_cansat_vertices = (R_cansat_to_inertial_mat @ vertices_body_frame.T).T
    cansat_faces_def = [ # 直方体の面定義
        [rotated_cansat_vertices[0], rotated_cansat_vertices[1], rotated_cansat_vertices[2], rotated_cansat_vertices[3]],
        [rotated_cansat_vertices[4], rotated_cansat_vertices[5], rotated_cansat_vertices[6], rotated_cansat_vertices[7]],
        [rotated_cansat_vertices[0], rotated_cansat_vertices[1], rotated_cansat_vertices[5], rotated_cansat_vertices[4]],
        [rotated_cansat_vertices[2], rotated_cansat_vertices[3], rotated_cansat_vertices[7], rotated_cansat_vertices[6]],
        [rotated_cansat_vertices[1], rotated_cansat_vertices[2], rotated_cansat_vertices[6], rotated_cansat_vertices[5]],
        [rotated_cansat_vertices[0], rotated_cansat_vertices[3], rotated_cansat_vertices[7], rotated_cansat_vertices[4]]
    ]
    # 缶サットの描画

    cansat_poly3d = Poly3DCollection(
    cansat_faces_def,
    facecolors='cyan',  # 面の色
    linewidths=0.5,     # 輪郭線を細くするのも効果的です
    edgecolors='gray',  # 輪郭線の色
    alpha=0.15          # 透過度を調整
    )    
    #ax_anim_cmg.add_collection3d(cansat_poly3d)

    # --- 2. 各CMGの描画 ---
    for k_cmg in range(NUM_CMGS):
        # a. CMGの基本設定 (CanSatボディ座標系)
        gimbal_axis_body = CMG_CONFIG[k_cmg]['gimbal_axis']
        initial_spin_axis_body = CMG_CONFIG[k_cmg]['initial_spin_axis']
        
        # CMGの取り付け位置 (CanSatボディ座標系) - 例: CanSatのZ軸に沿って少しずらす
        # (これは例なので、実際の配置に合わせて調整してください)
        offset_scale = 0.03 # CanSatの中心から少しずらす距離
        if k_cmg == 0: cmg_position_body = np.array([0, 0, offset_scale])  # 上の方
        elif k_cmg == 1: cmg_position_body = np.array([0, 0, 0])           # 中央
        else: cmg_position_body = np.array([0, 0, -offset_scale]) # 下の方
        # もしX,Y,Z軸にそれぞれ配置するなら:
        # if gimbal_axis_body[0] == 1: cmg_position_body = np.array([offset_scale * 0.7, 0, 0])
        # elif gimbal_axis_body[1] == 1: cmg_position_body = np.array([0, offset_scale * 0.7, 0])
        # else: cmg_position_body = np.array([0, 0, offset_scale * 0.7])


        # b. 現在のジンバル角度とジンバル回転行列 (ボディ座標系内でジンバル回転)
        current_gimbal_angle_rad = gimbal_angles_rad_hist[frame, k_cmg]
        R_gimbal_in_body = R.from_rotvec(gimbal_axis_body * current_gimbal_angle_rad).as_matrix()

        # c. ジンバル回転後のホイールスピン軸 (CanSatボディ座標系)
        current_spin_axis_body = R_gimbal_in_body @ initial_spin_axis_body

        # d. ホイールの変換行列を計算 (基準姿勢から現在の姿勢へ)
        #    基準姿勢: ホイールの中心が原点、スピン軸がZ軸=[0,0,1]
        #    目標姿勢: 中心が cmg_position_body、スピン軸が current_spin_axis_body (これら全てCanSatボディ座標系内)
        
        #    d1. Z軸を current_spin_axis_body に合わせる回転 (ボディ座標系内)
        vec_from = np.array([0., 0., 1.])
        vec_to = current_spin_axis_body / np.linalg.norm(current_spin_axis_body)
        
        R_spin_align_in_body = np.eye(3)
        if not np.allclose(vec_from, vec_to):
            if np.allclose(vec_from, -vec_to): # 180度回転
                # vec_fromに垂直な軸を一つ見つける (例: X軸とクロス積)
                ortho_axis = np.cross(vec_from, np.array([1.,0.,0.]))
                if np.linalg.norm(ortho_axis) < 1e-6: # vec_from がX軸に平行だった場合
                    ortho_axis = np.cross(vec_from, np.array([0.,1.,0.]))
                if np.linalg.norm(ortho_axis) > 1e-6:
                   R_spin_align_in_body = R.from_rotvec(np.pi * ortho_axis / np.linalg.norm(ortho_axis)).as_matrix()
                else: # 稀なケース、vec_from が Z軸で ortho_axis もゼロになることはないはず
                   R_spin_align_in_body = np.diag([1.,-1.,-1.]) # 180度回転の例
            else:
                rot_obj, _ = R.align_vectors(vec_to[np.newaxis,:], vec_from[np.newaxis,:])
                R_spin_align_in_body = rot_obj.as_matrix()
        
        # e. ホイールの描画
        #    e1. ホイールのメッシュ (円柱形状)
        base_wheel_verts, base_wheel_faces_def = create_cylinder_mesh(WHEEL_RADIUS, WHEEL_THICKNESS)
        
        #    e2. ボディ座標系でのホイール頂点
        #        まずスピン軸方向に回転させ、次にCMGの取り付け位置に平行移動
        wheel_verts_body = (R_spin_align_in_body @ base_wheel_verts.T).T + cmg_position_body
        
        #    e3. 慣性系でのホイール頂点
        wheel_verts_inertial = (R_cansat_to_inertial_mat @ wheel_verts_body.T).T

        #    e4. Poly3DCollectionでホイールの面を描画 (側面)
        # Poly3DCollectionは頂点リストのリストを受け取るので、facesの定義を調整
        wheel_faces_for_plot = []
        for face_indices_in_base_wheel_verts in base_wheel_faces_def: # これは現在頂点そのもの
            # wheel_faces_for_plot.append([wheel_verts_inertial[idx] for idx in face_indices_in_base_wheel_verts])
            # create_cylinder_mesh が直接頂点を返す場合
            face_verts_body = (R_spin_align_in_body @ np.array(face_indices_in_base_wheel_verts).T).T + cmg_position_body
            face_verts_inertial = (R_cansat_to_inertial_mat @ face_verts_body.T).T
            wheel_faces_for_plot.append(face_verts_inertial)


        wheel_poly3d = Poly3DCollection(wheel_faces_for_plot, facecolors=WHEEL_FACE_COLOR, edgecolors=WHEEL_EDGE_COLOR, alpha=0.9, linewidths=0.5)
        ax_anim_cmg.add_collection3d(wheel_poly3d)

        #    e5. ホイールの上面と底面を円として描画 (線を引く)
        disk_base_pts = create_disk_points(WHEEL_RADIUS) # (N,3)
        
        top_disk_center_body = cmg_position_body + R_spin_align_in_body @ np.array([0,0,WHEEL_THICKNESS/2])
        bottom_disk_center_body = cmg_position_body + R_spin_align_in_body @ np.array([0,0,-WHEEL_THICKNESS/2])

        top_disk_pts_body = (R_spin_align_in_body @ disk_base_pts.T).T + top_disk_center_body
        bottom_disk_pts_body = (R_spin_align_in_body @ disk_base_pts.T).T + bottom_disk_center_body
        
        top_disk_pts_inertial = (R_cansat_to_inertial_mat @ top_disk_pts_body.T).T
        bottom_disk_pts_inertial = (R_cansat_to_inertial_mat @ bottom_disk_pts_body.T).T

        #ax_anim_cmg.plot(top_disk_pts_inertial[:,0], top_disk_pts_inertial[:,1], top_disk_pts_inertial[:,2], color=WHEEL_EDGE_COLOR, linewidth=1)
        #ax_anim_cmg.plot(bottom_disk_pts_inertial[:,0], bottom_disk_pts_inertial[:,1], bottom_disk_pts_inertial[:,2], color=WHEEL_EDGE_COLOR, linewidth=1)


        #    f. ホイールのスピンを示す「スポーク」
        #       スピン角 (視覚効果用、実際の角速度とは異なる遅い速度)
        visual_spin_rate_factor = 0.02 # 遅く見せるための係数
        wheel_spin_angle_visual = (frame * DT * OMEGA_WHEEL_NOMINAL_RAD_S * visual_spin_rate_factor) % (2 * np.pi)

        #       スポークの端点 (ホイールのローカルZ軸周りに回転するXY平面上の線)
        spoke_start_local = np.array([0, 0, 0]) # ホイール中心
        spoke_end_local = np.array([WHEEL_RADIUS * 0.9, 0, 0]) # ホイール半径の90%
        
        R_spoke_visual_spin = R.from_euler('z', wheel_spin_angle_visual).as_matrix() # ローカルZ軸周り
        spoke_start_spun_local = R_spoke_visual_spin @ spoke_start_local
        spoke_end_spun_local = R_spoke_visual_spin @ spoke_end_local
        
        #       ボディ座標系へ -> 慣性系へ
        spoke_start_body = R_spin_align_in_body @ spoke_start_spun_local + cmg_position_body
        spoke_end_body   = R_spin_align_in_body @ spoke_end_spun_local + cmg_position_body
        
        spoke_start_inertial = R_cansat_to_inertial_mat @ spoke_start_body
        spoke_end_inertial   = R_cansat_to_inertial_mat @ spoke_end_body
        
        ax_anim_cmg.plot([spoke_start_inertial[0], spoke_end_inertial[0]],
                         [spoke_start_inertial[1], spoke_end_inertial[1]],
                         [spoke_start_inertial[2], spoke_end_inertial[2]],
                         color=SPOKE_COLOR, linewidth=1.5)

        # (オプション) ジンバル軸を描画
        gimbal_axis_inertial = R_cansat_to_inertial_mat @ gimbal_axis_body
        cmg_center_inertial = R_cansat_to_inertial_mat @ cmg_position_body
        axis_viz_length = WHEEL_RADIUS * 1.2
        g_start = cmg_center_inertial - gimbal_axis_inertial * axis_viz_length / 2
        g_end   = cmg_center_inertial + gimbal_axis_inertial * axis_viz_length / 2
        # ax_anim_cmg.plot([g_start[0], g_end[0]], [g_start[1], g_end[1]], [g_start[2], g_end[2]], color='blue', linestyle=':', linewidth=1)


    return []

# アニメーションの実行
# (num_steps と DT はシミュレーション結果から取得できていると仮定)
ani_cmg = FuncAnimation(fig_anim_cmg, update_animation_with_cmg, frames=num_steps, blit=False, interval=DT*1000)
ani_cmg.save("cansat_with_cmg_animation.mp4", writer=FFMpegWriter(fps=int(1/DT))) # 保存する場合
plt.show()