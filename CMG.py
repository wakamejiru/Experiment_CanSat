"落下データより必要なCMGを算出する" 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FFMpegWriter

# 物体の長方形の頂点
length = 0.018 # 奥行 (m)
width = 0.020  # 幅 (m)
height = 0.095  # 高さ (m)
# アニメーションの設定
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 物体の初期の回転角度
theta_x = 0  # X軸回転角度
theta_y = 0  # Y軸回転角度
theta_z = 0  # Z軸回転角度
# 長方体の8つの頂点座標
vertices = np.array([[-length/2, -width/2, -height/2],
					[ length/2, -width/2, -height/2],
					[ length/2,  width/2, -height/2],
					[-length/2,  width/2, -height/2],
					[-length/2, -width/2,  height/2],
					[ length/2, -width/2,  height/2],
					[ length/2,  width/2,  height/2],
					[-length/2,  width/2,  height/2]])

def main():
	gyro_datas_x = [0.841, 0.833, 0.852, 0.848, 0.791, 0.781, 0.891, 0.802, 0.884, 0.843, 0.875, 0.734, 0.769, 0.829, 0.786, 0.919, 0.889, 0.807, 0.834, 0.927, 0.886, 0.963, 1.007, 0.892, 0.855, 0.765, 0.479, 0.424, 0.633, 0.674, 0.577, 0.496, 0.572, 0.648, 0.587, 0.371, 0.448, 0.487, 0.599, 0.622, 0.655, 0.455, 0.992, 0.644, 0.99, 1.021, 0.877, 0.821, 0.805, 0.865, 0.802, 0.796, 0.813, 0.716, 0.799, 0.802, 0.855, 0.913, 0.834, 0.839, 0.806, 0.819, 0.783, 0.834, 0.877, 0.854, 0.726, 0.746, 0.785, 0.771, 0.786, 0.787, 0.817, 0.701, 0.725, 0.645, 0.441, 0.57, 0.262, 0.184, 0.29, 0.399, 0.321, 0.263, 0.251, 0.306, 0.274, 0.303, 0.342, 0.304, 0.296, 0.316, 0.348, 0.335, 0.336, 0.372, 0.37, 0.372, 0.362, 0.342, 0.337, 0.345, 0.337, 0.318, 0.329, 0.303, 0.309, 0.318, 0.316, 0.322, 0.315, 0.325, 0.321, 0.309, 0.313, 0.332, 0.346, 0.31, 0.297, 0.278, 0.294, 0.317, 0.293, 0.277, 0.274, 0.271, 0.257, 0.278, 0.292, 0.303, 0.257, 0.289, 0.347, 0.375, 0.379, 0.315, 0.404, 0.465, 0.519, 0.494, 0.544, 0.526, 0.514, 0.52, 0.501, 0.531, 0.519, 0.599, 0.597, 0.576, 0.601, 0.574, 0.548, 0.543, 0.53, 0.566, 0.567, 0.637, 0.087, 0.137, 0.341, 0.457, 0.425, 0.532, 0.565, 0.505, 0.836, 0.717, 0.572, 0.349, 0.421, 0.542, 0.745, 0.76, 0.656, 0.586, 0.515, 0.382, 0.149, -0.022, 0.442, 0.516, -0.021, -0.553, -1, -0.268, -0.287, -0.219, -0.228, -0.375, -0.255, -0.25, -0.136, -0.204, -0.08, 0.059, -0.083, -0.041, -0.005, -0.016, -0.079, -0.056, 0.084, 0.038, 0.122, 0.098, 0.053, 0.005, -0.008, 0.148, 0.204, 0.268, 0.26, 0.234, 0.216, 0.249, 0.285, 0.152, 0.209, 0.37, 0.091, 0.419, 0.217, 0.334, 0.267, 0.339, 0.226, 0.307, 0.295]
	gyro_datas_y = [-0.037, -0.041, -0.021, -0.168, 0.082, 0.094, 0.013, 0.119, 0.191, 0.134, 0.168, 0.168, 0.181, 0.181, 0.155, 0.16, 0.198, 0.354, 0.299, 0.443, 0.32, 0.229, -0.077, -0.103, -0.343, -0.419, -0.604, -0.685, -0.799, -0.767, -0.677, -0.654, -0.7, -0.82, -0.724, -0.132, -0.682, -0.729, -0.696, -0.49, -0.611, -0.141, -0.386, -0.132, -0.386, -0.263, -0.342, -0.332, -0.338, -0.341, -0.32, -0.358, -0.398, -0.422, -0.492, -0.425, -0.38, -0.438, -0.389, -0.358, -0.372, -0.333, -0.335, -0.365, -0.416, -0.42, -0.423, -0.468, -0.491, -0.46, -0.482, -0.466, -0.485, -0.382, -0.429, -0.337, -0.226, -0.61, -0.321, -0.329, -0.372, -0.472, -0.401, -0.34, -0.423, -0.451, -0.442, -0.446, -0.387, -0.362, -0.33, -0.382, -0.414, -0.404, -0.396, -0.396, -0.387, -0.407, -0.4, -0.377, -0.381, -0.377, -0.378, -0.381, -0.393, -0.372, -0.381, -0.377, -0.386, -0.397, -0.393, -0.403, -0.386, -0.386, -0.396, -0.429, -0.405, -0.44, -0.398, -0.395, -0.447, -0.502, -0.502, -0.485, -0.463, -0.456, -0.445, -0.466, -0.421, -0.409, -0.425, -0.503, -0.478, -0.479, -0.607, -0.479, -0.545, -0.515, -0.625, -0.445, -0.495, -0.382, -0.329, -0.362, -0.359, -0.396, -0.385, -0.411, -0.388, -0.352, -0.365, -0.357, -0.319, -0.332, -0.312, -0.31, -0.29, -0.297, -0.039, -0.064, -0.162, -0.325, -0.291, -0.373, -0.3, -0.139, -0.478, -0.261, -0.475, -0.412, -0.516, -0.517, -0.724, -0.706, -0.525, -0.669, -0.663, -0.384, -0.45, -0.929, -0.387, -0.301, -5.189, -0.387, -1.412, -0.163, -0.353, -0.575, -0.495, -0.298, -0.388, -0.491, -0.437, -0.397, -0.5, -0.48, -0.546, -0.377, -0.421, -0.449, -0.406, -0.521, -0.504, -0.531, -0.338, -0.284, -0.317, -0.273, -0.189, -0.316, -0.219, -0.153, -0.022, -0.165, -0.145, -0.042, 0.033, -0.049, -0.019, -0.007, -0.034, -0.057, -0.254, -0.232, -0.39, -0.266, -0.316, -0.068, -0.146]
	gyro_datas_z = [-0.609,-0.585,-0.604,-0.622,-0.567,-0.607,-0.665,-0.596,-0.662,-0.645,-0.665,-0.58,-0.606,-0.647,-0.606,-0.688,-0.675,-0.571,-0.438,-0.42,-0.331,-0.332,-0.257,-0.153,-0.271,-0.525,-0.561,-0.491,-0.375,-0.342,-0.509,-0.503,-0.445,-0.385,-0.281,-0.413,-0.937,-0.759,-0.703,-0.636,-0.66,-0.433,-0.907,-0.557,-0.663,-0.632,-0.566,-0.481,-0.457,-0.498,-0.476,-0.481,-0.451,-0.424,-0.482,-0.47,-0.51,-0.545,-0.492,-0.49,-0.448,-0.443,-0.425,-0.464,-0.562,-0.514,-0.479,-0.513,-0.559,-0.541,-0.556,-0.553,-0.585,-0.535,-0.578,-0.553,-0.454,-1.005,-0.917,-0.906,-0.892,-1.016,-0.896,-0.826,-0.962,-0.908,-0.893,-0.987,-1.023,-0.888,-0.875,-0.914,-0.998,-0.909,-0.895,-0.957,-0.948,-0.999,-0.913,-0.889,-0.898,-0.929,-0.923,-0.908,-0.919,-0.891,-0.928,-0.949,-0.927,-0.918,-0.897,-0.916,-0.91,-0.885,-0.917,-0.958,-0.936,-0.883,-0.844,-0.874,-0.885,-0.952,-0.917,-0.887,-0.935,-0.879,-0.86,-0.913,-0.921,-0.92,-0.829,-0.857,-0.75,-0.742,-0.895,-0.779,-0.937,-0.807,-0.847,-0.743,-0.838,-0.808,-0.762,-0.755,-0.781,-0.815,-0.787,-0.871,-0.856,-0.826,-0.872,-0.839,-0.782,-0.781,-0.745,-0.775,-0.758,-0.838,-0.135,-0.272,-0.296,-0.435,-0.39,-0.478,-0.63,-0.583,-0.605,-0.833,-0.787,-0.75,-1.035,-1.058,-0.988,-0.891,-0.895,-1.134,-0.953,-0.635,-0.725,-1.155,-0.602,-1.457,7.93,1.188,-1.413,-0.945,-0.962,-0.753,-0.701,-0.892,-0.803,-0.795,-0.901,-0.839,-0.926,-0.919,-1.081,-0.89,-0.883,-0.906,-0.885,-0.961,-1.133,-1.193,-0.998,-0.84,-0.955,-0.993,-1.011,-1.072,-1.112,-1.021,-1.031,-0.983,-0.884,-1.005,-1.124,-1.123,-0.994,-0.851,-0.941,-1.175,-1.043,-0.967,-0.983,-0.94,-1.009,-0.969,-0.982]
	
	data_num = len(gyro_datas_x)
	time_datas = np.arange(0, data_num * 0.1, 0.1)
	CalcCMG(gyro_datas_x, gyro_datas_y, gyro_datas_z, time_datas, data_num)

"CMGの値を計算する"
def CalcCMG(gyro_datas_x, gyro_datas_y, gyro_datas_z, time_datas, data_num):
	# 角速度をラジアン毎秒に変換（deg/s -> rad/s）
    gyro_rad_x = np.deg2rad(gyro_datas_x)
    gyro_rad_y = np.deg2rad(gyro_datas_y)
    gyro_rad_z = np.deg2rad(gyro_datas_z)
	# 角加速度を計算（微分）
    delta_time = 0.1  # 0.1秒の間隔.
	# Xの角速度
    alpha_list_x = np.diff(gyro_rad_x, axis=0) / delta_time  # 微分（角速度の変化量）
    # 配列サイズ合わせ
    alpha_list_x = np.append(alpha_list_x, alpha_list_x[-1])  # 最後の値を追加してサイズを合わせる
	# Yの角速度
    alpha_list_y = np.diff(gyro_rad_y, axis=0) / delta_time  # 微分（角速度の変化量）
    # 配列サイズ合わせ
    alpha_list_y = np.append(alpha_list_y, alpha_list_y[-1])  # 最後の値を追加してサイズを合わせる
	# Zの角速度
    alpha_list_z = np.diff(gyro_rad_z, axis=0) / delta_time  # 微分（角速度の変化量）
    # 配列サイズ合わせ
    alpha_list_z = np.append(alpha_list_z, alpha_list_z[-1])  # 最後の値を追加してサイズを合わせ
    a_x = 0.020
    b_z = 0.045
    c_y = 0.018 # 物体の物理特性(仮定)
    mass = 0.0641250  # 質量（kg）
    """
    height = 0.09  # 高さ（m）# 大きさを直方体として再考慮
	radius = 0.025  # 半径（m）
	# 円柱の慣性モーメント計算
	# X軸回転（横軸）
    I_x = (1 / 12) * mass * (3 * radius**2 + height**2)
	# Y軸回転（縦軸）
    I_y = (1 / 12) * mass * (3 * radius**2 + height**2)
	# Z軸回転（中心軸）
    I_z = (1 / 2) * mass * radius**2
    """
    #円柱から直方体にして計算
	
    I_x = mass / 4 *(a_x**2+b_z**2)
    I_y = mass / 4 *(c_y**2+b_z**2)
    I_z = mass / 4 *(c_y**2+a_x**2)
	# トルクの計算（τ = I * α）
	# 各軸（X, Y, Z）のトルクを計算
    torque_x = I_x * alpha_list_x  # X軸のトルク
    torque_y = I_y * alpha_list_y  # Y軸のトルク
    torque_z = I_z * alpha_list_z  # Z軸のトルク
    # 合成トルクの大きさを計算
    torque_magnitude = np.sqrt(torque_x**2 + torque_y**2 + torque_z**2)
    # 最大トルクを求める
    max_torque = np.max(torque_magnitude)
    print(f"最大トルク: {max_torque:.30f} N·m")

    plt.figure(figsize=(10, 6))
    plt.plot(time_datas, torque_x, label="Torque X")
    plt.plot(time_datas, torque_y, label="Torque Y")
    plt.plot(time_datas, torque_z, label="Torque Z")
    plt.xlabel("Time [s]")
    plt.ylabel("Torque [N·m]")
    plt.title("Calculated Torque over Time")
    plt.legend()
    plt.grid(True)
	# ファイル名の入力
    #filename = input("保存するファイル名（拡張子.pngは不要）: ")
    #plt.savefig(filename + '.png')  # PNG形式で保存
    plt.close()  # プロットを閉じる
    
	# 3Dアニメに変換
    Create3D(gyro_rad_x, gyro_rad_y, gyro_rad_z)
     
# 3Dアニメの初期プロット
def anime_init():
    scale = 0.05  # 全体の表示範囲（m）を大きめに設定
    ax.set_xlim([-length, length])
    ax.set_ylim([-width, width])
    ax.set_zlim([-height, height])
    return []

"3Dのアニメーションを作成"
def Create3D(gyro_rad_x, gyro_rad_y, gyro_rad_z):
	# アニメーションの実行
	ani = FuncAnimation(fig, update, fargs=(gyro_rad_x, gyro_rad_y, gyro_rad_z, vertices), init_func=anime_init, blit=False, interval=100, save_count=len(gyro_rad_x))
	
    # 表示
	#plt.show()# 保存用のwriterを作成（fps=20など自由に調整OK）
	writer = FFMpegWriter(fps=20)

    # 保存
	ani.save("rotation_animation.mp4", writer=writer)
	
# 回転行列の定義
def rotation_matrix(axis, theta):
    """
    回転行列を生成する関数
    axis: 回転軸 ('x', 'y', 'z')
    theta: 回転角度（ラジアン）
    """
    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, np.cos(theta), -np.sin(theta)],
                         [0, np.sin(theta), np.cos(theta)]])
    elif axis == 'y':
        return np.array([[np.cos(theta), 0, np.sin(theta)],
                         [0, 1, 0],
                         [-np.sin(theta), 0, np.cos(theta)]])
    elif axis == 'z':
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])
    return np.eye(3)




# 更新関数
def update(frame, gyro_rad_x, gyro_rad_y, gyro_rad_z, vertices):
    global theta_x, theta_y, theta_z
    delta_time = 0.1  # 時間間隔

    # フレーム数超え防止（動画保存時に多めに呼ばれても大丈夫にする）
    if frame >= len(gyro_rad_x):
        return []

    # 角速度から回転角度の変化を計算
    theta_x += gyro_rad_x[frame] * delta_time
    theta_y += gyro_rad_y[frame] * delta_time
    theta_z += gyro_rad_z[frame] * delta_time

    # X, Y, Z 軸回転の順番で回転行列を適用
    rot_x = rotation_matrix('x', theta_x)
    rot_y = rotation_matrix('y', theta_y)
    rot_z = rotation_matrix('z', theta_z)

    # 物体の頂点を回転
    rotated_vertices = np.dot(vertices, rot_x)
    rotated_vertices = np.dot(rotated_vertices, rot_y)
    rotated_vertices = np.dot(rotated_vertices, rot_z)
    # 回転ベクトル（角速度）を取得（現在のフレーム）
    # 角速度ベクトルから回転リングを描く
    
    # 回転行列を合成して物体の現在の姿勢に変換
    R_total = rot_x @ rot_y @ rot_z

    # 回転ベクトルをローカル座標系に合わせて回す
    omega_world = np.array([gyro_rad_x[frame], gyro_rad_y[frame], gyro_rad_z[frame]])
    omega_local = R_total @ omega_world
    # 回転ベクトル（角速度）を取得（現在のフレーム）
    omega_vector = np.array([
        gyro_rad_x[frame],
        gyro_rad_y[frame],
        gyro_rad_z[frame]
    ])
	# プロットを更新
    ax.cla()  # 現在のプロットをクリア
    #anime_init()
    max_range = np.max(np.abs(rotated_vertices)) * 1.0  # 安全マージンを20%くらい取る

    ax.set_xlim([-length, length])
    ax.set_ylim([-width, width])
    ax.set_zlim([-height, height])
    # 長方形のエッジを描画
    faces = [
        [rotated_vertices[0], rotated_vertices[1], rotated_vertices[2], rotated_vertices[3]],  # 底面
        [rotated_vertices[4], rotated_vertices[5], rotated_vertices[6], rotated_vertices[7]],  # 上面
        [rotated_vertices[0], rotated_vertices[1], rotated_vertices[5], rotated_vertices[4]],  # 側面1
        [rotated_vertices[2], rotated_vertices[3], rotated_vertices[7], rotated_vertices[6]],  # 側面2
        [rotated_vertices[1], rotated_vertices[2], rotated_vertices[6], rotated_vertices[5]],  # 側面3
        [rotated_vertices[0], rotated_vertices[3], rotated_vertices[7], rotated_vertices[4]]   # 側面4
    ]

    #回転盤の表示
    #draw_rotation_circle(ax, omega_local, scale=0.5)

    # 面ごとの色設定
    face_colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']

    # ポリゴンで面を描画（透過あり）
    poly3d = Poly3DCollection(faces, facecolors=face_colors, linewidths=1, edgecolors='k', alpha=0.5)
    ax.add_collection3d(poly3d)

    return []

"""
円盤を描く
"""
def draw_rotation_circle(ax, omega_vec, scale=0.5, steps=100):
    # omega_vec がゼロに近ければスキップ
    if np.linalg.norm(omega_vec) < 1e-6:
        return

    # 回転軸を正規化
    axis = omega_vec / np.linalg.norm(omega_vec)

    # 回転の強さに比例した半径
    radius = scale * np.linalg.norm(omega_vec)

    # 円弧の点を作る（XY平面上の円）
    theta = np.linspace(0, 2 * np.pi, steps)
    circle_pts = np.vstack([radius * np.cos(theta),
                            radius * np.sin(theta),
                            np.zeros_like(theta)])  # XY平面円

    # 回転軸（Z軸→axis）への回転行列を作成
    def rotation_matrix_from_vectors(a, b):
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        if s == 0:
            return np.eye(3)
        kmat = np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
        return np.eye(3) + kmat + (kmat @ kmat) * ((1 - c) / (s ** 2))

    R = rotation_matrix_from_vectors(np.array([0, 0, 1]), axis)

    # 円を回転軸方向に回転
    rotated_circle = R @ circle_pts

    # 描画（原点中心にリング）
    ax.plot(rotated_circle[0], rotated_circle[1], rotated_circle[2], color='red', linewidth=2)



if __name__ == "__main__":
    main()