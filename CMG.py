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
	gyro_datas_x = [-158.264,-107.208,119.385,-11.047,175.171,533.691,552.979,107.666,395.233,-1.526,-244.202,-118.744,269.318,102.57,-147.797,-8.82,123.535,137.207,178.68,89.63,-78.918,-9.399,183.594,88.867,61.829,103.21,38.086]
	gyro_datas_y = [37.476,55.786,77.942,10.803,-132.782,-284.698,-421.326,-533.234,-644.348,-657.104,-482.544,-223.022,-366.302,-572.662,-451.447,-321.991,-296.844,-349.884,-458.679,-426.941,-173.218,-140.808,-656.982,-567.444,-5.402,162.262,129.944]
	gyro_datas_z = [98.541,127.93,72.052,-59.692,-174.652,-217.102,-220.306,-346.832,-318.787,-189.789,-399.17,-669.708,-730.042,-565.43,-557.22,-543.61,-441.01,-327.209,-213.013,-232.239,-466.156,-575.134,-947.266,-59.326,-153.839,-49.774,-127.686]
	
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
    filename = input("保存するファイル名（拡張子.pngは不要）: ")
    plt.savefig(filename + '.png')  # PNG形式で保存
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
	ani = FuncAnimation(fig, update, fargs=(gyro_rad_x, gyro_rad_y, gyro_rad_z, vertices), init_func=anime_init, blit=False, interval=50, save_count=len(gyro_rad_x))
	
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
    max_range = length * 2.0  # 安全マージンを20%くらい取る

    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
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