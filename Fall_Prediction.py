import numpy as np
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick


air_density = 1.206 # 1気圧の空気密度
parachute_fai = 0.18 # パラシュートの半径
inner_fai = 0.05
parachute_s = parachute_fai * parachute_fai * np.pi  - (inner_fai*inner_fai*np.pi) # パラシュートの面積
drag_coefficent = 0.5 # 抵抗係数
gravitational_acceleration = 9.80665
del_time = 0.1

def main():
 weight = 0.079 # 単位はkg
 hegiht = 8
 wind_speed = 2 # 風速 s/m
 print("=== 垂直方向のみ ===")
 time, v = Freefall(hegiht, 0, weight)
 print(f"落下時間: {time:.2f} 秒")
 print(f"着地時の速度: {v:.2f} m/s")
 print("=== 2D落下（風あり） ===")
 time2d, verocity_z, drift = Freefall2D(hegiht, 0, 0, weight, wind_speed)
 print(f"落下時間: {time2d:.2f} 秒")
 print(f"着地時の速度: {verocity_z:.2f} m/s")
 print(f"横方向のドリフト: {drift:.2f} m")

"""
 横向きと縦向きを考慮する力
 空気抵抗がある場合は速度が時々刻々と変化するため、加算で求める
 height_now 求める時点の高さ
 verocity_now_z 現在の速度
 verocity_now_xy 横方向の力(スカラ量)(ロケットの横向きの力もここに加わる)
 weight 落下物の重さ
"""
def Freefall2D(height_now, verocity_now_z, verocity_now_xy, weight, wind_speed):
 # X方向の空気抵抗(風)
 height = height_now
 verocity_xy = verocity_now_xy
 verocity_z = verocity_now_z
 g = gravitational_acceleration
 time = 0
 x_position = 0

 while height > 0:
  # x方向の空気抵抗（風を考慮）
  relative_wind_x = verocity_now_xy - wind_speed
  Fd_x = 0.5 * air_density * relative_wind_x**2 * drag_coefficent * parachute_s * np.sign(relative_wind_x)
  a_x = -Fd_x / weight
  verocity_xy += a_x * del_time
  x_position += verocity_xy * del_time

  # z方向の重力と空気抵抗
  Fd_z = 0.5* air_density * verocity_z**2 * drag_coefficent * parachute_s
  a_z = g - Fd_z / weight
  verocity_z += a_z * del_time
  height -= verocity_z * del_time
  time += del_time

  return time, verocity_z, x_position

"""
 直線的な落下
 空気抵抗がある場合は速度が時々刻々と変化するため、加算で求める
 height_now 求める時点の高さ
 verocity_now_z 現在の速度
 weight 落下物の重さ
"""
def Freefall(height_now, verocity_now_z, weight):
 del_time = 0.1
 height = height_now
 verocity_z = verocity_now_z
 g = gravitational_acceleration
 time = 0
 
 while height > 0:
   Fd_z = 0.5* air_density * verocity_z**2 * drag_coefficent * parachute_s
   a_z = g - Fd_z / weight
   verocity_z += a_z * del_time
   height -= verocity_z * del_time
   time += del_time

 return time, verocity_z


#ここから先は，実測値に対する計算を行う


"""
 9軸センサによる姿勢計算
 time_array 時間配列
 gyro_data ジャイロ配列
 accel_data 加速度配列
 mag_data 磁気配列
"""
def MadgwickCalc(time_array, gyro_data, accel_data, mag_data, sample_number):
 madgwick_filter = Madgwick(sampleperiod=del_time)
 q = np.array([1, 0, 0, 0])  # 初期クォータニオン（単位クォータニオン）
 # 結果を保存する配列
 quaternions = np.zeros((sample_number, 4))
 euler_angles = np.zeros((sample_number, 3))  # [Roll, Pitch, Yaw]（単位: ラジアン）
 # --- フィルタ更新ループ ---
 for i in range(sample_number):  
  # ここでは磁気センサも利用するので updateMARG を利用します
  q = madgwick_filter.updateMARG(q, gyr=gyro_data[i, :], acc=accel_data[i, :], mag=mag_data[i, :])
  quaternions[i, :] = q
    
  # クォータニオンからオイラー角（Roll, Pitch, Yaw）に変換
  q0, q1, q2, q3 = q
  # Roll: X軸周りの回転
  roll = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))
  # Pitch: Y軸周り（ここでは arcsin で得る）
  pitch = np.arcsin(2*(q0*q2 - q3*q1))
  # Yaw: Z軸周りの回転
  yaw = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))
  euler_angles[i, :] = np.array([roll, pitch, yaw])
  # --- 結果のプロット ---
  plt.figure(figsize=(10, 6))
  plt.plot(time_array, euler_angles[:, 0], label='Roll')
  plt.plot(time_array, euler_angles[:, 1], label='Pitch')
  plt.plot(time_array, euler_angles[:, 2], label='Yaw')
  plt.xlabel('Time [s]')
  plt.ylabel('Angle [rad]')
  plt.title('Attitude Estimation using Madgwick Filter')
  plt.legend()
  plt.grid(True)
  # ファイル名の入力
  filename = input("保存するファイル名（拡張子.pngは不要）: ")
  plt.savefig(filename + '.png')  # PNG形式で保存
  plt.close()  # プロットを閉じる




if __name__ == "__main__":
 main()