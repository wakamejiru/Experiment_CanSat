import numpy as np
air_density = 1.206 # 1気圧の空気密度
parachute_fai = 0.18 # パラシュートの半径
inner_fai = 0.05
parachute_s = parachute_fai * parachute_fai * np.pi  - (inner_fai*inner_fai*np.pi) # パラシュートの面積
drag_coefficent = 0.5 # 抵抗係数
gravitational_acceleration = 9.80665

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
 del_time = 0.1
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

if __name__ == "__main__":
 main()