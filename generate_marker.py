import cv2
import cv2.aruco as aruco
import numpy as np
import os

def main():
    # 사용할 ArUco 사전 선택 (6x6 비트, 250개 ID)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

    # 마커를 그릴 이미지 생성 (픽셀 크기: 200x200), ID: 42
    marker_id = 42
    marker_size = 200
    
    # 빈 캔버스 생성
    marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)

    # 마커 이미지 생성
    marker_image = aruco.drawMarker(dictionary, marker_id, marker_size, marker_image, 1)

    # 이미지 저장
    save_path = "marker_42.png"
    cv2.imwrite(save_path, marker_image)
    print(f"[성공] ID {marker_id} 마커 이미지가 '{os.path.abspath(save_path)}' 경로에 저장되었습니다.")

if __name__ == "__main__":
    main()
