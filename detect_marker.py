import cv2
import cv2.aruco as aruco
import numpy as np
import os

# 리눅스 GStreamer 충돌(Segmentation fault) 방지를 위해 환경변수 설정
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"

def main():
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    # 인쇄된 마커의 실제 크기 (단위: 미터). 예: 5cm 다면 0.05
    # 정확한 거리를 원하시면 자로 재서 수정하세요.
    marker_length_meters = 0.05

    print("웹캠을 웁니다... (화면 창을 클릭하고 'q'를 누르면 종료됩니다)")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("에러: 카메라를 열 수 없습니다.")
        return

    # 카메라 내부 파라미터(캘리브레이션)를 위한 가상 세팅 설정
    # 정확한 자세 추정을 위해서는 원래 체스보드를 이용한 카메라 캘리브레이션이 필수지만,
    # 여기서는 해상도 기준 대략적인 초점 거리로 행렬(Camera Matrix)을 강제 세팅합니다.
    ret, frame = cap.read()
    if not ret: return
    
    h, w = frame.shape[:2]
    focal_length = w
    center = (w/2, h/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))  # 렌즈 왜곡은 0으로 가정

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            
            # 카메라 파라미터와 마커 실제 크기를 바탕으로 XYZ 좌표와 3D 자세 회전값(rvec) 계산
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length_meters, camera_matrix, dist_coeffs)

            for i in range(len(ids)):
                # rvecs[i]는 마커의 3D 회전각, tvecs[i]는 카메라로부터의 X,Y,Z 이동 거리 정보입니다.
                x, y, z = tvecs[i][0]

                # 3D X,Y,Z 축을 마커 위에 시각화 (빨간색:X, 초록색:Y, 파란색:Z축)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_length_meters)
                
                # Z 거리 및 ID 표시
                cv2.putText(frame, f"ID: {ids[i][0]} Z:{z:.2f}m",
                            (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('ArUco Marker Detection (3D Pose)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
