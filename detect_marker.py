import cv2
import cv2.aruco as aruco

def main():
    # 생성할 때 사용했던 것과 동일한 사전(Dictionary) 세팅
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()

    # 웹캠 켜기 (리눅스 기준 기본 카메라 디바이스는 0번 /dev/video0)
    print("웹캠을 켭니다... (화면 창을 클릭하고 'q'를 누르면 종료됩니다)")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("에러: 카메라를 열 수 없습니다. 카메라가 리눅스에 잘 연결되어 있는지 확인해주세요.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라에서 영상을 읽을 수 없습니다.")
            break

        # RGB 이미지를 흑백으로 변환 (인식률 및 속도 향상)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 마커 감지 수행
        corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)

        # 감지된 마커가 있다면 화면에 그리기
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            
            # 각 마커의 ID와 중심점 터미널 출력 및 화면 표시
            for i in range(len(ids)):
                c = corners[i][0]
                center_x = int((c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4)
                center_y = int((c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4)
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
                
                # 좌상단에 텍스트로 정보 그리기
                cv2.putText(frame, f"ID: {ids[i][0]} Center: ({center_x}, {center_y})",
                            (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 결과 화면 출력
        cv2.imshow('ArUco Marker Detection', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
