def image_callback(self, msg: Image):
        now = rospy.Time.now().to_sec()
        img = self.msg_to_cv2(msg) # 이미지는 미리 변환
        if img is None: return

        # [1. 장애물 회피 상태 머신] - 기존 로직 유지하되 return 위치 확인
        if self.state == "BACK":
            # ... (후진 로직)
            return
        if self.state == "ESCAPE_TURN":
            # ... (회전 로직)
            return
        if self.state == "ESCAPE_STRAIGHT":
            # ... (직진 탈출 로직)
            return

        # [2. 장애물 감지 시 상태 전환]
        if self.front_dist < 0.45:
            self.state = "BACK"
            self.state_start = now
            return

        # [3. 미션 수행: 콘 감지 vs 차선 추적]
        # 만약 빨간 콘이 보이면 콘 제어를 우선함
        if self.detect_cone(img):
            self.cone_control(img)
            rospy.loginfo("🔴 콘 발견! 콘 추적 중...")
        else:
            # 콘이 없으면 기존 차선 인식 실행
            self.follow_lane(img) # 차선 인식 부분을 별도 함수로 빼는 것 추천
