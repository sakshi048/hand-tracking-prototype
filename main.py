import cv2
import time
import numpy as np

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: camera not accessible")
        return

    bg_model = None
    warmup_frames = 60
    frame_count = 0

    prev_time = time.time()
    fps = 0.0

    
    prev_hand_center = None
    smoothing_alpha = 0.6     
    state_display = "SAFE"
    prev_state = "SAFE"
    state_count = 0
    state_stable_frames = 3    

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: frame not read")
            break

        frame = cv2.resize(frame, (640, 480))
        frame_count += 1

        # STEP 1: build background model for first few frames
        if frame_count <= warmup_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype("float32")
            if bg_model is None:
                bg_model = gray.copy()
            else:
                cv2.accumulateWeighted(gray, bg_model, 1.0 / (frame_count))
            cv2.putText(frame, f"Calibrating background: {frame_count}/{warmup_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # STEP 2: background subtraction (with blur to reduce noise)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        bg_uint8 = cv2.convertScaleAbs(bg_model)
        bg_blur = cv2.GaussianBlur(bg_uint8, (5, 5), 0)

        diff = cv2.absdiff(bg_blur, gray_blur)

        # STEP 3: threshold to get foreground mask
        _, fg_mask = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)

        # STEP 4: clean the mask (slightly stronger)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel, iterations=2)


        # STEP 5: find contours in foreground mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_hand_center = None
        hand_center = None

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > 1200:  # ignore very small noise; increase if tiny hands
                cv2.drawContours(frame, [largest], -1, (255, 0, 0), 2)
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    raw_hand_center = (cx, cy)

        # ===== smoothing (EMA) for center to reduce jumps =====
        if raw_hand_center is not None:
            if prev_hand_center is None:
                hand_center = raw_hand_center
            else:
                px, py = prev_hand_center
                cx, cy = raw_hand_center
                sx = int(smoothing_alpha * cx + (1 - smoothing_alpha) * px)
                sy = int(smoothing_alpha * cy + (1 - smoothing_alpha) * py)
                hand_center = (sx, sy)
            prev_hand_center = hand_center
        else:
            # if no detection this frame, slowly forget previous center
            prev_hand_center = None
            hand_center = None

        if hand_center is not None:
            cv2.circle(frame, hand_center, 8, (0, 255, 0), -1)
            cv2.putText(frame, "HAND", (hand_center[0] + 10, hand_center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ==== VIRTUAL BOUNDARY ====
        boundary_x = 300
        cv2.line(frame, (boundary_x, 0), (boundary_x, 480), (0, 255, 255), 3)

        # ==== DISTANCE LOGIC (with debounce) ====
        computed_state = "SAFE"
        if hand_center is not None:
            distance = abs(hand_center[0] - boundary_x)
            if distance > 120:
                computed_state = "SAFE"
            elif distance > 50:
                computed_state = "WARNING"
            else:
                computed_state = "DANGER"

        # debounce logic: require state to be stable for several frames
        if computed_state == prev_state:
            state_count += 1
        else:
            state_count = 1
            prev_state = computed_state

        if state_count >= state_stable_frames:
            state_display = prev_state  # commit to the stable state

        # Show state on screen (color-coded)
        color = (0, 255, 0) if state_display == "SAFE" else (0, 255, 255) if state_display == "WARNING" else (0, 0, 255)
        cv2.putText(frame, f"STATE: {state_display}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if state_display == "DANGER":
            cv2.putText(frame, "DANGER  DANGER", (150, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # STEP 6: FPS calculation and display
        cur_time = time.time()
        if cur_time != prev_time:
            fps = 1.0 / (cur_time - prev_time)
        prev_time = cur_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # STEP 7: show windows
        cv2.imshow("Webcam", frame)
        cv2.imshow("Foreground Mask", fg_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
