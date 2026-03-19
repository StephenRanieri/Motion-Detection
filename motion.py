import cv2
import time
import datetime

# --- Configuration ---
CAMERA_INDEX = 0          # 0 = default/front-facing camera
SENSITIVITY = 500         # Minimum contour area to count as motion (lower = more sensitive)
BLUR_SIZE = 21            # Gaussian blur kernel size (must be odd)
THRESHOLD = 25            # Pixel difference threshold (0-255)
SHOW_CONTOURS = True      # Draw green boxes around motion areas
SHOW_DIFF = False         # Show the difference frame in a second window
SAVE_ON_MOTION = False    # Set True to save a snapshot when motion is first detected


def run_motion_detector():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌  Could not open camera. Check CAMERA_INDEX setting.")
        return

    print("✅  Camera opened. Press 'q' to quit.")
    print(f"    Sensitivity (min area): {SENSITIVITY}")
    print(f"    Threshold:              {THRESHOLD}")
    print()

    # Warm up the camera
    for _ in range(5):
        cap.read()

    ret, prev_frame = cap.read()
    if not ret:
        print("❌  Failed to grab initial frame.")
        cap.release()
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.GAUSSIAN_BLUR if False else cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (BLUR_SIZE, BLUR_SIZE), 0)

    motion_start = None
    snapshot_taken = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Frame grab failed – exiting.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)

        # Compute absolute difference between current and previous frame
        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)

        # Dilate to fill small holes, then find contours
        dilated = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) < SENSITIVITY:
                continue  # Ignore tiny blobs (noise)

            motion_detected = True

            if SHOW_CONTOURS:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ── Overlay status text ──────────────────────────────────────────────
        now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, now, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if motion_detected:
            if motion_start is None:
                motion_start = time.time()
                snapshot_taken = False
                print(f"🔴  Motion detected at {now}")

            elapsed = time.time() - motion_start
            label = f"MOTION  ({elapsed:.1f}s)"
            cv2.putText(frame, label, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Optional: save one snapshot per motion event
            if SAVE_ON_MOTION and not snapshot_taken:
                filename = f"motion_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"   📸  Snapshot saved → {filename}")
                snapshot_taken = True
        else:
            if motion_start is not None:
                print(f"⚪  Motion stopped  (lasted {time.time() - motion_start:.1f}s)")
                motion_start = None
            cv2.putText(frame, "No Motion", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

        # ── Show windows ─────────────────────────────────────────────────────
        cv2.imshow("Motion Detector  [q = quit]", frame)

        if SHOW_DIFF:
            cv2.imshow("Difference Frame", dilated)

        # Update reference frame
        prev_gray = gray

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋  Quit requested.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_motion_detector()