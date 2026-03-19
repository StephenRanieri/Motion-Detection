import cv2
import time
import webbrowser
from collections import deque

# --- Configuration ---
CAMERA_INDEX = 0       # 0 = default/front-facing camera
SENSITIVITY = 1500     # Min contour area to track (filters out noise)
BLUR_SIZE = 21         # Gaussian blur kernel size (must be odd)
THRESHOLD = 25         # Pixel difference threshold

# Swipe detection tuning
SWIPE_HISTORY = 12     # How many frames of centroid history to keep
SWIPE_MIN_X = 180      # Minimum horizontal pixels traveled to count as swipe
SWIPE_MAX_Y = 80       # Maximum vertical drift allowed (keeps it horizontal)
SWIPE_MAX_TIME = 0.6   # Swipe must complete within this many seconds
COOLDOWN = 2.5         # Seconds to wait before allowing another swipe


def get_motion_centroid(contours):
    """Return the weighted centroid of all significant motion contours."""
    total_area = 0
    cx_sum = 0
    cy_sum = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < SENSITIVITY:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cx_sum += cx * area
        cy_sum += cy * area
        total_area += area

    if total_area == 0:
        return None
    return (int(cx_sum / total_area), int(cy_sum / total_area))


def detect_swipe(history):
    """
    Given a deque of (timestamp, x, y) tuples, return 'left', 'right', or None.
    A swipe is: fast horizontal movement, limited vertical drift, within time window.
    """
    if len(history) < 4:
        return None

    oldest_t, oldest_x, oldest_y = history[0]
    newest_t, newest_x, newest_y = history[-1]

    elapsed = newest_t - oldest_t
    if elapsed > SWIPE_MAX_TIME:
        return None

    dx = newest_x - oldest_x
    dy = abs(newest_y - oldest_y)

    if dy > SWIPE_MAX_Y:
        return None  # Too much vertical movement — not a clean horizontal swipe

    if dx > SWIPE_MIN_X:
        return "right"
    if dx < -SWIPE_MIN_X:
        return "left"

    return None


def run():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌  Could not open camera.")
        return

    print("✅  Camera ready. Swipe your hand left or right to open YouTube.")
    print("    Press 'q' to quit.\n")

    for _ in range(5):
        cap.read()

    ret, prev_frame = cap.read()
    if not ret:
        print("❌  Failed to grab initial frame.")
        cap.release()
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (BLUR_SIZE, BLUR_SIZE), 0)

    history = deque(maxlen=SWIPE_HISTORY)
    last_swipe_time = 0
    status_text = "Ready — swipe to open YouTube"
    status_color = (200, 200, 200)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame so it feels natural (like a selfie camera)
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)

        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroid = get_motion_centroid(contours)
        now = time.time()
        cooldown_active = (now - last_swipe_time) < COOLDOWN

        if centroid:
            history.append((now, centroid[0], centroid[1]))

            # Draw centroid dot
            cv2.circle(frame, centroid, 10, (0, 255, 255), -1)

            # Draw trail
            pts = list(history)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                cv2.line(frame, (pts[i-1][1], pts[i-1][2]),
                         (pts[i][1], pts[i][2]), color, 2)

            if not cooldown_active:
                swipe = detect_swipe(history)
                if swipe:
                    print(f"👋  {swipe.upper()} swipe detected — opening YouTube!")
                    webbrowser.open("https://www.youtube.com")
                    last_swipe_time = now
                    history.clear()
                    status_text = f"✓ {swipe.capitalize()} swipe! Opening YouTube..."
                    status_color = (0, 255, 100)
        else:
            # No motion — slowly clear history so stale data doesn't trigger swipes
            if history and (now - history[-1][0]) > 0.3:
                history.clear()

        # Cooldown countdown
        if cooldown_active:
            remaining = COOLDOWN - (now - last_swipe_time)
            status_text = f"Cooldown: {remaining:.1f}s"
            status_color = (100, 100, 255)
        elif not centroid:
            status_text = "Ready — swipe to open YouTube"
            status_color = (200, 200, 200)

        # ── UI overlay ────────────────────────────────────────────────────────
        h, w = frame.shape[:2]

        # Semi-transparent bottom bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, status_text, (12, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        # Swipe direction arrows (subtle guide)
        cv2.putText(frame, "<-- swipe -->", (w // 2 - 75, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)

        prev_gray = gray

        cv2.imshow("Swipe to YouTube  [q = quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋  Quit.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
