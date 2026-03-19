import cv2
import time
import random
import numpy as np
from collections import deque

# --- Configuration ---
CAMERA_INDEX = 0
SENSITIVITY = 1200
BLUR_SIZE = 21
THRESHOLD = 25

# Swipe detection
SWIPE_HISTORY = 14
SWIPE_MIN_X = 160       # Min horizontal travel for left/right
SWIPE_MIN_Y = 120       # Min vertical travel for up/down
SWIPE_MAX_DRIFT = 90    # Max off-axis drift to keep swipe clean
SWIPE_MAX_TIME = 0.7
COOLDOWN = 1.2

# Game settings
TIME_LIMIT = 4.0        # Seconds to swipe before it counts as a miss
DIRECTIONS = ["up", "down", "left", "right"]

# Colors (BGR)
C_WHITE   = (255, 255, 255)
C_BLACK   = (0,   0,   0)
C_GREEN   = (80,  220, 100)
C_RED     = (60,  60,  220)
C_YELLOW  = (0,   220, 220)
C_CYAN    = (220, 220, 0)
C_GRAY    = (120, 120, 120)
C_DARK    = (20,  20,  20)
C_ORANGE  = (0,   160, 255)


# ── Arrow drawing ─────────────────────────────────────────────────────────────

def draw_arrow(frame, direction, cx, cy, size, color, thickness=6):
    """Draw a bold arrow pointing in the given direction."""
    s = size
    if direction == "right":
        pts_shaft = [(cx - s, cy), (cx + s // 2, cy)]
        tip = (cx + s, cy)
        head = [(cx + s // 2, cy - s // 2), (cx + s, cy), (cx + s // 2, cy + s // 2)]
    elif direction == "left":
        pts_shaft = [(cx + s, cy), (cx - s // 2, cy)]
        tip = (cx - s, cy)
        head = [(cx - s // 2, cy - s // 2), (cx - s, cy), (cx - s // 2, cy + s // 2)]
    elif direction == "up":
        pts_shaft = [(cx, cy + s), (cx, cy - s // 2)]
        tip = (cx, cy - s)
        head = [(cx - s // 2, cy - s // 2), (cx, cy - s), (cx + s // 2, cy - s // 2)]
    else:  # down
        pts_shaft = [(cx, cy - s), (cx, cy + s // 2)]
        tip = (cx, cy + s)
        head = [(cx - s // 2, cy + s // 2), (cx, cy + s), (cx + s // 2, cy + s // 2)]

    cv2.line(frame, pts_shaft[0], pts_shaft[1], color, thickness, cv2.LINE_AA)
    cv2.fillPoly(frame, [np.array(head, np.int32)], color)


def draw_big_arrow(frame, direction, color, alpha=1.0):
    """Draw a large centered arrow overlay."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    size = min(w, h) // 5

    overlay = frame.copy()
    # Outer glow ring
    cv2.circle(overlay, (cx, cy), size + 30, color, 3, cv2.LINE_AA)
    draw_arrow(overlay, direction, cx, cy, size, color, thickness=10)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ── Motion centroid ───────────────────────────────────────────────────────────

def get_motion_centroid(contours):
    total_area, cx_sum, cy_sum = 0, 0, 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < SENSITIVITY:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx_sum += int(M["m10"] / M["m00"]) * area
        cy_sum += int(M["m01"] / M["m00"]) * area
        total_area += area
    if total_area == 0:
        return None
    return (int(cx_sum / total_area), int(cy_sum / total_area))


# ── Swipe detection (4 directions) ───────────────────────────────────────────

def detect_swipe(history):
    if len(history) < 4:
        return None

    oldest_t, oldest_x, oldest_y = history[0]
    newest_t, newest_x, newest_y = history[-1]

    if newest_t - oldest_t > SWIPE_MAX_TIME:
        return None

    dx = newest_x - oldest_x
    dy = newest_y - oldest_y   # positive = downward in image coords
    adx, ady = abs(dx), abs(dy)

    # Dominant axis must clearly win
    if adx > ady:
        if adx < SWIPE_MIN_X or ady > SWIPE_MAX_DRIFT:
            return None
        return "right" if dx > 0 else "left"
    else:
        if ady < SWIPE_MIN_Y or adx > SWIPE_MAX_DRIFT:
            return None
        return "down" if dy > 0 else "up"


# ── HUD helpers ───────────────────────────────────────────────────────────────

def draw_score_bar(frame, score, streak, lives):
    h, w = frame.shape[:2]
    bar_h = 52
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), C_DARK, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, f"SCORE  {score:04d}", (14, 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, C_CYAN, 2, cv2.LINE_AA)

    streak_color = C_ORANGE if streak >= 3 else C_WHITE
    cv2.putText(frame, f"STREAK x{streak}", (w // 2 - 70, 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, streak_color, 2, cv2.LINE_AA)

    heart = "♥"
    lives_str = " ".join(["O" if i < lives else "X" for i in range(3)])
    life_color = C_RED if lives == 1 else C_GREEN
    cv2.putText(frame, f"LIVES  {lives_str}", (w - 200, 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, life_color, 2, cv2.LINE_AA)


def draw_timer_bar(frame, elapsed, limit):
    h, w = frame.shape[:2]
    ratio = max(0.0, 1.0 - elapsed / limit)
    bar_w = int((w - 40) * ratio)
    color = C_GREEN if ratio > 0.5 else (C_ORANGE if ratio > 0.25 else C_RED)
    cv2.rectangle(frame, (20, h - 18), (20 + bar_w, h - 6), color, -1)
    cv2.rectangle(frame, (20, h - 18), (w - 20, h - 6), C_GRAY, 1)


def draw_feedback(frame, text, color, size=1.4):
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, size, 3)
    cx = (w - tw) // 2
    cy = h // 2 + 80
    cv2.putText(frame, text, (cx + 2, cy + 2), cv2.FONT_HERSHEY_DUPLEX, size, C_BLACK, 4, cv2.LINE_AA)
    cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, size, color, 3, cv2.LINE_AA)


def draw_game_over(frame, score, high_score):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), C_BLACK, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    lines = [
        ("GAME OVER", 1.8, C_RED,   h // 2 - 80),
        (f"SCORE:  {score}", 1.2, C_WHITE, h // 2 - 10),
        (f"BEST:   {high_score}", 1.2, C_CYAN,  h // 2 + 50),
        ("Swipe ANY direction to play again", 0.7, C_GRAY, h // 2 + 120),
        ("Press Q to quit", 0.6, C_GRAY, h // 2 + 160),
    ]
    for text, size, color, y in lines:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, size, 2)
        x = (w - tw) // 2
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, size, color, 2, cv2.LINE_AA)


# ── Main game loop ────────────────────────────────────────────────────────────

def run():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌  Could not open camera.")
        return

    print("✅  Camera ready. Swipe to play!")
    print("    Up / Down / Left / Right  —  Press Q to quit\n")

    for _ in range(5):
        cap.read()

    ret, prev_frame = cap.read()
    if not ret:
        print("❌  Failed to grab initial frame.")
        cap.release()
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (BLUR_SIZE, BLUR_SIZE), 0)

    history        = deque(maxlen=SWIPE_HISTORY)
    last_swipe_t   = 0

    # Game state
    score       = 0
    high_score  = 0
    lives       = 3
    streak      = 0
    target      = random.choice(DIRECTIONS)
    prompt_t    = time.time()
    feedback    = None      # (text, color, expire_time)
    game_over   = False
    waiting_restart = False  # debounce restart swipe

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)

        diff    = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroid = get_motion_centroid(contours)
        now      = time.time()
        cooldown_active = (now - last_swipe_t) < COOLDOWN

        # Track centroid history
        if centroid:
            history.append((now, centroid[0], centroid[1]))
            cv2.circle(frame, centroid, 8, C_YELLOW, -1)
            pts = list(history)
            for i in range(1, len(pts)):
                a = i / len(pts)
                col = (0, int(220 * a), int(220 * (1 - a)))
                cv2.line(frame, (pts[i-1][1], pts[i-1][2]),
                         (pts[i][1],   pts[i][2]),   col, 2)
        else:
            if history and (now - history[-1][0]) > 0.3:
                history.clear()

        swipe = None
        if not cooldown_active and centroid:
            swipe = detect_swipe(history)
            if swipe:
                last_swipe_t = now
                history.clear()

        # ── Game over screen ──────────────────────────────────────────────────
        if game_over:
            draw_game_over(frame, score, high_score)
            if swipe and not waiting_restart:
                # Reset game
                score = 0
                lives = 3
                streak = 0
                target = random.choice(DIRECTIONS)
                prompt_t = now
                feedback = None
                game_over = False
                waiting_restart = False
            elif not swipe:
                waiting_restart = False  # allow restart once swipe clears

        else:
            # ── Active gameplay ───────────────────────────────────────────────
            elapsed = now - prompt_t

            # Check timeout (missed)
            if elapsed >= TIME_LIMIT:
                lives -= 1
                streak = 0
                feedback = ("MISS!", C_RED, now + 0.8)
                target = random.choice(DIRECTIONS)
                prompt_t = now

                if lives <= 0:
                    high_score = max(high_score, score)
                    game_over = True
                    waiting_restart = True

            # Check swipe result
            elif swipe:
                if swipe == target:
                    pts = 10 + streak * 5
                    score += pts
                    streak += 1
                    label = f"+{pts}  {'PERFECT!' if streak >= 3 else 'CORRECT!'}"
                    feedback = (label, C_GREEN, now + 0.7)
                else:
                    lives -= 1
                    streak = 0
                    feedback = (f"WRONG!  ({swipe})", C_RED, now + 0.8)
                    if lives <= 0:
                        high_score = max(high_score, score)
                        game_over = True
                        waiting_restart = True

                if not game_over:
                    target = random.choice(DIRECTIONS)
                    prompt_t = now

            # ── Draw target arrow ─────────────────────────────────────────────
            if not game_over:
                pulse = 0.75 + 0.25 * abs(np.sin(now * 4))
                urgency = elapsed / TIME_LIMIT
                arrow_color = (
                    int(C_GREEN[0] * (1 - urgency) + C_RED[0] * urgency),
                    int(C_GREEN[1] * (1 - urgency) + C_RED[1] * urgency),
                    int(C_GREEN[2] * (1 - urgency) + C_RED[2] * urgency),
                )
                draw_big_arrow(frame, target, arrow_color, alpha=pulse)

                # Direction label
                label_map = {"up": "SWIPE UP", "down": "SWIPE DOWN",
                             "left": "SWIPE LEFT", "right": "SWIPE RIGHT"}
                lbl = label_map[target]
                (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
                cv2.putText(frame, lbl, ((w - lw) // 2, h // 2 + min(w, h) // 5 + 45),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, arrow_color, 2, cv2.LINE_AA)

            draw_score_bar(frame, score, streak, lives)
            draw_timer_bar(frame, elapsed if not game_over else 0, TIME_LIMIT)

        # ── Feedback flash ────────────────────────────────────────────────────
        if feedback and now < feedback[2]:
            draw_feedback(frame, feedback[0], feedback[1])
        elif feedback and now >= feedback[2]:
            feedback = None

        prev_gray = gray
        cv2.imshow("Swipe Game  [Q = quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n🏆  Final high score: {high_score}")


if __name__ == "__main__":
    run()
