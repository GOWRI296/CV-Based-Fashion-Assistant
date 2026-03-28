import cv2
import numpy as np
from collections import Counter

MATCHING_SUGGESTIONS = {
    "Black": {"match": ["White", "Grey", "Beige", "Red", "Blue"], "style": "Black goes with almost everything."},
    "White": {"match": ["Black", "Blue", "Grey", "Brown", "Olive"], "style": "White is fresh and versatile."},
    "Red": {"match": ["Black", "White", "Grey", "Beige"], "style": "Red works best with neutral tones."},
    "Blue": {"match": ["White", "Grey", "Beige", "Brown", "Black"], "style": "Blue is stylish and easy to pair."},
    "Green": {"match": ["White", "Black", "Brown", "Beige", "Grey"], "style": "Green matches well with earthy shades."},
    "Yellow": {"match": ["Black", "White", "Blue", "Grey"], "style": "Yellow looks better with simple tones."},
    "Orange": {"match": ["White", "Black", "Grey", "Navy"], "style": "Orange is vibrant and bold."},
    "Purple": {"match": ["White", "Black", "Grey", "Beige"], "style": "Purple gives a rich and stylish vibe."},
    "Pink": {"match": ["White", "Grey", "Black", "Blue"], "style": "Pink pairs well with soft or dark tones."},
    "Brown": {"match": ["Beige", "White", "Olive", "Black", "Cream"], "style": "Brown gives a warm earthy aesthetic."},
    "Grey": {"match": ["Black", "White", "Blue", "Pink", "Burgundy"], "style": "Grey is elegant and minimal."},
    "Beige": {"match": ["Brown", "White", "Black", "Olive", "Navy"], "style": "Beige gives a premium soft look."}
}

REFERENCE_COLORS = {
    "Black": (30, 30, 30),
    "White": (230, 230, 230),
    "Red": (40, 40, 200),
    "Blue": (200, 70, 30),
    "Green": (50, 150, 50),
    "Yellow": (0, 220, 220),
    "Orange": (0, 140, 255),
    "Purple": (150, 70, 150),
    "Pink": (180, 105, 255),
    "Brown": (42, 42, 165),
    "Grey": (128, 128, 128),
    "Beige": (180, 220, 230),
    "Olive": (60, 100, 60),
    "Navy": (128, 0, 0),
    "Cream": (200, 240, 255),
    "Burgundy": (60, 20, 120)
}

def closest_color_name(bgr):
    min_dist = float("inf")
    best_match = "Unknown"
    for color_name, ref_bgr in REFERENCE_COLORS.items():
        dist = np.linalg.norm(np.array(bgr) - np.array(ref_bgr))
        if dist < min_dist:
            min_dist = dist
            best_match = color_name
    return best_match

def remove_skin_pixels(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    non_skin_mask = cv2.bitwise_not(skin_mask)
    filtered = cv2.bitwise_and(roi, roi, mask=non_skin_mask)
    return filtered

def get_dominant_color(roi):
    pixels = roi.reshape(-1, 3)
    valid_pixels = []
    for pixel in pixels:
        if np.mean(pixel) > 25:
            valid_pixels.append(tuple(pixel))

    if len(valid_pixels) == 0:
        return (128, 128, 128)

    quantized_pixels = []
    for p in valid_pixels:
        quantized = tuple((np.array(p) // 32) * 32)
        quantized_pixels.append(quantized)

    most_common = Counter(quantized_pixels).most_common(1)[0][0]
    return most_common

def draw_color_box(frame, color_bgr, x, y, label):
    color_bgr = tuple(int(c) for c in color_bgr)
    cv2.rectangle(frame, (x, y), (x + 60, y + 60), color_bgr, -1)
    cv2.rectangle(frame, (x, y), (x + 60, y + 60), (255, 255, 255), 2)
    cv2.putText(frame, label, (x, y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open webcam")
    input("Press Enter to exit...")
    raise SystemExit

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    x1 = int(w * 0.35)
    y1 = int(h * 0.35)
    x2 = int(w * 0.65)
    y2 = int(h * 0.75)

    roi = frame[y1:y2, x1:x2]
    filtered_roi = remove_skin_pixels(roi)
    dominant_bgr = get_dominant_color(filtered_roi)
    detected_color = closest_color_name(dominant_bgr)

    suggestion_data = MATCHING_SUGGESTIONS.get(
        detected_color,
        {"match": ["Black", "White", "Grey"], "style": "Balanced neutral shades work well."}
    )

    matches = suggestion_data["match"]
    style_tip = suggestion_data["style"]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Place shirt inside box", (x1 - 10, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"Detected Outfit Color: {detected_color}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, f"Style Tip: {style_tip}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 2)

    draw_color_box(frame, dominant_bgr, 20, 100, detected_color)

    cv2.putText(frame, "Best Matching Colors:", (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

    start_x = 20
    for i, match_color in enumerate(matches[:5]):
        color_bgr = REFERENCE_COLORS.get(match_color, (128, 128, 128))
        draw_color_box(frame, color_bgr, start_x + i * 90, 220, match_color)

    cv2.putText(frame, f"Pants/Shoes: {', '.join(matches[:3])}", (20, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.putText(frame, f"Accessories: {', '.join(matches[2:5])}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow("Outfit Color Matcher", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()