import cv2

print("Starting webcam test...")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam")
    input("Press Enter to exit...")
    raise SystemExit

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame")
        break

    cv2.imshow("Webcam Test - Press q to Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()