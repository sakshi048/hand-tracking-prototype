# Hand Tracking – Virtual Boundary Alert System

This project tracks the user's hand in real time using classical computer-vision 
techniques (no MediaPipe / no deep models).  
The system detects when the hand approaches or touches a virtual boundary and 
shows a clear "DANGER DANGER" warning.

## Features
- Real-time webcam hand tracking
- Background subtraction using running average
- Foreground mask creation (Gaussian blur + thresholding + morphology)
- Contour-based hand detection
- Hand center tracking with smoothing
- Virtual boundary line
- Distance-based state classification:
  - SAFE
  - WARNING
  - DANGER
- Visual overlay + FPS counter

## How to Run
1. Install dependencies:
  python -m pip install -r requirements.txt

2. Run the main script:
  python main.py

3. Controls:
- Keep the scene static for first 2 seconds for background calibration.
- Move your hand towards the boundary line to see SAFE → WARNING → DANGER.
- Press `q` to exit.

## Approach Summary
- Convert frames to grayscale.
- Build background model using running average for first N frames.
- Subtract background to extract moving hand region.
- Clean the foreground mask using morphological operations.
- Detect largest contour → treat as hand → compute center using moments.
- Draw a fixed virtual boundary at x=300.
- Compare hand_center_x with boundary_x to compute distance.
- Display state based on thresholds.

## Threshold Logic
- distance > 120 → SAFE  
- 50 < distance ≤ 120 → WARNING  
- distance ≤ 50 → DANGER (shows "DANGER DANGER")

## Requirements
- Python 3
- OpenCV
- NumPy

