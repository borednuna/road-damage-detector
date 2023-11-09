import cv2

# Open the video file
video_path = '/home/nuna/road-damage-detector/uploads/1699345090.555263.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit()

# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

last_frame = None  # Initialize a variable to store the last frame

# Loop through the video frames to reach the last frame
while True:
    ret, frame = cap.read()
    if not ret:
        # End of the video, break the loop
        break
    last_frame = frame  # Update the last frame with the current frame

# Check if the last frame is valid
if last_frame is not None:
    # Display the last frame
    cv2.imshow('Last Frame', last_frame)
    cv2.waitKey(0)  # Wait for a key press and then close the window
    cv2.destroyAllWindows()
else:
    print("No valid frames found in the video.")

# Release the video file
cap.release()

# Print the total number of frames
print("Total frames in the video:", total_frames)
