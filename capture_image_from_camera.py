import cv2

cam_port = 0  # Change the value to the appropriate camera port (0 orsammy4 1)

cam = cv2.VideoCapture(cam_port)

# Read the input person name
inp = input('Enter person name: ')

while True:
    # Read the frame from the camera
    ret, frame = cam.read()

    # Display the frame
    cv2.imshow(inp, frame.astype('uint8'))  # Convert the frame to the appropriate data type

    # Wait for key press (0xFF is used for 64-bit compatibility)
    key = cv2.waitKey(1) & 0xFF
    # If 's' is pressed, save the image
    if key == ord('s'):
        cv2.imwrite(f"{inp}.png", frame)
        print("Image taken")
        break

    # If 'q' is pressed, exit the loop
    if key == ord('q'):
        break

# Release the camera and close any open windows
cam.release()
cv2.destroyAllWindows()
