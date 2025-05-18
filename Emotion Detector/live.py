import cv2
import torch
import numpy as np
from modelarch import CustomCNN
from PIL import Image
import torchvision.transforms as transforms

# Load your PyTorch model
model = CustomCNN(num_classes=7)
model.load_state_dict(torch.load('models/modeldrop4.pth', map_location=torch.device('cpu')))
model.eval()

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Define the same transforms used during training
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_pil = Image.fromarray(face)  # Convert to PIL image
        face_tensor = transform(face_pil).unsqueeze(0)  # Shape: [1, 1, 48, 48]

        with torch.no_grad():
            output = model(face_tensor)
            predicted = torch.argmax(output, dim=1).item()
            emotion = emotion_labels[predicted]

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    cv2.imshow('Live Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
