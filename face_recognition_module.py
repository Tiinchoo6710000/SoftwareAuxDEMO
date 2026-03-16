import face_recognition
import os
import cv2

known_faces = []
known_names = []

def load_known_faces(folder="faces/rostros_registrados"):

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        img = face_recognition.load_image_file(path)

        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 0:

            known_faces.append(encodings[0])

            name = os.path.splitext(file)[0]

            known_names.append(name)


def recognize(frame, bbox):

    x1,y1,x2,y2 = bbox

    face = frame[y1:y2, x1:x2]

    if face.size == 0:
        return "Unknown"

    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    enc = face_recognition.face_encodings(rgb)

    if len(enc) == 0:
        return "Unknown"

    matches = face_recognition.compare_faces(
        known_faces,
        enc[0]
    )

    if True in matches:

        index = matches.index(True)

        return known_names[index]

    return "Unknown"