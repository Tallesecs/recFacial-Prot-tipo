import face_recognition
import cv2
import numpy as np

# versao demo para reconhecimento
video_capture = cv2.VideoCapture(0)

# Cod para aprender como reconhecer a foto no banco.
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]


biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Criar um array para faces conhecidas
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden"
]

# Variáveis
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Pegando um frame do video
    ret, frame = video_capture.read()

    # Redimensiona o frame do video
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Converter a imagem de BGR para RGB (face recognition usa)
    rgb_small_frame = small_frame[:, :, ::-1]


    if process_this_frame:
        # Descobrir as faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Ver se as faces vai dar match
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # primeira face detectada, primeira que aparece

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Mostrar os resultados
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Escalar a localização das faces
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Desenha um retangulo na face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Escrever uma label com o nome
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Mostrar o video
    cv2.imshow('Video', frame)

    # apertar q para fechar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar webcam
video_capture.release()
cv2.destroyAllWindows()
