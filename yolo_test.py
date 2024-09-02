import zmq
import cv2
import torch
import pickle
import numpy as np
from ultralytics import YOLO
import time

model = YOLO("model/yolov8-custom.pt")

with open('model/calibrated_distance_model_width.pkl', 'rb') as f:
    distance_model_width = pickle.load(f)

with open('model/calibrated_distance_model_height.pkl', 'rb') as f:
    distance_model_height = pickle.load(f)

def estimate_distance(box_side, side_type):
    box_side = np.array([[box_side]])
    if(side_type == "height"):
        distance = 1/distance_model_height.predict(box_side)[0]
    else:
        distance = 1/distance_model_width.predict(box_side)[0]

    return distance

def is_cut_off(box, image_shape):
    height, width, _ = image_shape
    x1, y1, x2, y2 = box

    return y1 <= 0, y2 >= height, x1 <= 0, x2 >= width


def run_detection_loop():
    context = zmq.Context()

    server_ip = "10.11.25.26"

    # Receiver Socket konfigurieren (zum Empfang von Frames)
    receiver_socket = context.socket(zmq.PULL)
    receiver_socket.setsockopt(zmq.CONFLATE, 1)  # Ältere Nachrichten verwerfen
    receiver_socket.connect(f"tcp://{server_ip}:5555")  # IP-Adresse des Servers

    # Sender Socket konfigurieren (zum Senden der Ergebnisse)
    sender_socket = context.socket(zmq.PUSH)
    sender_socket.connect(f"tcp://{server_ip}:5556")  # IP-Adresse des Servers

    while True:
        # Frame von ZeroMQ empfangen
        frame_bytes = receiver_socket.recv()

        # JPEG-Frame decodieren
        frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Bild in RGB umwandeln und Größe anpassen
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (224, 224))  # Stellen Sie sicher, dass die Größe konsistent ist

        # Inferenzzeit messen
        start_time = time.time()
        results = model(resized_frame)
        end_time = time.time()

        print(f"Inferenzzeit: {end_time - start_time} Sekunden")

        # Debugging-Ausgabe der Struktur von Results
        if isinstance(results, list) and hasattr(results[0], 'boxes'):
            detections = results[0].boxes.data.cpu().numpy()  # In numpy-Array umwandeln

            if len(detections) == 0:
                print("No detections found")
                sender_socket.send(pickle.dumps([-1, "No detections found"]))

            # Schleife über die Erkennungen
            distances = []
            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection[:6]

                # Berechnen der Box-Dimensionen
                box_width = x2 - x1
                box_height = y2 - y1

                # Überprüfen, ob das Objekt abgeschnitten ist
                cut_off_top, cut_off_bottom, cut_off_left, cut_off_right = is_cut_off([x1, y1, x2, y2], frame.shape)

                # Bestimmen, welches Modell verwendet werden soll
                if cut_off_left or cut_off_right:
                    distance = estimate_distance(box_height, "height")
                else:
                    distance = estimate_distance(box_width, "width")

                # Ausgabe der Entfernung
                distances.append([conf, distance])
                print(distances)
                sender_socket.send(pickle.dumps(distances))

            # Frame anzeigen
            #clear_output(wait=True)
            #plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            #plt.axis('off')
            #plt.show()
            # display(plt.gcf())

        else:
            print("Results object does not contain boxes attribute")

if __name__ == "__main__":
    print(run_detection_loop())