import cv2
import mediapipe as mp
import numpy as np
import os
import json
from datetime import datetime

class RecolectorDataset:
    def __init__(self, directorio_base="dataset_senas"):
        self.directorio_base = directorio_base
        
        # Inicializar diccionario de señas
        self.senas = {
            "hola": 30,
            "gracias": 30,
            "por_favor": 30,
            "ayuda": 30,
            "si": 30,
            "no": 30
        }
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Crear estructura de directorios
        self.crear_directorios()  # Esta línea debe estar después de inicializar `self.senas`
        
        # Metadata del dataset
        self.metadata = {
            "fecha_creacion": datetime.now().strftime("%Y-%m-%d"),
            "total_senas": len(self.senas),
            "grabaciones_por_sena": 30,
            "frames_por_grabacion": 30,
            "landmarks_por_frame": 63,
            "senas_incluidas": list(self.senas.keys())
        }


    def crear_directorios(self):
        # Crear directorio principal si no existe
        if not os.path.exists(self.directorio_base):
            os.makedirs(self.directorio_base)
            
        # Crear subdirectorios para cada seña
        for sena in self.senas.keys():
            directorio_sena = os.path.join(self.directorio_base, sena)
            if not os.path.exists(directorio_sena):
                os.makedirs(directorio_sena)

    def extraer_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar landmarks para visualización
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Extraer coordenadas
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        return frame, landmarks

    def grabar_sena(self, nombre_sena, num_grabacion):
        cap = cv2.VideoCapture(0)
        secuencia_landmarks = []
        contador_frames = 0
        grabando = False
        
        print(f"\nPreparando para grabar '{nombre_sena}' - Grabación #{num_grabacion}")
        print("Presiona 'ESPACIO' para comenzar a grabar")
        print("Presiona 'Q' para cancelar")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Agregar texto informativo al frame
            if not grabando:
                cv2.putText(frame, "Presiona ESPACIO para grabar", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"Grabando: Frame {contador_frames}/30", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            frame, landmarks = self.extraer_landmarks(frame)
            
            if grabando and landmarks:
                secuencia_landmarks.append(landmarks)
                contador_frames += 1
                
                if contador_frames >= 30:  # 30 frames por grabación
                    # Guardar secuencia
                    archivo = os.path.join(
                        self.directorio_base,
                        nombre_sena,
                        f"grabacion_{num_grabacion}.npy"
                    )
                    np.save(archivo, np.array(secuencia_landmarks))
                    print(f"\nGrabación guardada: {archivo}")
                    break
            
            cv2.imshow('Grabación de Señas', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Espacio para comenzar a grabar
                grabando = True
            elif key == ord('q'):  # Q para cancelar
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return contador_frames >= 30

    def recolectar_dataset(self):
        # Guardar metadata
        with open(os.path.join(self.directorio_base, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=4)
        
        # Recolectar datos para cada seña
        for sena, num_grabaciones in self.senas.items():
            print(f"\n=== Recolectando datos para: {sena} ===")
            
            grabacion_actual = 1
            while grabacion_actual <= num_grabaciones:
                print(f"\nGrabación {grabacion_actual} de {num_grabaciones}")
                
                if self.grabar_sena(sena, grabacion_actual):
                    grabacion_actual += 1
                
                # Preguntar si quiere continuar
                respuesta = input("\n¿Continuar con la siguiente grabación? (s/n): ")
                if respuesta.lower() != 's':
                    break
            
            print(f"\nCompletado: {sena}")

# Ejemplo de uso
if __name__ == "__main__":
    recolector = RecolectorDataset()
    recolector.recolectar_dataset()
