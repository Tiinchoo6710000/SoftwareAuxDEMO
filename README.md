 ### ENGLISH
# Intelligent Monitoring System (Computer Vision Demonstration)

This repository contains a simplified demonstration of a larger Intelligent Monitoring System based on Computer Vision and Artificial Intelligence.

The demonstration showcases real-time video analysis capable of detecting critical situations such as:

• Person lying on the ground
• Presence in restricted areas
• Detection of unknown persons (face recognition)

This project represents the main detection engine of a broader system designed to assist healthcare personnel in hospitals, clinics, and nursing homes.

---

# Demo Features

Real-time video processing with Python and OpenCV

Person detection using YOLOv8 pose models

Fall detection using body posture analysis

Restricted area monitoring using ROI (Region of Interest)

Face recognition for authorized personnel identification

Real-time alert activation system

Basic multithreaded pipeline for video capture and AI inference

---

# Detection Modes

The demo includes two monitoring modes:

1️⃣ **Person on the ground detection**

Detects when a person is lying on the ground and remains motionless for a period of time.

2️⃣ **Restricted area detection**

Detects unknown individuals entering predefined restricted areas.

---

# System Structure

Camera Input (IP Camera/Webcam)

↓
Frame Capture (OpenCV)

↓
Pose Detection with YOLOv8

↓
Tracking System

↓
Fall Detection Logic

↓
Face Reconigtion

↓
Region of Interest (ROI) Validation

↓
Alert Activation System

---

# Technologies Used

## Backend

Python
OpenCV
YOLOv8 (Ultralytics)
NumPy
Torch
Face Recognition

## Computer Vision Techniques

Real-time Video Processing
Human Pose Estimation using YOLOv8 Pose
Fall Detection Logic
Object Tracking
Region of Interest (ROI) Filtering
Face Recognition

## U S A G E  

  Clone the repository

git clone https://github.com/Tiinchoo6710000/SoftwareAuxDEMO

Install dependencies

pip install -r requirements.txt


(To upload faces, a minimum of 10 photos of the face from different angles are required and must be added to the rostros_registrados folder. *NOT OBLIGATORY)


A) Configure and place the IP camera.

B) Run the system from main.py.

C) The system will prompt you to select the detection mode:

1 - Person on the ground
2 - Restricted area

D) Next, you can draw a Region of Interest (ROI) directly on the video stream.

*IF YOU SELECT PERSON DETECTION ON THE FLOOR, ONLY MARK THE FLOOR PERIMETER AS ROI

Press **ENTER** to confirm the ROI.

--

# Demo Interface

The system displays:

• Live video stream
• Detected people
• ROI boundaries
• Detection alerts
• Monitoring status panel

# Author

Martin Nieto
Software Developer


-----------------------------------------------------------


  ## ESPAÑOL
# Sistema de Monitoreo Inteligente (Demostración de Visión por Computadora)

Este repositorio contiene una demostración simplificada de un Sistema de Monitoreo Inteligente más amplio, basado en Visión por Computadora e Inteligencia Artificial.

La demostración muestra el análisis de video en tiempo real capaz de detectar situaciones críticas como:

• Persona tendida en el suelo
• Presencia en áreas restringidas
• Detección de personas desconocidas (reconocimiento facial)

Este proyecto representa el motor de detección principal de un sistema más amplio diseñado para asistir al personal sanitario en hospitales, clínicas y residencias de ancianos.

---

# Funcionalidades de la demo

Procesamiento de vídeo en tiempo real con Python y OpenCV

Detección de personas mediante modelos de pose YOLOv8

Detección de caídas mediante análisis de postura corporal mediante Yolo pose

Monitorización de áreas restringidas mediante ROI (Región de Interés)

Reconocimiento facial para la identificación de personal autorizado 

Sistema de activación de alertas en tiempo real

Pipeline multihilo básico para captura de vídeo e inferencia de IA

---

# Modos de detección

La demo incluye dos modos de monitorización:

1️⃣ **Detección de personas en el suelo**

Detecta cuando una persona está tumbada en el suelo y permanece inmóvil durante un tiempo.

2️⃣ **Detección de áreas restringidas**

Detecta a personas desconocidas que entran en áreas restringidas predefinidas.

---

# Estructura del sistema

Entrada de cámara (cámara IP/webcam)

↓
Captura de fotogramas (OpenCV)

↓
Detección de pose con YOLOv8

↓
Sistema de seguimiento

↓
Lógica de detección de caídas

↓
Reconocimiento facial

↓
Validación de la región de interés (ROI)

↓
Sistema de activación de alertas

---

# Tecnologías utilizadas

## Backend

Python
OpenCV
YOLOv8 (Ultralytics)
NumPy
Torch
Reconocimiento facial

## Técnicas de visión artificial

Procesamiento de vídeo en tiempo real
Estimación de la pose humana
Lógica de detección de caídas
Seguimiento de objetos
Filtrado de la región de interés (ROI)
Reconocimiento facial

    
##  M A N U A L  
^^^^^^^^^^^^^^^^^^^^^^^^ 

  Clonar el repositorio

git clone https://github.com/Tiinchoo6710000/SoftwareAuxDEMO

  Instalar dependecias

pip install -r requirements.txt


(Para subir rostros, se requieren un mínimo de 10 fotos del rostro de diferentes ángulos y deben agregarse a la carpeta rostros_registrados. *NO OBLIGATORIO)

A) Configure y coloque la cámara IP.

B) Ejecute el sistema desde main.py.

C) El sistema le pedirá que seleccione el modo de detección:

1 - Persona en el suelo
2 - Área restringida

D) A continuación, puede dibujar una Región de Interés (ROI) directamente en la transmisión de video.
   *EN CASO DE SELECCIONAR DETECCION PERSONA EN EL SUELO MARCAR SOLAMENTE EL PERIMETRO DEL PISO COMO ROI

Presione **ENTER** para confirmar la ROI.

--

# Interfaz de demostración

El sistema muestra:

• Transmisión de video en vivo
• Personas detectadas
• Límites de la ROI
• Alertas de detección
• Panel de estado de monitoreo

# Autor

Martin Nieto
Desarrollador de software


