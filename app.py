from ultralytics import YOLO
import gradio as gr

# Carga el modelo (mismo directorio que app.py)
model = YOLO("best.pt")

def detect(frame):
    # frame llega como numpy array (RGB) desde la webcam del navegador
    results = model(frame, conf=0.7)
    annotated = results[0].plot()
    return annotated

demo = gr.Interface(
    fn=detect,
    inputs=gr.Image(source="webcam", streaming=True, type="numpy"),
    outputs=gr.Image(type="numpy"),
    live=True,
    title="YOLO – Webcam en tiempo real",
    description="Detección en vivo usando el modelo best.pt"
)

# IMPORTANTE para Codespaces / servidor remoto
demo.launch(server_name="0.0.0.0", server_port=7860)
