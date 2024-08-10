import gradio as gr
from sample_request import envoke_endpoint


interface = gr.Interface(
    fn=envoke_endpoint,
    inputs=gr.Image(),
    outputs=[gr.Image()],
    live=True,
    title="Face Enhancement with Real-ESRGAN",
    description="Upload an image and see it enhanced using Real-ESRGAN."
)


interface.launch(server_name="0.0.0.0", server_port=8080)#, share= True)