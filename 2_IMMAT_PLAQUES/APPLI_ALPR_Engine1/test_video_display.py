"""
Test simple pour vérifier si Gradio peut afficher la vidéo.
"""

import gradio as gr

def test_video():
    # Retourne le chemin de la vidéo de test
    return "/home/phili/datascience/projet plaque/demo/test_output_video.mp4"

with gr.Blocks() as demo:
    gr.Markdown("# Test Vidéo ALPR")
    
    btn = gr.Button("Charger la vidéo")
    video_out = gr.Video(label="Vidéo annotée")
    
    btn.click(fn=test_video, outputs=video_out)

if __name__ == "__main__":
    demo.launch(server_port=7861, share=False)
