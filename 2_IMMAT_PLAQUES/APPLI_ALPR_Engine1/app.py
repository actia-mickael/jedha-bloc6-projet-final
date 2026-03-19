"""
Inside the ALPR Engine - Interactive Demo
Progressive visualization of license plate recognition pipeline.
"""

import gradio as gr
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.pipeline import ALPRPipeline
from utils.visualizer import (
    create_step_images, 
    create_analysis_report,
    format_ocr_result
)
from utils.error_gallery import ErrorGallery
from utils.video_processor import (
    create_annotated_video,
    process_gif,
    sample_video_frames,
    create_static_video
)
from utils.access_control import AccessController
from utils.database import DatabaseManager
import tempfile
import textwrap
import pandas as pd

# Create outputs directory and ensure it's writable
try:
    os.makedirs("outputs", exist_ok=True)
    test_file = os.path.join("outputs", ".write_test")
    with open(test_file, "w") as f:
        f.write("test")
    os.remove(test_file)
    print("✅ Outputs directory is writable")
except Exception as e:
    print(f"❌ Error with outputs directory: {e}")


print("🚀 Launching Inside the ALPR Engine...")

# Initialize pipeline and gallery
print("📦 Initializing ALPR Pipeline...")
pipeline = ALPRPipeline()

print("📦 Initializing Error Gallery...")
error_gallery = ErrorGallery()

print("📦 Initializing Access Controller...")
access_controller = AccessController()

print("📦 Initializing Database Manager...")
db = DatabaseManager()


def process_upload(image, conf_threshold=0.5):
    """
    Process uploaded image through ALPR pipeline.
    
    Args:
        image: Uploaded image
        conf_threshold: Detection confidence threshold
        
    Returns:
        Tuple of outputs for Gradio interface
    """
    if image is None:
        return None, None, None, None, "Please upload an image first.", None
    
    # Save temporary image
    temp_path = "/tmp/upload_temp.jpg"
    image.save(temp_path)
    
    # Process through pipeline
    results = pipeline.process_image(temp_path, conf_threshold)
    
    # Create step visualizations
    steps = create_step_images(results)
    
    # Extract images for each step
    step1_img = results['step1_raw']
    step2_img = results['step2_detection']
    step5_img = results['step5_final']
    
    # Get ROI if available
    step3_img = results['step3_roi'][0] if results['step3_roi'] else None
    
    # Create OCR results text
    ocr_text = ""
    if results['step4_ocr']:
        for i, ocr_result in enumerate(results['step4_ocr'], 1):
            ocr_text += format_ocr_result(
                ocr_result['text'], 
                ocr_result['confidence']
            )
            ocr_text += "\n\n"
    else:
        ocr_text = "❌ No plates detected"
    
    # Create analysis report
    report = create_analysis_report(results)
    
    # Access Control Check
    if results and results['step4_ocr']:
        best_plate = results['step4_ocr'][0]['text']
        authorized, message = access_controller.check_access(best_plate)
        
        status_color = "green" if authorized else "red"
        status_icon = "✅" if authorized else "⛔"
        
        # Log the attempt (Single Image)
        access_controller.log_attempt(best_plate, authorized)
        
        access_banner = f"""<div class="access-banner" style="background-color: {status_color}; color: white; text-align: center;">
            {status_icon} {message}
        </div>"""
        report = access_banner + "\n\n" + report
    else:
        report = """<div class="access-banner" style="background-color: #64748b; color: white; text-align: center;">
            ⚠️ NO PLATE DETECTED - ACCESS UNKNOWN
        </div>""" + "\n\n" + report
    
    # Handle Error Gallery
    gallery_update = gr.update()
    
    if not results['step4_ocr']:
        # No plate detected - Add to gallery
        print("⚠️ No plate detected - adding to error gallery")
        try:
            error_gallery.add_example(
                image_path=temp_path,
                issue="No Plate Detected",
                expected="Unknown",
                predicted="None",
                detection_conf=0.0,
                ocr_conf=0.0,
                analysis=f"Failed to detect plate. {report[0:50]}..." # Brief context
            )
            # Return updated markdown
            gallery_update = error_gallery.create_gallery_markdown()
        except Exception as e:
            print(f"Error updating gallery: {e}")

    # Cleanup
    os.remove(temp_path)
    
    return step1_img, step2_img, step3_img, ocr_text, step5_img, report, gallery_update


def process_video_upload(video, mode, conf_threshold=0.5, num_samples=10):
    """
    Process uploaded video through ALPR pipeline.
    
    Args:
        video: Uploaded video file
        mode: Processing mode ('sample', 'annotate', 'gif')
        conf_threshold: Detection confidence threshold
        num_samples: Number of frames to sample (for 'sample' mode)
        
    Returns:
        Tuple: (video_output_update, summary_text)
    """
    if video is None:
        return gr.update(visible=False, value=None), "Please upload a video first."
    
    try:
        if mode == "sample":
            # Sample frames mode
            results = sample_video_frames(pipeline, video, num_samples, conf_threshold)
            
            # Create summary
            summary = f"## 📊 Video Frame Sampling Results\n\n"
            summary += f"**Sampled Frames**: {len(results)}\n\n"
            
            total_detections = sum(len(r['metadata']['detections']) for r in results)
            summary += f"**Total Detections**: {total_detections}\n\n"
            
            summary += "### Frame Details:\n\n"
            
            # Access control check for sampled frames
            access_status_summary = []
            
            for i, result in enumerate(results, 1):
                frame_num = result['frame_number']
                timestamp = result['timestamp']
                num_det = len(result['metadata']['detections'])
                
                summary += f"**Frame {i}** (#{frame_num}, {timestamp:.2f}s): {num_det} plate(s)\n"
                
                if result['step4_ocr']:
                    for ocr in result['step4_ocr']:
                        if ocr['text']:
                            plate_text = ocr['text']
                            authorized, message = access_controller.check_access(plate_text)
                            access_status_summary.append(authorized)
                            summary += f"  - `{plate_text}` ({ocr['confidence']:.1%}) - {message}\n"
                summary += "\n"
            
            # Overall access banner for sample mode
            if access_status_summary:
                if any(access_status_summary):
                    access_banner = """<div class="access-banner" style="background-color: #00D9A3; color: #0F172A; text-align: center;">
                        ✅ ACCESS GRANTED (at least one plate on allowlist)
                    </div>"""
                else:
                    access_banner = """<div class="access-banner" style="background-color: #ef4444; color: white; text-align: center;">
                        ⛔ ACCESS DENIED (no plates on allowlist)
                    </div>"""
            else:
                access_banner = """<div class="access-banner" style="background-color: #64748b; color: white; text-align: center;">
                    ⚠️ NO PLATE DETECTED - ACCESS UNKNOWN
                </div>"""
            summary = access_banner + "\n\n" + summary
            
            # Create static video from first frame for preview
            if results:
                first_frame = results[0]['step5_final']
                output_path = f"outputs/sample_preview_{int(time.time())}.webm"
                create_static_video(first_frame, output_path, duration=3)
                output_path = os.path.abspath(output_path)
            else:
                output_path = None
            
            # Return: video visible with static preview
            return gr.update(value=output_path, visible=True), summary
            
        elif mode == "annotate":
            # Full video annotation - save to persistent location
            os.makedirs("outputs", exist_ok=True)
            output_path = f"outputs/alpr_annotated_{int(time.time())}.webm"
            
            stats = create_annotated_video(pipeline, video, output_path, conf_threshold, max_fps=10)
            
            summary = f"## 🎬 Annotated Video Created\n\n"
            summary += f"**Total Frames**: {stats['total_frames']}\n"
            summary += f"**Processed Frames**: {stats['processed_frames']}\n"
            summary += f"**Output FPS**: {stats['output_fps']:.1f}\n"
            summary += f"**Total Detections**: {stats['total_detections']}\n\n"
            
            # Add OCR results summary
            summary += "### 🎯 Detection Summary\n\n"
            if stats['total_detections'] > 0:
                summary += f"✅ Successfully detected {stats['total_detections']} license plates across {stats['processed_frames']} frames.\n\n"
                
                # Check access & Log Unique
                authorized_plates = []
                logged_plates = set()
                
                for plate_text in stats['detected_plates']:
                    authorized, _ = access_controller.check_access(plate_text)
                    if authorized:
                        authorized_plates.append(plate_text)
                    
                    if plate_text not in logged_plates:
                        access_controller.log_attempt(plate_text, authorized)
                        logged_plates.add(plate_text)
                
                if authorized_plates:
                    access_banner = f"""<div style="background-color: green; color: white; padding: 10px; border-radius: 5px; margin-bottom: 20px; text-align: center; font-weight: bold; font-size: 18px;">
                        ✅ ACCESS GRANTED (Detected: {', '.join(authorized_plates)})
                    </div>"""
                else:
                    access_banner = """<div style="background-color: red; color: white; padding: 10px; border-radius: 5px; margin-bottom: 20px; text-align: center; font-weight: bold; font-size: 18px;">
                        ⛔ ACCESS DENIED (No detected plates on allowlist)
                    </div>"""
                summary += access_banner + "\n\n"
                summary += "**Download the video** using the download button above to see all annotations!\n"
            else:
                summary += "⚠️ No license plates detected. Try lowering the confidence threshold.\n"
                summary += """<div style="background-color: gray; color: white; padding: 10px; border-radius: 5px; margin-bottom: 20px; text-align: center; font-weight: bold;">
                    ⚠️ NO PLATE DETECTED - ACCESS UNKNOWN
                </div>"""
                
            # Return: image hidden, video visible with value
            # Return: video visible with value
            return gr.update(value=os.path.abspath(output_path), visible=True), summary
            
        elif mode == "gif":
            # GIF processing - save to persistent location
            os.makedirs("outputs", exist_ok=True)
            output_path = f"outputs/alpr_annotated_{int(time.time())}.gif"
            
            stats = process_gif(pipeline, video, output_path, conf_threshold, max_frames=50)
            
            summary = f"## 🎞️ Annotated GIF Created\n\n"
            summary += f"**Total Frames**: {stats['total_frames']}\n"
            summary += f"**Processed Frames**: {stats['processed_frames']}\n"
            summary += f"**Total Detections**: {stats['total_detections']}\n\n"
            
            # Add OCR results summary
            summary += "### 🎯 Detection Summary\n\n"
            if stats['total_detections'] > 0:
                summary += f"✅ Successfully detected {stats['total_detections']} license plates across {stats['processed_frames']} frames.\n\n"
                
                # Check access & Log Unique
                authorized_plates = []
                logged_plates = set()
                
                for plate_text in stats['detected_plates']:
                    authorized, _ = access_controller.check_access(plate_text)
                    if authorized:
                        authorized_plates.append(plate_text)
                    
                    if plate_text not in logged_plates:
                        access_controller.log_attempt(plate_text, authorized)
                        logged_plates.add(plate_text)
                
                if authorized_plates:
                    access_banner = f"""<div style="background-color: green; color: white; padding: 10px; border-radius: 5px; margin-bottom: 20px; text-align: center; font-weight: bold; font-size: 18px;">
                        ✅ ACCESS GRANTED (Detected: {', '.join(authorized_plates)})
                    </div>"""
                else:
                    access_banner = """<div style="background-color: red; color: white; padding: 10px; border-radius: 5px; margin-bottom: 20px; text-align: center; font-weight: bold; font-size: 18px;">
                        ⛔ ACCESS DENIED (No detected plates on allowlist)
                    </div>"""
                summary += access_banner + "\n\n"
                summary += "**Download the GIF** to see the animated annotations!\n"
            else:
                summary += "⚠️ No license plates detected. Try lowering the confidence threshold.\n"
                summary += """<div style="background-color: gray; color: white; padding: 10px; border-radius: 5px; margin-bottom: 20px; text-align: center; font-weight: bold;">
                    ⚠️ NO PLATE DETECTED - ACCESS UNKNOWN
                </div>"""
            
            # Return: video visible with GIF
            return gr.update(value=os.path.abspath(output_path), visible=True), summary
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return gr.update(visible=False, value=None), f"❌ Error processing video: {str(e)}\n\n```\n{error_details}\n```"


def create_demo():
    """Create Gradio demo interface."""
    
    # Custom CSS for premium look
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    :root {
        color-scheme: light dark;
        --primary: #00D9A3;
        --primary-glow: rgba(0, 217, 163, 0.4);
        --secondary: #5B7FFF;
        --bg-dark: #0F172A;
        --card-bg: rgba(30, 41, 59, 0.6);
        --border-glass: rgba(255, 255, 255, 0.1);
        --text-main: #FFFFFF;
        --text-muted: rgba(255, 255, 255, 0.7);
        --shadow-main: rgba(0,0,0,0.3);
    }

    @media (prefers-color-scheme: light) {
        :root {
            --bg-dark: #F1F5F9;
            --card-bg: rgba(255, 255, 255, 0.9);
            --border-glass: rgba(0, 0, 0, 0.08);
            --text-main: #0F172A;
            --text-muted: #475569;
            --shadow-main: rgba(0,0,0,0.05);
        }
        
        .gradio-container {
            background: #F1F5F9 !important;
        }

        button[role="tab"] {
            background: rgba(0,0,0,0.05) !important;
            color: #475569 !important;
        }
        
        .upload-zone {
            background: rgba(0,0,0,0.02) !important;
        }
    }

    * {
        font-family: 'Inter', sans-serif;
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), 
                    background-color 0.3s ease, 
                    color 0.3s ease, 
                    box-shadow 0.3s ease;
    }
    
    .gradio-container {
        background: var(--bg-dark) !important;
        color: var(--text-main) !important;
        max-width: 1400px !important;
        margin: 0 auto !important;
        padding: 40px 20px !important;
    }

    .gradio-container p, 
    .gradio-container span, 
    .gradio-container li,
    .gradio-container label,
    .gradio-container strong,
    .gradio-container b,
    .gradio-container h1,
    .gradio-container h2,
    .gradio-container h3,
    .gradio-container table,
    .gradio-container td,
    .gradio-container th {
        color: var(--text-main) !important;
    }
    
    /* Specific Markdown/Prose targeting */
    .prose * {
        color: var(--text-main) !important;
    }

    /* DataFrame / Table Fixes */
    .gradio-container table {
        background-color: transparent !important;
    }
    
    .gradio-container tr, .gradio-container td, .gradio-container th {
        background-color: var(--card-bg) !important;
        border-color: var(--border-glass) !important;
    }

    /* Radio & Inputs visibility */
    .gradio-container input[type="radio"] + span {
        color: var(--text-main) !important;
    }

    @media (prefers-color-scheme: light) {
        :root {
            --bg-dark: #F1F5F9;
            --card-bg: #FFFFFF;
            --border-glass: #E2E8F0;
            --text-main: #1E293B;
            --text-muted: #64748B;
            --shadow-main: rgba(0,0,0,0.08);
        }
        
        .gradio-container {
            background: #F8FAFC !important;
        }

        button[role="tab"] {
            background: #E2E8F0 !important;
            color: #64748B !important;
        }
        
        .upload-zone {
            background: #F1F5F9 !important;
        }
        
        /* Force high contrast for bold/headers in light mode */
        .gradio-container strong, .gradio-container b, .gradio-container h1, .gradio-container h2, .gradio-container h3 {
            color: #0F172A !important;
        }
    }
    
    /* Typography */
    .title-main h1 {
        font-size: 48px !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #00D9A3, #5B7FFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
        margin-bottom: 8px !important;
    }
    
    .subtitle-main p {
        font-size: 18px !important;
        color: var(--text-muted) !important;
        letter-spacing: 0.5px;
        margin-bottom: 32px !important;
        font-weight: 500 !important;
    }

    /* Cards & Containers */
    .glass-card {
        background: var(--card-bg) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 8px 32px var(--shadow-main) !important;
        margin-bottom: 32px !important;
    }

    /* Upload Zone */
    .upload-zone {
        background: linear-gradient(135deg, rgba(91,127,255,0.05), rgba(0,217,163,0.05)) !important;
        border: 2px dashed rgba(0, 217, 163, 0.3) !important;
        border-radius: 16px !important;
        padding: 20px !important;
    }
    
    .upload-zone:hover {
        border-color: var(--primary) !important;
        background: linear-gradient(135deg, rgba(91,127,255,0.1), rgba(0,217,163,0.1)) !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px var(--primary-glow) !important;
    }

    /* Primary Buttons */
    .btn-hero {
        background: linear-gradient(135deg, #00D9A3, #00B386) !important;
        border: none !important;
        padding: 16px 48px !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        box-shadow: 0 8px 24px var(--primary-glow) !important;
        color: #0F172A !important;
    }
    
    .btn-hero:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 12px 32px rgba(0,217,163,0.6) !important;
        filter: brightness(1.1);
    }

    /* Tabs */
    button[role="tab"] {
        background: rgba(255,255,255,0.05) !important;
        backdrop-filter: blur(10px) !important;
        color: var(--text-muted) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 12px 12px 0 0 !important;
        margin-right: 4px !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
    }
    
    button[role="tab"].selected {
        background: linear-gradient(135deg, #00D9A3, #00B386) !important;
        color: #0F172A !important;
        box-shadow: 0 4px 12px rgba(0,217,163,0.3) !important;
        transform: translateY(-2px) !important;
        border: none !important;
    }
    
    button[role="tab"]:hover:not(.selected) {
        background: rgba(255,255,255,0.1) !important;
        color: var(--text-main) !important;
    }

    /* Slider */
    .gr-slider input[type="range"]::-webkit-slider-runnable-track {
        background: linear-gradient(to right, #00D9A3, #5B7FFF) !important;
        height: 8px !important;
        border-radius: 4px !important;
    }
    
    .gr-slider input[type="range"]::-webkit-slider-thumb {
        width: 20px !important;
        height: 20px !important;
        background: #FFFFFF !important;
        border: 2px solid var(--primary) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
        margin-top: -6px !important;
    }

    /* Headers / Labels */
    .section-title h2 {
        color: var(--primary) !important;
        font-weight: 700 !important;
        font-size: 24px !important;
        margin-bottom: 20px !important;
    }
    
    /* Result Banners */
    .access-banner {
        border-radius: 12px !important;
        padding: 16px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2) !important;
    }

    /* Input Text Color Fix */
    input, textarea, .gr-textbox {
        background-color: var(--card-bg) !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border-glass) !important;
    }

    /* Micro-interactions */
    .interactive-element:hover {
        transform: translateY(-2px);
        filter: brightness(1.1);
    }
    """
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
        
        # Header
        with gr.Column(elem_classes="title-main"):
            gr.Markdown("# 🚗 Inside the ALPR Engine")
        with gr.Column(elem_classes="subtitle-main"):
            gr.Markdown("### See how AI reads license plates, step by step")
            gr.Markdown("Upload an image, video, or GIF and watch the pipeline process it in real-time. This demo reveals the complete journey from raw input to recognized plate text.")
        
        with gr.Tabs():
            # IMAGE TAB
            with gr.Tab("📸 Image Processing"):
                with gr.Row():
                    with gr.Column(scale=3, elem_classes="glass-card"):
                        # Input section
                        gr.Markdown("## 📤 Input", elem_classes="section-title")
                        image_input = gr.Image(
                            type="pil",
                            label="Upload Image",
                            height=300,
                            elem_classes="upload-zone"
                        )
                        
                        conf_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.5,
                            step=0.05,
                            label="Detection Confidence Threshold",
                            info="Lower = more detections, Higher = more confident detections"
                        )
                        
                        process_btn = gr.Button("🔍 Process Image", variant="primary", size="lg", elem_classes="btn-hero")
                        
                        gr.Markdown("---")
                        
                    
                    with gr.Column(scale=7, elem_classes="glass-card"):
                        # Pipeline visualization
                        gr.Markdown("## 🔄 Pipeline Steps", elem_classes="section-title")
                        
                        with gr.Tabs():
                            with gr.Tab("1️⃣ Raw Input"):
                                step1_output = gr.Image(label="Original Image", height=300)
                                gr.Markdown("*The image as received, before any processing*")
                            
                            with gr.Tab("2️⃣ YOLOv8 Detection"):
                                step2_output = gr.Image(label="Detection Result", height=300)
                                gr.Markdown("*YOLOv8 identifies potential license plate regions*")
                            
                            with gr.Tab("3️⃣ ROI Extraction"):
                                step3_output = gr.Image(label="Extracted Plate", height=300)
                                gr.Markdown("*Cropped region of interest for OCR processing*")
                            
                            with gr.Tab("4️⃣ OCR Result"):
                                ocr_output = gr.Markdown(label="Recognized Text")
                    
                    with gr.Tab("5️⃣ Final Result"):
                        step5_output = gr.Image(label="Annotated Image", height=300)
                        gr.Markdown("*Complete result with detected text overlay*")
                
                # Analysis section for images
                with gr.Row():
                    with gr.Column(elem_classes="glass-card"):
                        gr.Markdown("## 📊 Detailed Analysis", elem_classes="section-title")
                        analysis_output = gr.Markdown()
            
            # VIDEO/GIF TAB
            with gr.Tab("🎬 Video/GIF Processing"):
                gr.Markdown("""
                ## Process Videos and Animated GIFs
                
                Upload a video or GIF to detect license plates across multiple frames. Choose from three processing modes:
                - **📊 Sample Frames**: Quick preview by sampling key frames
                - **🎬 Annotate Video**: Create fully annotated video (slower)
                - **🎞️ Process GIF**: Annotate animated GIF frame-by-frame
                """)
                
                with gr.Row():
                    with gr.Column(scale=3, elem_classes="glass-card"):
                        gr.Markdown("## 📤 Upload", elem_classes="section-title")
                        video_input = gr.Video(
                            label="Upload Video or GIF",
                            height=300,
                            elem_classes="upload-zone"
                        )
                        
                        video_mode = gr.Radio(
                            choices=[
                                ("📊 Sample Frames (Fast)", "sample"),
                                ("🎬 Annotate Full Video (Slow)", "annotate"),
                                ("🎞️ Process GIF", "gif")
                            ],
                            value="sample",
                            label="Processing Mode",
                            info="Sample mode is fastest for quick preview"
                        )
                        
                        video_conf_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.5,
                            step=0.05,
                            label="Detection Confidence Threshold"
                        )
                        
                        num_samples_slider = gr.Slider(
                            minimum=5,
                            maximum=30,
                            value=10,
                            step=5,
                            label="Number of Frames to Sample",
                            info="Only for Sample Frames mode",
                            visible=True
                        )
                        
                        gr.Markdown("---")
                        gr.Markdown("### 🎬 Try Demo Videos")
                        gr.Examples(
                            examples=[
                                ["video/Animation_de_voiture_entrant_dans_un_parking.mp4"],
                                ["video/Voitures_rapides_doublant_sur_autoroute.mp4"]
                            ],
                            inputs=video_input,
                            label="Click to load demo videos"
                        )
                        
                        process_video_btn = gr.Button("🎬 Process Video/GIF", variant="primary", size="lg", elem_classes="btn-hero")
                        
                        gr.Markdown("""
                        ---
                        ### ⏱️ Processing Times
                        - **Sample**: ~5-10 seconds
                        - **Full Video**: ~1-5 minutes (depends on length)
                        - **GIF**: ~30-60 seconds
                        """)
                    
                    with gr.Column(scale=7, elem_classes="glass-card"):
                        gr.Markdown("## 📹 Results", elem_classes="section-title")
                        
                        # Single video component handles all outputs (sample preview, full video, GIF)
                        # Always visible to avoid layout shifts/errors
                        video_file_output = gr.Video(label="Processed Result", height=400, interactive=False)
                        video_summary = gr.Markdown(label="Processing Summary")
                        
                        gr.Markdown("""
                        ### 💡 Tips
                        - For long videos, use **Sample Frames** mode first
                        - **Annotate Video** creates a downloadable MP4
                        - **GIF mode** preserves animation with annotations
                        - Lower confidence threshold to detect more plates
                        """)
            
            # SETTINGS TAB
            
            with gr.Tab("⚙️ Settings"):
                with gr.Column(elem_classes="glass-card"):
                    gr.Markdown("## 🔑 Access Control Allowlist", elem_classes="section-title")
                    gr.Markdown("""
                    **Lecture Seule** : Cette liste est synchronisée automatiquement avec la base de données.
                    Pour modifier les accès, utilisez l'onglet **Administration Clients**.
                    """)
                    
                    # Fetch current allowlist from DB safely
                    def get_current_allowlist():
                        try:
                            access_controller.sync_from_database()
                            return access_controller.get_list_as_text()
                        except:
                            return "Erreur de synchronisation"

                    auth_input = gr.TextArea(
                        label="Plaques Autorisées (DB)",
                        value=get_current_allowlist,
                        lines=10,
                        interactive=False
                    )
                    
                    refresh_allowlist_btn = gr.Button("🔄 Actualiser la liste")
                    
                    refresh_allowlist_btn.click(
                        fn=get_current_allowlist,
                        inputs=[],
                        outputs=[auth_input]
                    )
                    
                    # Dummy component to keep layout specific
                    current_allowlist_display = gr.State()
                
                with gr.Column(elem_classes="glass-card"):
                    gr.Markdown("## 🤖 Detection Model Selection", elem_classes="section-title")
                    gr.Markdown("""
                    Choose which YOLOv8 model to use for license plate detection.
                    **Note:** Changing models will reload the detection engine (~2-3 seconds).
                    """)
                
                # Get available models
                available_models = pipeline.get_available_models()
                current_model = os.path.basename(pipeline.model_path)
                
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    value=current_model if current_model in available_models else available_models[0],
                    label="Select Model",
                    info="Models from the models/ directory"
                )
                
                reload_model_btn = gr.Button("🔄 Reload Model", variant="secondary")
                
                model_status = gr.Textbox(
                    label="Model Status",
                    value=f"✅ Currently loaded: {current_model}",
                    interactive=False,
                    lines=2
                )
            
            # ADMINISTRATION TAB
            with gr.Tab("👥 Administration Clients"):
                with gr.Row():
                    # KPIs
                    kpi_total = gr.Number(label="Total Résidents", value=0, interactive=False)
                    kpi_active = gr.Number(label="Accès Activé", value=0, interactive=False)
                    kpi_blocked = gr.Number(label="Bloqués", value=0, interactive=False)
                    kpi_subs = gr.Number(label="Abonnés", value=0, interactive=False)
                
                with gr.Row():
                    with gr.Column(scale=3, elem_classes="glass-card"):
                        gr.Markdown("### 🔍 Recherche & Filtres")
                        search_box = gr.Textbox(label="Rechercher (Plaque, Nom, Prénom)", placeholder="Tapez pour filtrer...")
                        with gr.Row():
                            search_btn = gr.Button("🔎 Rechercher", variant="primary")
                            reset_search_btn = gr.Button("🔄 Réinitialiser")
                        
                        gr.Markdown("---")
                        gr.Markdown("### ➕ Ajouter un Résident")
                        with gr.Group():
                            add_plaque = gr.Textbox(label="Plaque (ex: AA-123-BB)")
                            with gr.Row():
                                add_nom = gr.Textbox(label="Nom")
                                add_prenom = gr.Textbox(label="Prénom")
                            with gr.Row():
                                add_age = gr.Number(label="Âge", value=30)
                                add_tel = gr.Textbox(label="Téléphone")
                            add_adresse = gr.Textbox(label="Adresse")
                            with gr.Row():
                                add_ville = gr.Textbox(label="Ville")
                                add_cp = gr.Textbox(label="Code Postal")
                            with gr.Row():
                                add_abo = gr.Radio(["oui", "non"], label="Abonnement", value="non")
                                add_acces = gr.Radio(["oui", "non"], label="Accès Autorisé", value="non")
                            
                            add_btn = gr.Button("Ajouter Résident", variant="primary")
                            add_status = gr.Markdown()

                    with gr.Column(scale=7, elem_classes="glass-card"):
                        gr.Markdown("### 📋 Liste des Résidents")
                        gr.Markdown("*Cochez la case 'Sélection' pour supprimer ou modifier l'accès. Double-cliquez pour éditer les valeurs.*")
                        
                        residents_table = gr.DataFrame(
                            headers=["Sélection", "id", "plaque", "nom", "prenom", "age", "telephone", "adresse", "ville", "code_postal", "abonnement", "acces"],
                            datatype=["bool", "number", "str", "str", "str", "number", "str", "str", "str", "str", "str", "str"],
                            interactive=True, 
                            label="Résidents",
                            col_count=(12, "fixed")
                        )
                        
                        with gr.Row():
                            refresh_table_btn = gr.Button("🔄 Actualiser Tableau")
                            delete_btn = gr.Button("🗑️ Supprimer Cochés", variant="stop")
                            toggle_access_btn = gr.Button("🔄 Toggle Accès (Cochés)")
                            save_changes_btn = gr.Button("💾 Sauvegarder Modifications", variant="primary")
                        
                        db_status_msg = gr.Markdown()
                
                with gr.Row(elem_classes="glass-card"):
                    with gr.Column():
                        gr.Markdown("### 📜 Logs de Détection (Backend)")
                        logs_table = gr.DataFrame(
                            headers=["id", "plaque", "timestamp", "resultat", "normalized"],
                            interactive=False,
                            label="Historique d'accès"
                        )
                        refresh_logs_btn = gr.Button("🔄 Actualiser Logs")

                # Admin Logic
                def load_residents(query=""):
                    try:
                        stats = db.get_statistics()
                        residents = db.search_residents(query)
                        # Add False for checkbox column at index 0
                        data = [[False, r['id'], r['plaque'], r['nom'], r['prenom'], r['age'], r['telephone'], 
                                r['adresse'], r['ville'], r['code_postal'], r['abonnement'], r['acces']] 
                                for r in residents]
                        
                        return (
                            data, 
                            stats['total'], 
                            stats['active'], 
                            stats['blocked'],
                            stats['subscribed']
                        )
                    except Exception as e:
                        print(f"Error loading residents: {e}")
                        return [], 0, 0, 0, 0

                def add_resident_handler(plaque, nom, prenom, age, tel, adresse, ville, cp, abo, acces):
                    if not plaque:
                        return "❌ La plaque est obligatoire", gr.update()
                    
                    data = {
                        'plaque': plaque, 'nom': nom, 'prenom': prenom, 'age': age,
                        'telephone': tel, 'adresse': adresse, 'ville': ville,
                        'code_postal': cp, 'abonnement': abo, 'acces': acces
                    }
                    
                    success, msg = db.add_resident(data)
                    if success:
                        if acces == 'oui':
                            access_controller.sync_from_database()
                        
                        # Refresh table
                        residents_data, t, a, b, s = load_residents()
                        return msg, residents_data, t, a, b, s
                    
                    return msg, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

                def delete_checked_handler(current_data):
                    try:
                        # Iterate through dataframe to find checked rows
                        # Gradio DataFrame comes as a pandas DF or list of lists depending on config
                        # With datatype set, it usually comes as DataFrame if interactive=True? 
                        # Let's handle pandas DataFrame which is standard for interactive tables
                        
                        processed_count = 0
                        last_msg = ""
                        
                        if hasattr(current_data, 'iloc'):
                            # Pandas DataFrame
                            # Column 0 is "Sélection"
                            for index, row in current_data.iterrows():
                                if row.iloc[0] == True: # If checked
                                    resident_id = int(row.iloc[1]) # ID is col 1
                                    success, msg = db.delete_resident(resident_id)
                                    if success: processed_count += 1
                                    last_msg = msg
                        else:
                            # List of lists
                            for row in current_data:
                                if row[0] == True:
                                    resident_id = int(row[1])
                                    success, msg = db.delete_resident(resident_id)
                                    if success: processed_count += 1
                                    last_msg = msg
                        
                        if processed_count > 0:
                            access_controller.sync_from_database()
                            residents_data, t, a, b, s = load_residents()
                            return f"✅ {processed_count} résidents supprimés.", residents_data, t, a, b, s
                        elif processed_count == 0:
                            return "⚠️ Aucune ligne cochée", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                        
                        return last_msg, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        return f"❌ Erreur lors de la suppression: {e}", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

                def toggle_access_checked_handler(current_data):
                    try:
                        processed_count = 0
                        
                        if hasattr(current_data, 'iloc'):
                            for index, row in current_data.iterrows():
                                if row.iloc[0] == True:
                                    resident_id = int(row.iloc[1])
                                    success, _, _ = db.toggle_access(resident_id)
                                    if success: processed_count += 1
                        else:
                            for row in current_data:
                                if row[0] == True:
                                    resident_id = int(row[1])
                                    success, _, _ = db.toggle_access(resident_id)
                                    if success: processed_count += 1
                        
                        if processed_count > 0:
                            access_controller.sync_from_database()
                            residents_data, t, a, b, s = load_residents()
                            return f"✅ Accès modifié pour {processed_count} résidents.", residents_data, t, a, b, s
                        
                        return "⚠️ Aucune ligne cochée", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                    except Exception as e:
                        return f"❌ Erreur: {e}", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

                def save_changes_handler(current_data):
                    """Save edited dataframe back to database"""
                    try:
                        updated_count = 0
                        
                        if hasattr(current_data, 'iloc'):
                             for index, row in current_data.iterrows():
                                resident_id = int(row.iloc[1])
                                data = {
                                    'plaque': str(row.iloc[2]),
                                    'nom': str(row.iloc[3]),
                                    'prenom': str(row.iloc[4]),
                                    'age': int(row.iloc[5]),
                                    'telephone': str(row.iloc[6]),
                                    'adresse': str(row.iloc[7]),
                                    'ville': str(row.iloc[8]),
                                    'code_postal': str(row.iloc[9]),
                                    'abonnement': str(row.iloc[10]),
                                    'acces': str(row.iloc[11])
                                }
                                success, _ = db.update_resident(resident_id, data)
                                if success: updated_count += 1
                        
                        access_controller.sync_from_database()
                        residents_data, t, a, b, s = load_residents()
                        return f"✅ {updated_count} lignes sauvegardées.", residents_data, t, a, b, s
                        
                    except Exception as e:
                        return f"❌ Erreur sauvegarde: {e}", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

                def load_logs():
                    logs = db.get_logs(limit=50)
                    data = [[l['id'], l['plaque'], l['timestamp'], l['resultat'], l['normalized_plate']] for l in logs]
                    return data

                # Bindings
                refresh_table_btn.click(
                    fn=load_residents,
                    inputs=[search_box],
                    outputs=[residents_table, kpi_total, kpi_active, kpi_blocked, kpi_subs]
                )
                
                search_btn.click(
                    fn=load_residents,
                    inputs=[search_box],
                    outputs=[residents_table, kpi_total, kpi_active, kpi_blocked, kpi_subs]
                )
                
                reset_search_btn.click(
                    fn=lambda: ("", *load_residents("")),
                    inputs=[],
                    outputs=[search_box, residents_table, kpi_total, kpi_active, kpi_blocked, kpi_subs]
                )
                
                add_btn.click(
                    fn=add_resident_handler,
                    inputs=[add_plaque, add_nom, add_prenom, add_age, add_tel, add_adresse, add_ville, add_cp, add_abo, add_acces],
                    outputs=[add_status, residents_table, kpi_total, kpi_active, kpi_blocked, kpi_subs]
                )

                delete_btn.click(
                    fn=delete_checked_handler,
                    inputs=[residents_table],
                    outputs=[db_status_msg, residents_table, kpi_total, kpi_active, kpi_blocked, kpi_subs]
                )

                toggle_access_btn.click(
                     fn=toggle_access_checked_handler,
                     inputs=[residents_table],
                     outputs=[db_status_msg, residents_table, kpi_total, kpi_active, kpi_blocked, kpi_subs]
                )
                
                save_changes_btn.click(
                    fn=save_changes_handler,
                    inputs=[residents_table],
                    outputs=[db_status_msg, residents_table, kpi_total, kpi_active, kpi_blocked, kpi_subs]
                )

                refresh_logs_btn.click(
                    fn=load_logs,
                    inputs=[],
                    outputs=[logs_table]
                )
                
                # Initial Load
                demo.load(
                    fn=load_residents, 
                    inputs=[], 
                    outputs=[residents_table, kpi_total, kpi_active, kpi_blocked, kpi_subs]
                )
            
            # HISTORY TAB (Top-level)
            def get_history_df():
                try:
                    if os.path.exists("outputs/access_log.csv"):
                        return pd.read_csv("outputs/access_log.csv")
                    return pd.DataFrame(columns=["Timestamp", "Plate", "Status", "Normalized"])
                except Exception as e:
                    return pd.DataFrame({"Error": [str(e)]})

            with gr.Tab("📜 History"):
                with gr.Column(elem_classes="glass-card"):
                    gr.Markdown("## 📋 Access Log History", elem_classes="section-title")
                    refresh_btn = gr.Button("🔄 Refresh Log", variant="secondary")
                history_table = gr.DataFrame(
                    value=get_history_df,
                    headers=["Timestamp", "Plate", "Status", "Normalized"],
                    interactive=False,
                    row_count=10,
                    wrap=True
                )
                
                refresh_btn.click(fn=get_history_df, inputs=[], outputs=history_table)
        
        # Error gallery section (shared across tabs)
        with gr.Row():
            with gr.Column(elem_classes="glass-card"):
                error_gallery_component = gr.Markdown(error_gallery.create_gallery_markdown())
                with gr.Row():
                    refresh_gallery_btn = gr.Button("🔄 Refresh Gallery", variant="secondary")
                    clear_gallery_btn = gr.Button("🗑️ Clear Gallery", variant="stop")
        
        # Gallery Logic
        def refresh_gallery():
            return error_gallery.create_gallery_markdown()
            
        def clear_gallery_handler():
            success, msg = error_gallery.clear_gallery()
            return error_gallery.create_gallery_markdown()

        refresh_gallery_btn.click(fn=refresh_gallery, inputs=[], outputs=[error_gallery_component])
        clear_gallery_btn.click(fn=clear_gallery_handler, inputs=[], outputs=[error_gallery_component])
        
        # Footer
        gr.Markdown("""
        ---
        
        ### 🧠 About This Demo
        
        This demo showcases a complete ALPR (Automatic License Plate Recognition) pipeline:
        - **Detection**: YOLOv8 nano fine-tuned on UC3M-LP dataset
        - **OCR**: fast-plate-ocr with global model (40+ countries)
        - **Dataset**: 24,238 images with 49% in challenging conditions
        
        **Why show the process?**  
        Understanding *how* the AI works is as important as *what* it produces. This transparency builds trust and reveals both capabilities and limitations.
        
        ---
        
        Built with ❤️ using Gradio | [View on GitHub](#) | [Deploy on HF Spaces](#)
        """)
        
        # Connect processing function
        process_btn.click(
            fn=process_upload,
            inputs=[image_input, conf_slider],
            outputs=[
                step1_output,
                step2_output,
                step3_output,
                ocr_output,
                step5_output,
                analysis_output,
                error_gallery_component
            ]
        )
        
        # Connect Allowlist Update
        auth_input.change(
            fn=access_controller.update,
            inputs=auth_input,
            outputs=current_allowlist_display
        )
        
        # Connect Model Reload
        def reload_model_handler(model_name):
            """Handler for model reload button."""
            success, message = pipeline.reload_model(model_name)
            return message
        
        reload_model_btn.click(
            fn=reload_model_handler,
            inputs=[model_dropdown],
            outputs=[model_status]
        )
        
                # Connect video processing function
        process_video_btn.click(
            fn=process_video_upload,
            inputs=[video_input, video_mode, video_conf_slider, num_samples_slider],
            outputs=[video_file_output, video_summary]
        )

    return demo


if __name__ == "__main__":
    print("🚀 Building Gradio Interface...")
    demo = create_demo()
    
    print("✨ Starting Gradio Server on port 7860...")
    
    # Ensure assets directory exists for allowed_paths
    os.makedirs("assets", exist_ok=True)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        max_file_size="200MB",
        allowed_paths=["."] # Allow access to current directory and subdirectories (like assets)
    )
    print("🛑 Gradio Server has stopped.")
