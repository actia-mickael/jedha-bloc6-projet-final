---
title: ALPR Parking Demo
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🚗 Inside the ALPR Engine - Interactive Demo

An interactive demonstration of Automatic License Plate Recognition (ALPR) that reveals the complete AI pipeline, step by step.

## 🎯 Concept

Unlike traditional demos that only show final results, **Inside the ALPR Engine** provides a transparent, educational view of how AI processes license plates:

1. **Raw Input** - Original image
2. **YOLOv8 Detection** - Plate localization with bounding boxes
3. **ROI Extraction** - Cropped plate regions
4. **OCR Processing** - Character recognition
5. **Final Result** - Annotated output with confidence scores

## ✨ Features

- 📊 **Progressive Visualization**: See each pipeline step with smooth transitions
- 🎨 **Premium Dark UI**: Modern glassmorphism design with custom styling
- 📈 **Detailed Analysis**: Confidence scores, lighting conditions, and image quality metrics
- ⚠️ **Error Gallery**: Curated examples of failure cases with explanations
- 🎥 **Multi-format Support**: Images, videos, and GIFs (coming soon)

## 🛠️ Technical Stack

- **Detection**: YOLOv8 nano (fine-tuned on UC3M-LP dataset)
- **OCR**: fast-plate-ocr (global model, 40+ countries)
- **Interface**: Gradio 4.0+ with custom CSS
- **Dataset**: 24,238 images (49% challenging conditions)

## 🚀 Quick Start

### Local Installation

```bash
# Navigate to demo directory
cd demo

# Install dependencies
pip install -r requirements.txt

# Run the demo
python app.py
```

The demo will launch at `http://localhost:7860`

### Hugging Face Spaces Deployment

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Select **Gradio** as the SDK
3. Upload all files from the `demo/` directory
4. Add your trained model to `models/best.pt`
5. The Space will automatically build and deploy

### Deployment Method 2: Docker SDK (Recommended)

For the best stability and control over dependencies (especially for video processing), use the provided `Dockerfile`. This setup is already optimized for Hugging Face Spaces (permissions, user 1000, libraries).

1. Create a new Space and select **Docker** as the SDK.
2. Upload the entire `demo/` folder content.
3. The Space will automatically build using the `Dockerfile`.

### Alternative: Use the HF CLI

```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create and push to Space
huggingface-cli repo create --type space --space_sdk gradio your-space-name
cd demo
git init
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/your-space-name
git add .
git commit -m "Initial commit"
git push space main
```

## 📁 Project Structure

```
demo/
├── app.py                      # Main Gradio application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── utils/
│   ├── __init__.py
│   ├── pipeline.py            # ALPR processing pipeline
│   ├── visualizer.py          # Visualization utilities
│   └── error_gallery.py       # Error examples management
├── assets/
│   ├── style.css              # Custom CSS (optional)
│   └── error_examples/        # Failure case gallery
│       └── annotations.json   # Error annotations
└── models/
    └── best.pt                # Trained YOLOv8 model
```

## 🎨 Customization

### Adjust Detection Threshold

Use the slider in the interface to control detection sensitivity:
- **Lower (0.1-0.4)**: More detections, may include false positives
- **Medium (0.5-0.7)**: Balanced (recommended)
- **Higher (0.8-1.0)**: Only high-confidence detections

### Add Custom Examples

Edit `app.py` to add example images:

```python
gr.Examples(
    examples=[
        ["path/to/example1.jpg"],
        ["path/to/example2.jpg"],
    ],
    inputs=image_input,
)
```

### Modify Styling

Edit the `custom_css` variable in `app.py` to change colors, fonts, and layout.

## 📊 Understanding the Analysis

The demo provides detailed metrics:

- **Lighting Conditions**: Estimated from image brightness
  - 🌙 Night / Very low light (< 60)
  - 🌆 Low light (60-100)
  - ☁️ Medium light (100-160)
  - ☀️ Daylight (> 160)

- **Confidence Scores**:
  - ✔️ High (> 85%)
  - ⚠️ Medium (60-85%)
  - ❌ Low (< 60%)

- **Image Quality**: Blur detection using Laplacian variance

## 🎯 Error Gallery

The error gallery showcases real failure cases to demonstrate:
- Understanding of model limitations
- Transparency about edge cases
- Continuous improvement opportunities

Common failure modes:
- Extreme viewing angles
- Heavy dirt or damage
- Low light + reflections
- Partial occlusion
- Multiple plates in frame

## 🔧 Troubleshooting

### Model Not Found

If you see "⚠️ Trained model not found", ensure:
1. Your trained model is at `models/best.pt`
2. Or update the path in `utils/pipeline.py`

### CUDA/GPU Issues

The demo works on CPU by default. For GPU acceleration:
```python
# In utils/pipeline.py, modify YOLO initialization:
self.yolo_model = YOLO(self.model_path).to('cuda')
```

### Gradio Version Conflicts

Ensure you're using Gradio 4.0+:
```bash
pip install --upgrade gradio
```

## 📝 License

This demo is part of the ALPR project using the UC3M-LP dataset (CC BY 4.0).

## 🙏 Credits

- **Dataset**: UC3M-LP via Roboflow Universe
- **Detection**: Ultralytics YOLOv8
- **OCR**: fast-plate-ocr
- **Interface**: Gradio

## 🚀 Next Steps

- [ ] Add video processing support
- [ ] Implement GIF animation support
- [ ] Add batch processing mode
- [ ] Create API endpoint for integration
- [ ] Add real-time webcam processing

---

**Built with ❤️ for transparency in AI**

*"Show the journey, not just the destination"*
