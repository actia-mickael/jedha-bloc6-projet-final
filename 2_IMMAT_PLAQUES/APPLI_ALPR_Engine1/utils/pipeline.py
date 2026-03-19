"""
ALPR Pipeline wrapper for demo application.
Provides step-by-step processing with intermediate results.
"""

import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer
from .custom_yolo import SimpleYOLO, load_custom_model


class ALPRPipeline:
    """ALPR Pipeline with step-by-step visualization support."""
    
    def __init__(self, model_path=None):
        """
        Initialize ALPR pipeline.
        
        Args:
            model_path: Path to YOLOv8 model. If None, searches for best.pt
        """
        # Find model path
        if model_path is None:
            model_path = self._find_model()
        
        self.model_path = model_path
        self.yolo_model = None
        self.ocr_model = None
        self.is_custom_model = False
        self._load_models()
    
    def _find_model(self):
        """Find the trained YOLOv8 model."""
        # Search in common locations
        search_paths = [
            'runs/detect/LP_roboflow/weights/best.pt',
            '../runs/detect/LP_roboflow/weights/best.pt',
            'models/best.pt',
            'demo/models/best.pt'
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        # If not found, use base YOLOv8n (will need to be fine-tuned)
        print("⚠️ Trained model not found, using yolov8n.pt")
        return 'yolov8n.pt'
    
    def _load_models(self):
        """Load YOLOv8 and OCR models."""
        print(f"📦 Loading YOLOv8 from {self.model_path}...")
        
        model_name = os.path.basename(self.model_path)
        if model_name == "modelemaison.pt" or model_name == "YOLO_From_Scratch_LicensePlatev2.pt":
            print(f"🏠 Detected custom model '{model_name}', using SimpleYOLO architecture.")
            self.yolo_model = load_custom_model(self.model_path)
            self.is_custom_model = True
        else:
            self.yolo_model = YOLO(self.model_path)
            self.is_custom_model = False
            
        print("📦 Loading fast-plate-ocr...")
        self.ocr_model = LicensePlateRecognizer('global-plates-mobile-vit-v2-model')
        
        print("✅ Models loaded successfully!")
    
    def reload_model(self, model_name):
        """
        Reload YOLOv8 model from specified filename.
        
        Args:
            model_name: Filename of model in models/ directory (e.g., 'best.pt')
            
        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            # Construct full path
            if model_name.startswith('models/'):
                model_path = model_name
            else:
                model_path = f"models/{model_name}"
            
            # Check if file exists
            if not os.path.exists(model_path):
                return False, f"❌ Model not found: {model_name}"
            
            # Load new model FIRST (before deleting old one)
            print(f"📦 Loading new model: {model_name}")
            is_custom = (os.path.basename(model_path) in ["modelemaison.pt", "YOLO_From_Scratch_LicensePlatev2.pt"])
            
            if is_custom:
                print("🏠 Using custom SimpleYOLO loader...")
                new_model = load_custom_model(model_path)
            else:
                new_model = YOLO(model_path)
            
            # Only if successful, unload old model and replace
            print(f"🔄 Unloading old model: {os.path.basename(self.model_path)}")
            old_model = self.yolo_model
            self.yolo_model = new_model
            self.model_path = model_path
            self.is_custom_model = is_custom
            
            # Clean up old model
            del old_model
            import gc
            gc.collect()
            
            return True, f"✅ Model '{model_name}' loaded successfully!"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"❌ Error loading model: {str(e)}"
    
    @staticmethod
    def get_available_models():
        """
        Get list of available .pt models in models/ directory.
        
        Returns:
            list: List of model filenames
        """
        models_dir = "models"
        if not os.path.exists(models_dir):
            return ["best.pt"]  # Fallback
        
        models = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
        return sorted(models) if models else ["best.pt"]
    
    def process_image(self, image_path, conf_threshold=0.5):
        """
        Process image through complete ALPR pipeline.
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detection
            
        Returns:
            dict with pipeline steps and results
        """
        # Step 1: Load image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = {
            'step1_raw': img_rgb.copy(),
            'step2_detection': None,
            'step3_roi': [],
            'step4_ocr': [],
            'step5_final': None,
            'metadata': {
                'image_size': img.shape,
                'detections': [],
                'conditions': self._estimate_conditions(img_rgb)
            }
        }
        
        # Step 2: Detection
        plates_data = []
        img_detected = img_rgb.copy()
        
        if self.is_custom_model:
            # Custom SimpleYOLO inference
            print("🚀 Performing custom model inference...")
            plates_data = self._run_custom_inference(img, conf_threshold)
        else:
            # Standard YOLOv8 Detection
            detections = self.yolo_model(img, conf=conf_threshold, verbose=False)
            
            for detection in detections:
                boxes = detection.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    plates_data.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'roi': img_rgb[y1:y2, x1:x2]
                    })
        
        # Draw detections for step 2
        for plate in plates_data:
            x1, y1, x2, y2 = plate['bbox']
            cv2.rectangle(img_detected, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        results['step2_detection'] = img_detected
        results['metadata']['detections'] = plates_data
        
        # Step 3 & 4: ROI extraction and OCR
        final_img = img_rgb.copy()
        
        for plate_data in plates_data:
            roi = plate_data['roi']
            x1, y1, x2, y2 = plate_data['bbox']
            
            # Store ROI
            results['step3_roi'].append(roi)
            
            # OCR processing
            try:
                # Convert ROI to grayscale (OCR model expects single channel)
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                roi_array = np.ascontiguousarray(roi_gray)
                ocr_result = self.ocr_model.run(roi_array, return_confidence=True)
                if ocr_result and len(ocr_result) == 2:
                    plates, confidences = ocr_result
                    if plates and len(plates) > 0:
                        text = plates[0]
                        # Average confidence across characters
                        ocr_conf = float(np.mean(confidences[0])) if len(confidences) > 0 else 0.0
                    else:
                        text = ""
                        ocr_conf = 0.0
                else:
                    text = ""
                    ocr_conf = 0.0
            except Exception as e:
                print(f"⚠️ OCR Error: {e}")
                text = ""
                ocr_conf = 0.0
            
            results['step4_ocr'].append({
                'text': text,
                'confidence': ocr_conf,
                'detection_confidence': plate_data['confidence']
            })
            
            # Annotate final image
            cv2.rectangle(final_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Add text label
            label = text if text else "???"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(final_img, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), (0, 255, 0), -1)
            cv2.putText(final_img, label, (x1 + 5, y1 - 5), font, font_scale, (0, 0, 0), thickness)
        
        results['step5_final'] = final_img
        
        return results

    def _run_custom_inference(self, img_bgr, conf_threshold):
        """Helper for SimpleYOLO inference and post-processing."""
        import torch
        
        # Preprocessing
        img_h, img_w = img_bgr.shape[:2]
        img_resized = cv2.resize(img_bgr, (416, 416))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        
        # Model forward
        with torch.no_grad():
            output = self.yolo_model(img_tensor) # (1, 13, 13, 6)
        
        pred = output[0] # (13, 13, 6)
        GRID_SIZE = 13
        
        plates = []
        for cy in range(GRID_SIZE):
            for cx in range(GRID_SIZE):
                obj_conf = pred[cy, cx, 0].item()
                if obj_conf > conf_threshold:
                    # Decoding coordinates
                    xc = (cx + pred[cy, cx, 1].item()) / GRID_SIZE
                    yc = (cy + pred[cy, cx, 2].item()) / GRID_SIZE
                    w = abs(pred[cy, cx, 3].item())
                    h = abs(pred[cy, cx, 4].item())
                    
                    # Back to absolute pixel coordinates
                    x1 = int((xc - w/2) * img_w)
                    y1 = int((yc - h/2) * img_h)
                    x2 = int((xc + w/2) * img_w)
                    y2 = int((yc + h/2) * img_h)
                    
                    # Clamp
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_w, x2), min(img_h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        # Extract ROI from original image
                        img_orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        roi = img_orig_rgb[y1:y2, x1:x2]
                        
                        plates.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': obj_conf,
                            'roi': roi
                        })
        
        return plates
    
    def _estimate_conditions(self, img_rgb):
        """
        Estimate image conditions (lighting, quality).
        
        Args:
            img_rgb: RGB image
            
        Returns:
            dict with condition estimates
        """
        # Convert to grayscale for brightness analysis
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(gray)
        
        # Categorize lighting
        if mean_brightness < 60:
            lighting = "Night / Very low light"
            lighting_emoji = "🌙"
        elif mean_brightness < 100:
            lighting = "Low light"
            lighting_emoji = "🌆"
        elif mean_brightness < 160:
            lighting = "Medium light"
            lighting_emoji = "☁️"
        else:
            lighting = "Daylight / Well lit"
            lighting_emoji = "☀️"
        
        # Estimate blur (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 100:
            blur = "High blur"
        elif laplacian_var < 500:
            blur = "Medium blur"
        else:
            blur = "Sharp"
        
        return {
            'brightness': float(mean_brightness),
            'lighting': lighting,
            'lighting_emoji': lighting_emoji,
            'blur': blur,
            'blur_score': float(laplacian_var)
        }
    
    def process_video(self, video_path, max_frames=30, conf_threshold=0.5):
        """
        Process video frames (sample frames for demo).
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process
            conf_threshold: Detection confidence threshold
            
        Returns:
            List of frame results
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames evenly
        frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
        
        results = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Save frame temporarily
            temp_path = f"/tmp/frame_{idx}.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Process frame
            frame_result = self.process_image(temp_path, conf_threshold)
            frame_result['frame_number'] = idx
            results.append(frame_result)
            
            # Cleanup
            os.remove(temp_path)
        
        cap.release()
        
        return results
