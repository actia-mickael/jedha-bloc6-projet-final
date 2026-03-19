
import cv2
import numpy as np
import os

def test_codec(codec, ext):
    filename = f"test_{codec}.{ext}"
    height, width = 64, 64
    fps = 10
    
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"❌ {codec} ({ext}): Failed to open")
            return False
            
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        out.write(frame)
        out.release()
        
        size = os.path.getsize(filename)
        if size > 0:
            print(f"✅ {codec} ({ext}): Success (Size: {size} bytes)")
            return True
        else:
            print(f"❌ {codec} ({ext}): File empty")
            return False
            
    except Exception as e:
        print(f"❌ {codec} ({ext}): Exception {e}")
        return False

print("Testing codecs...")
test_codec('avc1', 'mp4')
test_codec('h264', 'mp4')
test_codec('x264', 'mp4')
test_codec('mp4v', 'mp4')
test_codec('vp80', 'webm')
test_codec('vp09', 'webm')
test_codec('VP80', 'webm')
test_codec('VP90', 'webm')
test_codec('MJPG', 'avi')
