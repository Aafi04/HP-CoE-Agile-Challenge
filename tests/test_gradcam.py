import sys
sys.path.insert(0, '.')
import os
import cv2
from evaluation.gradcam import gradcam_from_path

# Pick one real and one fake image from FF++
REAL_IMG = '/home/mdaafi04/data/faceforensics/dataset_processed_split/test/Real/000_f0.jpg'
FAKE_IMG = '/home/mdaafi04/data/faceforensics/dataset_processed_split/test/Deepfakes/000_f0.jpg'
MODEL_PATH = 'checkpoints/hybrid_best.pt'

def test_gradcam(img_path, label):
    confidence, heatmap, is_fake = gradcam_from_path(
        img_path, MODEL_PATH, model_type='hybrid', device='cuda'
    )
    result = 'FAKE' if is_fake else 'REAL'
    print(f"[{label}] Confidence: {confidence:.4f} | Prediction: {result}")
    # Save heatmap
    out_path = f"evaluation/gradcam_{label.lower()}.png"
    os.makedirs('evaluation', exist_ok=True)
    import cv2
    cv2.imwrite(out_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    print(f"Heatmap saved: {out_path}")

if __name__ == '__main__':
    # Find actual existing images
    import os
    real_dir = '/home/mdaafi04/data/faceforensics/dataset_processed_split/test/Real/'
    fake_dir = '/home/mdaafi04/data/faceforensics/dataset_processed_split/test/Deepfakes/'
    real_img = os.path.join(real_dir, os.listdir(real_dir)[0])
    fake_img = os.path.join(fake_dir, os.listdir(fake_dir)[0])

    test_gradcam(real_img, 'REAL')
    test_gradcam(fake_img, 'FAKE')
