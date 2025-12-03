import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from ultralytics import YOLO

# ==========================================
# 1. Setup & Configuration
# ==========================================
# Update this path to match your specific folder location
os.chdir(r"C:\Users\pandy\Desktop\School Work\AER850\Project 3\AER850-Project-3\Project 3 Code Data Final")
print(f"Working Directory: {os.getcwd()}")

# GPU Check
if torch.cuda.is_available():
    device_id = 0
    print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
else:
    device_id = 'cpu'
    print("WARNING: No GPU found. Training will be slow.")

# ==========================================
# Step 1: Object Masking (Image Processing)
# ==========================================
def run_step1_masking():
    print("\n--- Starting Step 1: Object Masking ---")
    img_path = 'motherboard_image.JPEG'
    image = cv2.imread(img_path)

    if image is None:
        print(f"Error: Could not find {img_path}")
        return

    # A. Preprocessing (Grayscale + Blur)
    process_img = image.copy()
    gray = cv2.cvtColor(process_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # B. Edge Detection (Canny)
    edges = cv2.Canny(blurred, 50, 150)

    # C. Dilation (Close gaps in edges)
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)

    # D. Contour Detection
    contours, hierarchy = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # E. Filter (Largest Contour = Motherboard)
        largest_contour = max(contours, key=cv2.contourArea)

        # F. Masking
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)

        # G. Extraction
        result = cv2.bitwise_and(image, image, mask=mask)

        # H. Visualization
        plt.figure(figsize=(20, 6))
        
        plt.subplot(1, 3, 1)
        plt.imshow(dilated_edges, cmap='gray')
        plt.title("Canny Edges")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Generated Mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title("Final Extraction")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()

        cv2.imwrite('extracted_motherboard.png', result)
        print("Step 1 Complete. Saved 'extracted_motherboard.png'.")
    else:
        print("Step 1 Failed: No contours found.")

# ==========================================
# Step 2: YOLO Training
# ==========================================
def run_step2_training():
    print("\n--- Starting Step 2: YOLO Training ---")
    model = YOLO('yolo11n.pt') 

    results = model.train(
        data='data.yaml',   
        epochs=150,         
        imgsz=1024,         
        batch=4,            
        device=device_id,   
        patience=15,        
        workers=2,          
        name='pcb_model',   
        exist_ok=True,      
        plots=True,         
        verbose=True
    )
    print("Step 2 Training Finished.")

# ==========================================
# Step 3: Evaluation (Visuals Only)
# ==========================================
def run_step3_evaluation():
    print("\n--- Starting Step 3: Evaluation ---")
    
    # Load your custom trained model
    model_path = r'runs\detect\pcb_model\weights\best.pt'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    model = YOLO(model_path)
    eval_folder = 'evaluation' 
    
    if not os.path.exists(eval_folder):
        print(f"Error: Folder '{eval_folder}' missing.")
        return

    # Run Prediction
    print("Running predictions...")
    results = model.predict(
        source=eval_folder, 
        conf=0.25, 
        save=True, 
        project='runs/detect', 
        name='evaluation_results',
        exist_ok=True,
        line_width=3,      # Thicker lines for visibility
        show_conf=False,   # Hide scores to clean up the view
        show_labels=True
    )

    # Visualize the 3 Result Images
    saved_folder = r'runs\detect\evaluation_results'
    eval_images = glob.glob(os.path.join(saved_folder, '*.*'))
    
    if len(eval_images) > 0:
        plt.figure(figsize=(20, 8))
        for i, img_file in enumerate(eval_images):
            if i >= 3: break 
            
            img = cv2.imread(img_file)
            if img is None: continue
            
            plt.subplot(1, 3, i + 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"Evaluation Image {i+1}")
            plt.axis("off")
        
        plt.tight_layout()
        plt.show()
        print(f"Step 3 Complete. Results in '{saved_folder}'.")
    else:
        print("No evaluation images found.")

# ==========================================
# Main Execution Flow
# ==========================================
if __name__ == '__main__':
    # 1. Run Object Masking
    run_step1_masking()

    # 2. Run Training (Set to False if yu want model to trian again)
    SKIP_TRAINING = False
    
    if not SKIP_TRAINING:
        run_step2_training()
    else:
        print("\nSkipping Step 2 (Training already complete).")

    # 3. Run Evaluation
    run_step3_evaluation()