import argparse
import cv2
import time
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Real-time webcam segmentation demo.")
    parser.add_argument("--model", type=str, required=True, choices=["p2at", "ddrnet"], 
                        help="Model to run: 'p2at' or 'ddrnet'")
    parser.add_argument("--cfg", type=str, default=None, 
                        help="Path to p2at config yaml (required for p2at).")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to the model weights (.pth).")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to run on (e.g. 'cuda' or 'cpu').")
    parser.add_argument("--camera", type=int, default=0, 
                        help="Camera index to open.")
    args = parser.parse_args()

    # Load Model
    print(f"Loading model '{args.model}' on {args.device}...")
    start_time = time.time()
    
    if args.model == "p2at":
        if not args.cfg:
            print("Error: --cfg is required when using p2at model.")
            sys.exit(1)
        from rtseg.models.p2at_segmenter import P2ATSegmenter
        try:
            segmenter = P2ATSegmenter(args.cfg, args.checkpoint, args.device)
        except Exception as e:
            print(f"Failed to load P2AT: {e}")
            sys.exit(1)
            
    elif args.model == "ddrnet":
        from rtseg.models.ddrnet_segmenter import DDRNet23SlimSegmenter
        try:
            segmenter = DDRNet23SlimSegmenter(args.checkpoint, args.device)
        except Exception as e:
            print(f"Failed to load DDRNet: {e}")
            sys.exit(1)
            
    print(f"Model loaded in {time.time() - start_time:.2f}s")
    
    # Open Webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        sys.exit(1)
        
    print("Starting inference loop. Press 'ESC' to exit.")
    
    # Loop
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera. Exiting.")
                break
                
            start_infer = time.time()
            
            # Inference
            seg_bgr = segmenter.segment(frame)
            
            infer_time = time.time() - start_infer
            fps = 1.0 / infer_time if infer_time > 0 else 0.0
            
            # Overlay
            overlay = cv2.addWeighted(frame, 0.5, seg_bgr, 0.5, 0)
            
            # Add FPS text
            cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show
            cv2.imshow("Real-Time Segmentation Demo", overlay)
            
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break
                
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    main()
