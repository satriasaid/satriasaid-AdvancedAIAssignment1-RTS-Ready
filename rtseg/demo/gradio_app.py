import cv2
import gradio as gr
import numpy as np

# A global dictionary to keep Segmenters active
# Prevents reloading weights on every inference
_models = {
    "p2at": None,
    "ddrnet": None
}

CITYSCAPES_CLASSES = [
    "Road", "Sidewalk", "Building", "Wall", "Fence", "Pole",
    "Traffic Light", "Traffic Sign", "Vegetation", "Terrain",
    "Sky", "Person", "Rider", "Car", "Truck", "Bus", "Train",
    "Motorcycle", "Bicycle"
]

def get_html_legend():
    from rtseg.common.palette import CITYSCAPES_CAMVID_PALETTE
    html = "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 8px; margin-top: 15px; padding: 10px; background: rgba(0,0,0,0.05); border-radius: 8px;'>"
    for i, name in enumerate(CITYSCAPES_CLASSES):
        bgr = CITYSCAPES_CAMVID_PALETTE.get(i, (0, 0, 0))
        # Convert BGR to RGB for HTML display
        rgb = (bgr[2], bgr[1], bgr[0])
        color_str = f"rgb{rgb}"
        html += f"<div style='display: flex; align-items: center; font-size: 14px; color: #333;'>" \
                f"<div style='width: 18px; height: 18px; background-color: {color_str}; margin-right: 8px; border-radius: 3px; border: 1px solid rgba(0,0,0,0.1);'></div>" \
                f"<span>{name}</span></div>"
    html += "</div>"
    return html

def load_models_if_needed():
    # Only try to load them if they don't exist yet
    # Adjust paths or load from environment variables if necessary.
    # For the web demo, we assume the user has configured the standard paths.
    global _models
    
    device = "cuda"
    
    if _models["p2at"] is None:
        try:
            from rtseg.models.p2at_segmenter import P2ATSegmenter
            # Correct paths for P2AT-Medium
            cfg = "third_party/p2at/configs/camvid/P2AT_medium_camvid.yaml"
            ckpt = "checkpoints/P2AT-M_best.pth"
            _models["p2at"] = P2ATSegmenter(cfg, ckpt, device)
            print("Loaded P2AT web wrapper.")
        except Exception as e:
            print(f"P2AT optional load failed: {e}")
            
    if _models["ddrnet"] is None:
        try:
            from rtseg.models.ddrnet_segmenter import DDRNet23SlimSegmenter
            # Correct path for DDRNet-23s
            ckpt = "checkpoints/DDRNet-23s_best.pth"
            _models["ddrnet"] = DDRNet23SlimSegmenter(ckpt, device)
            print("Loaded DDRNet web wrapper.")
        except Exception as e:
            print(f"DDRNet optional load failed: {e}")


def process_video_frame(frame: np.ndarray, model_name: str) -> np.ndarray:
    """
    Gradio image input is usually RGB.
    Our segmenters expect BGR.
    """
    if _models.get(model_name) is None:
        return frame # Model not loaded, return original
        
    segmenter = _models[model_name]
    
    # Gradio passes RGB, convert to BGR for our pipelines
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Inference
    seg_bgr = segmenter.segment(bgr_frame)
    
    # Overlay
    overlay_bgr = cv2.addWeighted(bgr_frame, 0.5, seg_bgr, 0.5, 0)
    
    # Convert back to RGB for Gradio
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    
    return overlay_rgb

def create_app():
    load_models_if_needed()
    
    available_models = [name for name, mdl in _models.items() if mdl is not None]
    if not available_models:
        default_model = "None (Download checkpoints and config)"
        available_models = [default_model]
    else:
        default_model = available_models[0]
        
    with gr.Blocks(title="Real-Time Segmentation") as app:
        gr.Markdown("# PyTorch Real-Time Segmentation Demo\nCompare **P2AT** and **DDRNet** on webcam.")
        
        with gr.Row():
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    value=default_model,
                    label="Select Model"
                )
                gr.Markdown("Make sure your webcam is enabled. The app streams frames, processes them through the PyTorch model, and returns the segmentation overlay.")
            
            with gr.Column(scale=3):
                # Gradio 4.x streaming webcam
                # Use Image component with source="webcam" and streaming=True
                img_input = gr.Image(sources=["webcam"], streaming=True, label="Webcam Stream (Input/Output)")
                
                # Add Legend
                gr.Markdown("### Class Legend")
                gr.HTML(get_html_legend())
                
        # Connect the stream to the process function
        img_input.stream(
            fn=process_video_frame,
            inputs=[img_input, model_dropdown],
            outputs=[img_input],
            time_limit=15 # Avoid infinite stream locks on HuggingFace, unneeded for local but safe
        )

    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
