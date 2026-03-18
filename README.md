# Real-Time Segmentation Demo (P2AT & DDRNet)

A Python project that wraps both **P2AT** and **DDRNet-23-Slim** PyTorch models into a common interface for real-time webcam inference and web demo capabilities. Optimized for Python 3.10 on Linux with CUDA support.

## Key Features
- **Shared Interface**: Common API for different real-time segmentation backends.
- **Dynamic Configuration**: Automatic class detection from model checkpoints (P2AT).
- **Webcam Integration**: Real-time inference CLI for direct camera stream processing.
- **Gradio Web Demo**: Interactive web interface with a built-in class legend and camera streaming.
- **Third-Party Compatibility**: Seamless integration with official repos without code modification.

## Setup Instructions

1.  **Clone Repositories**:
    Clone the official P2AT and DDRNet repositories into the respective `third_party/` directories:
    ```bash
    git clone https://github.com/mohamedac29/P2AT third_party/p2at
    git clone https://github.com/ydhongHIT/DDRNet third_party/ddrnet
    ```

2.  **Install Environment**:
    Install the Python requirements. We assume Python 3.10 and a CUDA-capable GPU.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Pre-trained Weights**:
    Place the pretrained models in the `checkpoints/` directory. 
    - **P2AT**: e.g. `checkpoints/P2AT-M_best_cityscapes`.
    - **DDRNet**: e.g. `checkpoints/DDRNet-23s_best_cityscapes`.

## Running the Demos

Before running, ensure your project root is in the PYTHONPATH:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Webcam CLI Demo
Processes the stream from camera `0` and displays an overlay.

**Run P2AT Model (Cityscapes)**
```bash
python3 -m rtseg.demo.webcam_demo \
    --model p2at \
    --cfg third_party/p2at/configs/camvid/P2AT_medium_camvid.yaml \
    --checkpoint checkpoints/P2AT-M_best_cityscapes \
    --device cuda \
    --camera 0
```

**Run DDRNet Model (Cityscapes)**
```bash
python3 -m rtseg.demo.webcam_demo \
    --model ddrnet \
    --checkpoint checkpoints/DDRNet-23s_best_cityscapes \
    --device cuda \
    --camera 0
```

### Gradio Web App
Starts a local web interface [http://localhost:7860](http://localhost:7860) where you can select between 4 different checkpoints (P2AT-M/DDRNet-23s on Cityscapes/Sydneyscapes).

```bash
python3 -m rtseg.demo.gradio_app
```

## Advanced Usage
Models automatically handle class mismatches (e.g., loading a 19-class checkpoint into an 11-class config) by inspecting the weight shapes during loading. This allows mixed use of Cityscapes and CamVid checkpoints without manual code changes.
# satriasaid-AdvancedAIAssignment1-RTS-Ready
