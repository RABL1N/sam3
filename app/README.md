# SAM3 Annotation Pipeline

Cross-platform image annotation pipeline using SAM3 (PyTorch) with a Next.js frontend and FastAPI backend.

## Features

- **Text Prompts**: Segment objects using natural language descriptions
- **Box Prompts**: Draw bounding boxes to include or exclude regions
- **Point Prompts**: Click on images to add positive or negative point prompts
- **Real-time Segmentation**: Fast inference with visual feedback
- **COCO Format Export**: Save annotations in standard COCO format
- **Cross-Platform**: Works on Windows, macOS, and Linux

## Prerequisites

- Python 3.10+
- Node.js 18+
- PyTorch (with CUDA support if using GPU)
- SAM3 model weights (see Model Weights section below)

## Quick Start

### Backend Setup

1. **Navigate to sam3 root directory:**
   ```bash
   cd /path/to/sam3
   ```

2. **Install SAM3 package:**
   ```bash
   # Using uv (recommended)
   uv pip install -e .
   
   # Or using pip
   pip install -e .
   ```

3. **Install backend dependencies:**
   ```bash
   cd app/backend
   pip install -r requirements.txt
   # Or with uv:
   uv pip install -r requirements.txt
   ```

4. **Install PyTorch:**
   ```bash
   # For CUDA (recommended if you have NVIDIA GPU)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # For CPU only
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

5. **Set up model weights:**
   
   The SAM3 model repository on Hugging Face is gated. You have three options:
   
   **Option A: Authenticate with Hugging Face (recommended)**
   ```bash
   huggingface-cli login
   # Enter your Hugging Face token when prompted
   ```
   
   **Option B: Use local checkpoint**
   ```bash
   export SAM3_CHECKPOINT_PATH=/path/to/sam3.pt
   ```
   
   **Option C: Download manually**
   - Get access to https://huggingface.co/facebook/sam3
   - Download `sam3.pt` 
   - Set `SAM3_CHECKPOINT_PATH` environment variable

6. **Run the backend:**
   ```bash
   # From sam3 root directory
   uv run python3 app/backend/main.py
   
   # Or with uvicorn for auto-reload:
   uv run uvicorn app.backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd app/frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Run the frontend:**
   ```bash
   npm run dev
   ```

4. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## Usage

1. **Upload an image** by clicking or dragging it into the upload area
2. **Add prompts:**
   - **Text**: Type a description (e.g., "person", "dog") and press Enter
   - **Box**: Click "Include" or "Exclude" buttons, then draw boxes on the image
   - **Point**: Click "Include" or "Exclude" buttons, then click on the image
3. **View results**: Masks and bounding boxes will appear on the canvas
4. **Save annotations**: Click "Save Annotations" to export in COCO format

## API Endpoints

- `GET /health` - Check backend status
- `POST /upload` - Upload and process an image
- `POST /segment/text` - Segment with text prompt
- `POST /segment/box` - Add box prompt
- `POST /segment/point` - Add point prompt
- `POST /reset` - Reset all prompts
- `POST /save-annotations` - Save annotations in COCO format
- `DELETE /session/{id}` - Delete a session

See http://localhost:8000/docs for interactive API documentation.

## Configuration

The backend automatically detects GPU availability. To force CPU mode:

```bash
# The backend will automatically use CPU if CUDA is not available
# No configuration needed
```

Key environment variables:
- `SAM3_CHECKPOINT_PATH`: Path to local SAM3 model checkpoint (optional, defaults to Hugging Face)

## Project Structure

```
sam3/
├── app/
│   ├── backend/          # FastAPI backend
│   │   ├── main.py       # Main application
│   │   └── requirements.txt
│   └── frontend/         # Next.js frontend
│       ├── src/
│       │   ├── app/      # Next.js app directory
│       │   ├── components/
│       │   └── lib/
│       └── package.json
└── README.md
```

## Troubleshooting

### Backend Issues

- **Model not loading**: Ensure SAM3 package is installed and model weights are available
- **CUDA errors**: Check PyTorch installation and CUDA compatibility. The backend will automatically fall back to CPU if CUDA is not available
- **Out of memory**: Reduce image resolution or the backend will automatically use CPU mode
- **"No module named 'numpy'"**: Make sure you installed the SAM3 package with `pip install -e .` or `uv pip install -e .`

### Frontend Issues

- **API connection errors**: Check that backend is running and accessible at http://localhost:8000
- **Build errors**: Ensure Node.js 18+ is installed and dependencies are up to date

### Device Detection

The backend automatically detects and uses GPU if available. If you see "Using device: cpu" in the logs:
- Check that PyTorch was installed with CUDA support
- Verify GPU is accessible: `nvidia-smi`
- The backend will work on CPU but will be slower

## License

See the main SAM3 repository license.
