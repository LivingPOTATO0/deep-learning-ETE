# Car Damage Severity Detection System

A deep learning application that automatically classifies car damage severity levels using a trained CNN model. The system consists of a FastAPI backend for inference and a Streamlit frontend for user interaction.

## Features

- **Deep Learning Model**: Custom CNN architecture (DamageNet) trained to classify car damage into three severity levels:
  - 🟢 Minor
  - 🟡 Moderate
  - 🔴 Severe
- **Test-Time Augmentation**: Improves prediction accuracy through multi-augmentation inference
- **Web Interface**: User-friendly Streamlit UI for uploading and analyzing car images
- **GPU Support**: Automatic GPU acceleration when available

## Project Structure

```
car_severity_model_ui/
├── backend/
│   ├── main.py                      # FastAPI application
│   ├── model.py                     # DamageNet CNN architecture
│   └── imporved_damage_model.pth   # Pre-trained model weights
├── frontend/
│   └── app.py                       # Streamlit UI application
├── README.md                        # This file
└── .gitignore                       # Git ignore rules
```

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

1. **Clone or download the repository**
   ```bash
   cd car_severity_model_ui
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Backend API

```bash
cd backend
python main.py
```

The API will start at `http://localhost:8000` with documentation available at `http://localhost:8000/docs`

**API Endpoint:**
- `POST /predict/` - Upload a car image and get damage severity prediction

### Running the Frontend

In a new terminal:

```bash
cd frontend
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Model Architecture

**DamageNet** is a custom CNN designed for car damage classification:

- **Input**: 224×224 RGB images
- **Feature Extraction**: 5 convolutional blocks with batch normalization and max pooling
- **Classification Head**: Global average pooling + 2 fully connected layers with dropout
- **Output**: 3-class softmax probabilities (minor, moderate, severe)

**Training Details:**
- Normalization: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Input size: 224×224 pixels
- Classes: 3 (minor, moderate, severe)
- Inference: Test-time augmentation with horizontal flips

## Dependencies

Key packages:
- **FastAPI** - Web framework for the backend API
- **Streamlit** - Frontend web application framework
- **PyTorch** - Deep learning framework
- **torchvision** - Computer vision utilities (transforms)
- **Pillow** - Image processing

## File Descriptions

### Backend Files

- **main.py**: FastAPI application that:
  - Loads the pre-trained model
  - Handles image uploads
  - Performs inference with test-time augmentation
  - Returns predictions with confidence scores

- **model.py**: Defines the DamageNet architecture with:
  - Convolutional blocks (conv + batch norm + ReLU + pooling)
  - Adaptive average pooling
  - Classification head with dropout regularization
  - Xavier/Kaiming weight initialization

### Frontend Files

- **app.py**: Streamlit application that:
  - Provides image upload interface
  - Displays uploaded image preview
  - Sends requests to the backend API
  - Shows prediction results with visualizations

## Future Improvements

- Add batch processing capabilities
- Implement model explainability (visualize attention maps)
- Add model retraining interface
- Implement caching for repeated predictions
- Add API rate limiting and authentication
- Support for multiple image formats and video input
- Model quantization for faster inference

## License

This project is provided as-is for educational and research purposes.

## Contact & Support

For issues or questions, please refer to the project documentation or contact the development team.
