#!/bin/bash  
  
set -e  
  
git clone https://github.com/MengqiZou/color-retrieval.git
cd color-retrieval 
  
conda create --name myenv python=3.10 -y  
  
if ! conda info --envs | grep -q "myenv"; then  
    echo "Failed to create Conda environment"  
    exit 1  
fi  

if [ -f "requirements.txt" ]; then  
    echo "Installing Python dependencies..."  
    conda run -n myenv pip install -r requirements.txt  
fi  
  

conda run -n myenv mkdir -p colorDetectionFastApi/checkpoints  
  

conda run -n myenv bash -c 'wget -O colorDetectionFastApi/checkpoints/face_landmarker_v2_with_blendshapes.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task || echo "Failed to download face_landmarker.task"'  
conda run -n myenv bash -c 'wget -O colorDetectionFastApi/checkpoints/hair_segmenter.tflite https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite || echo "Failed to download hair_segmenter.tflite"'  
   
echo "Project setup complete."
