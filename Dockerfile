FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model during build (not runtime!)
RUN python -c "from torchvision import models; models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)"

# Copy your code
COPY . .

CMD ["python", "train_model.py"]
