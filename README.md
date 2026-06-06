# 🔍 DeepFake Detector

> Binary facial inconsistency classifier using **EfficientNet-B0 + Dual Attention Maps**

---

## Architecture

```
Input Image (224×224)
       │
  EfficientNet-B0 Backbone
       │
  Dual Attention Module
  ├── Channel Attention  (global avg pool → MLP → sigmoid)
  └── Spatial Attention  (avg+max pool → Conv7×7 → sigmoid)
       │
  Global Average Pool
       │
  Classifier Head (1280 → 256 → 1) + GELU + Dropout
       │
  Sigmoid → FAKE/REAL probability
```

**Auxiliary outputs:**
- Spatial attention overlay heatmap (GradCAM-style)
- DCT frequency analysis map
- OpenCV face detection count
- Attention entropy metric

---

## Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🚀 Deploy to Streamlit Cloud

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "deepfake detector"
   git remote add origin https://github.com/YOUR_USERNAME/deepfake-detector.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click **New app**
   - Select your repo → `app.py`
   - Click **Deploy**

> ⚠️ First deploy may take ~5 min (PyTorch download). Subsequent loads use cache.

---

## Training on Real Data (Optional)

To fine-tune on a real deepfake dataset (e.g., FaceForensics++, DFDC):

```python
from app import EfficientNetAttentionClassifier
import torch

model = EfficientNetAttentionClassifier(pretrained=True)
# Replace final linear with your trained weights:
model.load_state_dict(torch.load("your_weights.pth", map_location="cpu"))
model.eval()
```

Save weights as `weights.pth` in the project root and update `load_model()` accordingly.

---

## Notes

- Currently uses **ImageNet pretrained weights** as a proxy classifier for demo purposes
- For production accuracy, fine-tune on labeled deepfake datasets
- Research/educational use only
