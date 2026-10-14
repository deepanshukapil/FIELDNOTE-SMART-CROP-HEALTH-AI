# Smart Crop Health AI

A responsive public website with real ML crop suitability recommendations and a trained CNN for leaf-photo disease classification. It uses the included public datasets and trained model checkpoints; it does not fabricate predictions when models or valid inputs are unavailable.

## Included models and data

- Crop Random Forest checkpoint: `models/crop_risk.joblib`, trained on 2,200 rows across 22 crop labels. A random 20% holdout run reported 99% accuracy on this dataset; this is not field-validation accuracy or calibrated crop-loss risk.
- Leaf CNN checkpoint: `models/leaf_cnn.pt`, with labels in `models/leaf_classes.json`. It was trained for 8 epochs on 3,800 real RGB images (100 per class across 38 PlantVillage classes). Best validation accuracy was 67.2% on a 760-image holdout sampled from the same curated source; this does not establish accuracy on real-world field photos.
- Dataset files and attribution: `data/crop_recommendation.csv`, `data/plantvillage_sample/` and `data/DATASETS.md`.

## Run locally

Python 3.12 is recommended. Create and activate a virtual environment. Install CPU PyTorch first, then website dependencies:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu torchvision==0.29.1+cpu
python -m pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 3000
```

Open `http://localhost:3000`. The health endpoint is `/health`.

## Retrain

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu torchvision==0.29.1+cpu
python -m pip install -r requirements.txt
python src/train_risk.py --csv data/crop_recommendation.csv
python src/train_leaf.py --data data/plantvillage_sample --epochs 8 --batch-size 32
```

## Production

The root `Dockerfile` creates a Python CPU inference image, installs the saved model files, and listens on the platform-supplied `PORT` (3000 by default). The development training-image sample is kept in the repository for reproducibility, but is intentionally not copied into the runtime image.

## Model limits

The crop classifier recommends only among labels present in its training CSV. The selected-crop risk band is a model-probability signal, not a calibrated crop-failure probability. The leaf CNN was trained on curated, controlled-background leaf images, and may be less reliable under different field lighting, crop varieties, cameras, or symptom stages. Outputs are informational—not a substitute for expert agronomic diagnosis. Seek local advice before treatment or pesticide decisions.
