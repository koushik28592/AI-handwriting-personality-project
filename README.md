# AI in Cognitive Modeling: Handwriting Classification

This is an experimental classifier for five dataset-defined labels: `Extrovert`, `Introvert`, `Optimistic`, `Pessimistic`, and `Stable_Mindset`.

It models the research idea: `Handwriting -> observable behavioral patterns -> feature representation -> AI model -> dataset-label classification`.

It does **not** establish that handwriting determines personality. These labels are not validated psychological traits, and predictions must not be used for diagnosis, hiring, education, or other high-impact decisions.

## Dataset and preprocessing

Extract the dataset beside `src/`:

```text
Dataset/{Extrovert,Introvert,Optimistic,Pessimistic,Stable_Mindset}/
```

The analyzer validates images, records dimensions and color modes, hashes files for exact duplicate detection, removes duplicates before splitting, and writes a deterministic stratified train/validation/test manifest. Images are RGB-decoded, resized with aspect-ratio-preserving padding to `224x224`, normalized for the selected backbone, and augmented with restrained rotation, translation, zoom, and contrast changes.

## Architecture

`src/data` owns integrity checks and leakage-safe splits. `src/features` extracts real OpenCV ink density, bounding-box geometry, contour statistics, estimated slant, baseline variation, and spacing features. `src/models` contains a custom CNN, ResNet50, and EfficientNetB0. `src/train.py` uses class-weighted loss, dropout, batch normalization, early stopping, learning-rate reduction, and fine-tuning. `reports/` stores generated evidence, `models/` stores trained models, and `tests/` contains focused tests.

## Installation and usage

The prediction app does not require the training dataset. Install the dependencies and launch it from the repository root:

```bash
pip install -r requirements.txt
python -m streamlit run src/app.py
```

Training and evaluation are optional workflows that do require a local `Dataset/` directory:

```bash
python src/analyze_dataset.py --data_dir Dataset
python src/train.py --data_dir Dataset --model cnn
python src/train.py --data_dir Dataset --model resnet50
python src/train.py --data_dir Dataset --model efficientnetb0
python src/classical.py --data_dir Dataset
python src/evaluate.py --data_dir Dataset --model resnet50
python src/predict.py --image path/to/handwriting.jpg --model resnet50
```

Select models by validation macro F1, not accuracy. Evaluation uses the held-out test split and saves accuracy, balanced accuracy, macro and weighted precision/recall/F1, a confusion matrix, and training curves. The app shows prediction, confidence, probabilities, top three classes, and Grad-CAM. Grad-CAM highlights influential image regions; it is not proof of a personality claim.

## Limitations and future work

Without writer IDs, image-level splitting may still overestimate generalization when several samples come from one writer. Writer-grouped splitting, validated psychological labels, external testing, calibration, confidence intervals, and an image-plus-OpenCV feature fusion model are important next steps. Dataset imbalance, label quality, language, scanning conditions, and demographic confounding can affect results.

## Streamlit Community Cloud deployment

The app is designed to run from the repository root and resolves CSS, reports, and models relative to the repository, not the current working directory. It does not use Windows absolute paths. The deployment runtime is declared in `runtime.txt`, and the inference dependencies are listed in `requirements.txt`.

Do not commit `Dataset/` or any raw handwriting images. The deployed app needs only the inference model, `models/class_names.json`, `app/styles.css`, generated report artifacts, and application code. The ResNet50 artifact is allow-listed in `.gitignore` at `models/handwriting_personality_resnet50.keras`. If the model exceeds your Git provider's file-size limit, host it at a direct HTTPS URL and configure the `HANDWRITING_MODEL_URL` Streamlit secret; the local model takes precedence when both are available.

In Streamlit Community Cloud, select the repository, branch, and main file:

```text
src/app.py
```

The app can also load the model from external storage. Add a Streamlit secret named `HANDWRITING_MODEL_URL` whose value is a direct HTTPS URL to `handwriting_personality_resnet50.keras`. Never place private credentials in source code or commit them.

Test locally from the repository root:

```bash
python -m streamlit run src/app.py
```

The app should open the dashboard without a dataset. Uploading an image requires the committed local model or `HANDWRITING_MODEL_URL`; training, dataset analysis, and classical-model scripts are not part of the Cloud startup path.
