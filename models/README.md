# Inference model

The deployment app expects `handwriting_personality_resnet50.keras` in this directory. The file is intentionally allow-listed in the repository `.gitignore`; training datasets and generated split artifacts are not deployment dependencies.

For repositories that keep large model files outside Git, set the Streamlit secret or environment variable `HANDWRITING_MODEL_URL` to a direct, publicly readable `.keras` URL. The app downloads that file only when the local ResNet50 model is absent.