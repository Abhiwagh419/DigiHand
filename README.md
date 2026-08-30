# Handwritten-to-Digital Converter

Digitizes handwritten digits, letters, and math expressions using a CNN +
OpenCV-based character segmentation. Draw on a canvas (or upload a scan),
and it reconstructs the text — evaluating it live if it's a math expression.

## How it works

1. **Model**: A CNN trained from scratch on EMNIST (`balanced` split — digits
   + letters, 47 classes) combined with a curated subset of the Kaggle
   Handwritten Math Symbols dataset (+, -, ×, ÷, =, (, ), .).
2. **Segmentation**: OpenCV contour detection finds each character as a
   separate blob in the canvas image, crops it, and orders them left-to-right.
3. **Inference**: Each cropped character is classified individually by the CNN.
4. **Reconstruction**: Predictions are joined in order into a string.
5. **Math evaluation**: If the reconstructed string matches a safe arithmetic
   pattern (digits and `+-*/(). ` only), it's evaluated and shown as the result.

## Project structure

```
handwritten-to-digital/
├── notebooks/
│   └── train_model.py      # Run in Google Colab (needs GPU) to train the CNN
├── backend/
│   ├── app.py               # Flask API: segmentation + inference + eval
│   ├── requirements.txt
│   └── model/                # Put char_cnn.pt + class_map.json here after training
├── frontend/
│   └── index.html            # Canvas UI, talks to the Flask backend
└── README.md
```

## Setup steps

### 1. Train the model (Google Colab)
- Open a new Colab notebook, set Runtime → GPU
- Copy `notebooks/train_model.py` cell-by-cell (or upload as .py and `%run` it)
- Upload your `kaggle.json` when prompted, to download the math symbols dataset
- Training takes ~10-20 minutes on Colab's free GPU for 15 epochs
- Download the two output files: `char_cnn.pt` and `class_map.json`

### 2. Set up the backend
```bash
cd backend
mkdir -p model
# place char_cnn.pt and class_map.json into backend/model/
pip install -r requirements.txt
python app.py
```
Backend runs at `http://localhost:5000`.

### 3. Open the frontend
Just open `frontend/index.html` in a browser (double-click it, or serve it
with any static server). Make sure the backend is running first.

## Known limitations (be upfront about these in a presentation/viva)

- Works on **well-spaced, block-style handwriting** — connected cursive
  writing will be segmented incorrectly since it relies on contour gaps.
- True sequence-based handwriting recognition (cursive, joined letters) needs
  a CRNN + CTC-loss architecture trained on stroke-trajectory data (e.g. the
  CROHME dataset) — that's a natural extension, not what this system does.
- Math evaluation only supports `+ - * / ( ) .` — no exponents, fractions,
  or multi-line expressions yet.
- Some EMNIST balanced classes merge visually similar upper/lowercase
  letters (e.g. 'c', 'o', 's', 'v', 'w', 'x', 'z') — this is a known EMNIST
  property, not a bug in this project.

## Possible extensions
- Add exponents (`^`) and square roots (`sqrt`) as additional symbol classes
- Swap the segmentation step for a CRNN to eventually support cursive input
- Add a Grad-CAM overlay per character (ties in nicely if paired with a
  computer-vision course concept)
- OCR on uploaded photos of paper (not just canvas) — would need adaptive
  thresholding to handle lighting variation
