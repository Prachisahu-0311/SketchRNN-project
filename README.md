# SketchRNN – Autoregressive Sketch Model with LSTM

This project utilizes a generative sequence model with LSTM networks to generate 2D sketches based on the Google QuickDraw Dataset. The model learns to draw objects such as apple, eye, square, triangle, and circle by predicting autoregressive sequences of strokes.

## Project Overview

- Developed a deep learning model with PyTorch that learned simple object drawings' temporal stroke patterns.
- Employed LSTM structure with embedding layers to condition generation on object category.
- Trained on more than 100,000 sketches per category from the QuickDraw dataset.
- Model produces sketches line-by-line from scratch, replicating how a person draws.

## Key Features and Implementation

- Preprocessing:
  - Tokenized .ndjson QuickDraw data and normalized relative coordinates to absolute (x, y) coordinates.
  - Balanced dataset with identical samples per chosen object classes.

- Model Architecture:
  - Class conditioning embedding layer
  - LSTM layers to capture sequential stroke generation patterns
  - Dropout regularization to avoid overfitting
  - Fully connected layer predicts next stroke coordinates and pen state

- Training Details:
  - Trained with Adam optimizer and Mean Squared Error (MSE) loss
  - Learning rate scheduling and early stopping implemented
- Visualized performance with both loss curves and generated images

## Results

The model generated recognizable and smooth stroke sequences for all target classes.

### Sample Outputs

| Object Class | Generated Sketch |
|--------------|------------------|
| Apple        | apple_sketch.gif  |
| Circle       | circle_sketch.png |
| Eye          | eye_sketch.png |
| Triangle     | triangle_sketch.png |
| Square       | square_sketch.png |

Note: All sketches are made by the model from scratch, and not copied from the dataset.

## Technologies Used

- Python
- PyTorch
- NumPy, Matplotlib
- Google Colab
- LSTM Networks
- Google QuickDraw Dataset

## Project Structure

SketchRNN-Project/
│
├── SketchRNN_Final.ipynb       # Colab notebook with full implementation
├── generated_outputs/          # Sample output images and GIFs
├── report.pdf                  # Project report
├── requirements.txt            # Dependencies (optional)
└── README.md                   # Project documentation

## How to Run

If you'd like to give it a try yourself:
1. Clone the repository:
2. Open `SketchRNN_Final.ipynb` in Google Colab.
3. Run all the cells step-by-step. The notebook includes full implementation and explanations.
4. Download the QuickDraw data from https://quickdraw.withgoogle.com/data to use with the model.

## License

This project is open-source under the MIT License.
