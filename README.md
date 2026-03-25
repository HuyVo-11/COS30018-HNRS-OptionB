# COS30018-HNRS-OptionB

Handwritten Number Recognition System for COS30018 Option B.

This project contains a handwritten character recognition workflow focused on:

- image segmentation
- handwritten digit recognition
- operator classification
- expression validation
- classification evaluation metrics

The repository is currently split into several task-specific modules rather than one fully unified end-to-end application. This README documents the repository as it exists now, including what runs correctly, what requires extra preparation, and which parts are still incomplete.

## 1. Project structure

```text
src/
	evaluation/
		evaluation.py
	model/
		Advance_CNN_MNIST.ipynb
		bestmodel.keras
		merge_minst_emnist_data.py
		test_model.py
		extracted_images/
	preprocessing/
		preprocessing.py
	segmentation/
		expression_parser.py
		main_extension.py
		operator_classifier.py
		segmentation.py
		Test_segmentation.ipynb
		input_image/
	main.py
```

## 2. Main components

### 2.1 Segmentation

`src/segmentation/segmentation.py`

Responsibilities:

- read input image safely
- convert to grayscale and threshold image
- detect contours
- merge broken character fragments
- remove nested noise boxes
- normalize each detected character into 28x28 images
- save segmented results into `output_digit`

### 2.2 Digit-only model

`src/model/test_model.py`

Responsibilities:

- load `src/model/bestmodel.keras`
- read segmented character images
- preprocess them into MNIST-like format
- predict the digit class and confidence

This is the most practical demo path because the trained Keras model is already included in the repository.

### 2.3 Expression pipeline

`src/segmentation/main_extension.py`

Responsibilities:

- call segmentation
- classify segmented characters using PyTorch model
- build an expression string
- display visualization with predicted characters

Important:

- this pipeline expects a trained file named `model_combined.pth`
- that file is not committed in the repository
- you must train it first by running `src/segmentation/operator_classifier.py`

### 2.4 Evaluation

`src/evaluation/evaluation.py`

This module now provides classification metrics aligned with the standard four metrics:

- Accuracy
- Precision
- Recall
- F1-score

It also returns:

- confusion matrix
- per-class metrics
- total sample count

## 3. Python version

Recommended Python version:

- Python 3.10.x

Reason:

- the project uses both TensorFlow/Keras and PyTorch
- Python 3.10 is the safest common version for compatibility on Windows
- newer versions such as Python 3.13 or 3.14 can cause dependency installation problems, especially with TensorFlow

If you are setting up the project on Windows, use Python 3.10 for the virtual environment.

## 4. Dependencies

Current `requirements.txt` contains:

- numpy
- pandas
- matplotlib
- tensorflow
- scikit-learn
- opencv-python

However, the expression pipeline also uses:

- torch
- torchvision

For that reason, install both the requirements file and the missing PyTorch dependencies.

## 5. Environment setup

### 5.1 Create virtual environment

From the project root:

```powershell
cd C:\Users\LENOVO\COS30018-HNRS-OptionB
py -3.10 -m venv .venv
```

### 5.2 Activate virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 5.3 Verify Python version

```powershell
python --version
```

Expected result:

```text
Python 3.10.x
```

### 5.4 Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision keras
```

## 6. Available demo paths

There are two realistic demo paths in this repository.

- Demo A: digit-only prediction using the provided Keras model
- Demo B: expression segmentation and classification using the PyTorch classifier

Demo A is the easiest path to run.

## 7. Demo A: digit-only workflow

This path uses:

- `src/segmentation/segmentation.py`
- `src/model/test_model.py`

### 7.1 Goal

Take an input handwritten image, segment the characters into separate images, then use the provided Keras model to predict the digit value of each segmented image.

### 7.2 Input images

The repository already includes sample inputs in:

- `src/segmentation/input_image/test.jpg`
- `src/segmentation/input_image/test2.jpg`
- `src/segmentation/input_image/test3.jpg`
- `src/segmentation/input_image/test4.jpg`

### 7.3 Step 1: generate segmented character images

Move into the segmentation folder:

```powershell
cd C:\Users\LENOVO\COS30018-HNRS-OptionB\src\segmentation
```

Run segmentation on a sample image:

```powershell
python -c "from segmentation import segment_image; segment_image('input_image/test.jpg')"
```

You can replace `test.jpg` with any of the other provided images.

Expected result:

- the script prints segmentation logs
- a folder named `output_digit` is created under `src/segmentation`
- each segmented character is saved as `digit_0.png`, `digit_1.png`, and so on

### 7.4 Step 2: run digit prediction

Return to the project root:

```powershell
cd C:\Users\LENOVO\COS30018-HNRS-OptionB
```

Run:

```powershell
python src/model/test_model.py
```

Expected console output:

- file name of each segmented image
- predicted digit
- confidence percentage

### 7.5 Notes for Demo A

- this path works only for digit classification
- it does not assemble the final mathematical expression
- it does not compute the final answer of the expression

## 8. Demo B: expression segmentation and character classification

This path uses:

- `src/segmentation/main_extension.py`
- `src/segmentation/operator_classifier.py`
- `src/segmentation/expression_parser.py`

### 8.1 Goal

Take an input handwritten expression image, segment all characters, classify them as digits or operators, build the expression string, and show a visualization.

### 8.2 Important limitation before running

This path requires a trained PyTorch model file:

- `model_combined.pth`

That file is not currently included in the repository. You must train it first.

### 8.3 Step 1: train the PyTorch classifier

Move to the segmentation folder:

```powershell
cd C:\Users\LENOVO\COS30018-HNRS-OptionB\src\segmentation
```

Run:

```powershell
python operator_classifier.py
```

This script will:

- generate synthetic operator images if they do not exist yet
- load MNIST data
- combine digit and operator data
- train a CNN classifier for 16 classes
- save the trained weights to `model_combined.pth`

### 8.4 Step 2: run the expression pipeline

After training finishes:

```powershell
python main_extension.py input_image/test.jpg
```

You can also run:

```powershell
python main_extension.py input_image/test2.jpg
python main_extension.py input_image/test3.jpg
python main_extension.py input_image/test4.jpg
```

Expected result:

- console logs for segmentation and classification
- predicted characters for each ROI
- matplotlib window showing:
	- original image with boxes
	- thresholded image
	- segmented ROIs
	- recognized expression text

### 8.5 Current limitation of Demo B

`src/segmentation/expression_parser.py` currently validates the expression format but does not fully compute the final mathematical result.

That means:

- expression recognition can run
- expression validation can run
- final computed answer may still be missing or shown as an error

## 9. Evaluation module

The evaluation module is implemented in:

- `src/evaluation/evaluation.py`

It supports the following metrics:

- Accuracy
- Precision
- Recall
- F1-score

Default aggregation mode for multi-class classification:

- `macro`

### 9.1 Example usage

```python
from evaluation.evaluation import evaluate_results, format_evaluation

y_true = [0, 1, 2, 2, 1, 0]
y_pred = [0, 1, 2, 1, 1, 0]

result = evaluate_results(y_true, y_pred)
print(format_evaluation(result))
```

### 9.2 Output fields

The result includes:

- `accuracy`
- `precision`
- `recall`
- `f1_score`
- `average`
- `labels`
- `confusion_matrix`
- `per_class`
- `total_samples`

## 10. Known repository limitations

The following limitations should be understood before trying to extend or demo the whole project.

### 10.1 `src/main.py` is not the actual runnable entry point

`src/main.py` imports modules that are not present in the current repository structure. It should be treated as outdated or unfinished.

### 10.2 Preprocessing module is incomplete

`src/preprocessing/preprocessing.py` is currently a placeholder.

### 10.3 Evaluation integration is not yet wired into the full pipeline

`src/evaluation/evaluation.py` exists and works as a standalone evaluation utility, but it is not yet automatically called from the main prediction scripts.

### 10.4 Output folder path depends on the working directory

`src/segmentation/segmentation.py` writes segmented images to a relative folder named `output_digit`.

For this reason, the recommended working directory for segmentation commands is:

```text
src/segmentation
```

## 11. Recommended demo order

If you need to present the project in a stable way, use this order:

1. Set up Python 3.10 virtual environment.
2. Install dependencies.
3. Run Demo A first to show segmentation and digit recognition.
4. Run Demo B only if the PyTorch model has been trained successfully.
5. Explain that final symbolic expression evaluation is still incomplete in the current repository state.

## 12. Quick command summary

### Setup

```powershell
cd C:\Users\LENOVO\COS30018-HNRS-OptionB
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision keras
```

### Demo A

```powershell
cd C:\Users\LENOVO\COS30018-HNRS-OptionB\src\segmentation
python -c "from segmentation import segment_image; segment_image('input_image/test.jpg')"

cd C:\Users\LENOVO\COS30018-HNRS-OptionB
python src/model/test_model.py
```

### Demo B

```powershell
cd C:\Users\LENOVO\COS30018-HNRS-OptionB\src\segmentation
python operator_classifier.py
python main_extension.py input_image/test.jpg
```
