import argparse

from preprocessing.preprocess_pipeline import preprocess_image
from segmentation.segment_digits import segment_digits
from models.predict import predict_digit
from evaluation.evaluate import evaluate_results


def run_pipeline(image_path):
    # Task 1: Image Preprocessing
    preprocessed_image = preprocess_image(image_path)

    # Task 2: Image Segmentation
    digit_images = segment_digits(preprocessed_image)

    # Task 3: ML Model Prediction
    predictions = []
    for digit_img in digit_images:
        pred = predict_digit(digit_img)
        predictions.append(str(pred))

    recognized_number = "".join(predictions)
    return recognized_number


def main():
    parser = argparse.ArgumentParser(description="Handwritten Number Recognition System")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image"
    )
    args = parser.parse_args()

    result = run_pipeline(args.input)
    print("Recognized number:", result)

    # Task 4: Evaluation (optional for demo run)
    # evaluate_results()


if __name__ == "__main__":
    main()
