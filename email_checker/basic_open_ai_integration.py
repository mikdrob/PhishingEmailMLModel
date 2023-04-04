from model.ml_regression_model_v1 import EmailClassifier

if __name__ == "__main__":
    # Create an instance of the EmailClassifier class
    ec = EmailClassifier()

    # Load the dataset
    ec.load_dataset("../dataset/emails_set.csv")

    # Split the dataset into training and test sets
    ec.split_dataset(test_size=0.3, random_state=42)

    # Extract features from the text data
    ec.extract_features()

    # Train the logistic regression model
    ec.train_model()

    # Evaluate the model's accuracy on the test set
    ec.evaluate_model()

    # Print misclassified emails
    ec.print_misclassified_emails()

    # Evaluate the model using the API
    ec.evaluate_with_api()
