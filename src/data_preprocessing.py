import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath=\"data/iris.csv\"):
    """Loads the Iris dataset and performs basic preprocessing."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Data file not found at {filepath}. Creating a dummy file.")
        # Create a dummy iris.csv for demonstration if not found
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df["target"] = iris.target
        df.to_csv(filepath, index=False)
        df = pd.read_csv(filepath)

    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == \"__main__\":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print("Data preprocessing complete.")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
