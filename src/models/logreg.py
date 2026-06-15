import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def load_snapshots(path):
    data = np.load(path)
    X, y, T = data['X'], data['y'] , data['T']
    return X, y, T

def flatten_snapshots(X):
    sh = X.shape
    return X.reshape(-1, sh[1] * sh[2])

def main():
    dataset_path = "data/ising_snapshots.npz"
    X, y, T = load_snapshots(dataset_path)
    X = flatten_snapshots(X)

    X_train, X_test, y_train, y_test, T_train, T_test = train_test_split(X, y, T, test_size=0.05, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    clas_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy = {acc}")
    print(f"F1-score = {f1}")
    print(f"Classification report: \n{clas_report}")
    print(f"Confusion matrix: \n{conf_matrix}")

    wrong_mask = y_pred != y_test
    print(T_test[wrong_mask])

if __name__ == '__main__':
    main()