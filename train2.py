from sklearn.kernel_ridge import KernelRidge
from misc import load_data, preprocess_data, train_and_evaluate

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df, scale=True)

    model = KernelRidge(alpha=1.0, kernel="rbf")
    mse = train_and_evaluate(model, X_train, X_test, y_train, y_test)

    print(f"KernelRidge Test MSE: {mse:.4f}")

if __name__ == "__main__":
    main()

