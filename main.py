import numpy as np
import sklearn
import sklearn.datasets
import sklearn.model_selection
import sklearn.decomposition
import sklearn.neighbors
import sklearn.metrics
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.discriminant_analysis
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold

    # els primers prints
def show_some_digit(x, y):
    plt.imshow(np.reshape(x[0], [8, 8]), cmap="plasma")
    plt.show()
    print("plot of first digit")
    plt.imshow(np.reshape(np.mean(x[y == 8], axis=0), [8, 8]), cmap="plasma")
    plt.show()
    print("plot of digit 8")

    # mitjanes dels digits
def show_mean_digits(x, y):
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))

    # flattening to 1D
    axes = axes.ravel()

    for digit in range(10):
        ax = axes[digit]
        ax.imshow(np.reshape(np.mean(x[y == digit], axis=0), [8, 8]), cmap="plasma")
        ax.set_title(f"Digit {digit} MEAN")
        ax.axis('off')

    for i in range(10, 12):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    # maxima variabilitat std
def show_std_var_digits(x, y):
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))

    # flattening to 1D
    axes = axes.ravel()

    for digit in range(10):
        ax = axes[digit]
        ax.imshow(np.reshape(np.std(x[y == digit], axis=0), [8, 8]), cmap="plasma")
        ax.set_title(f"Digit {digit} MEAN")
        ax.axis('off')

    for i in range(10, 12):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    # stratificacio
def stratification (y, y_train, y_test):

    for digit in range(10):
        print(f"Nomnbre d'imatges amb el digit {digit} es: {np.sum(y == digit)}")
    for digit in range(10):
        print(f"Nomnbre d'imatges de train amb el digit {digit} es: {np.sum(y_train == digit)}")
    for digit in range(10):
        print(f"Nomnbre d'imatges de test amb el digit {digit} es: {np.sum(y_test == digit)}")

    # normalitzacio -> mitjana a 0
    # normalitzacio Z score
    # (X-u)/sd
def normalization_z_score (x_train):
    u_train = np.mean(x_train, axis=0)
    sd_train = np.std(x_train, axis=0)

    print(u_train, sd_train)
    x_train_standard_scaler = (x_train - u_train) / sd_train  # z-score

    x_train_standard_scaler[np.where(np.isnan(x_train_standard_scaler))] = 0
    # nans -> 0
    scaler = sklearn.preprocessing.StandardScaler().fit(x_train_standard_scaler)
    x_train_standard_scaler_sklearn = scaler.fit_transform(x_train)

    print(np.round(np.mean(x_train_standard_scaler_sklearn, axis=0), 3))
    print(np.round(np.std(x_train_standard_scaler_sklearn, axis=0), 3))
    return x_train_standard_scaler_sklearn

    # PCA en 2 i 3dimensions.
def PCA_tests(x_train, x_train_standard_scaler_sklearn):

    for n_dim in range(2, 4):
        pca = sklearn.decomposition.PCA(n_components=n_dim)
        x_train_pca = pca.fit_transform(x_train)
        print(x_train_pca)

        pca = sklearn.decomposition.PCA(n_components=n_dim)
        x_train_pca_ss = pca.fit_transform(x_train_standard_scaler_sklearn)
        print(x_train_pca_ss)

        fig, axes = plt.subplots(1, 2, figsize=(10, 12), subplot_kw={"projection": "3d"})
        axes[0].scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=Y_train, cmap="plasma")
        axes[0].set_title('PCA')
        axes[1].scatter(x_train_pca_ss[:, 0], x_train_pca_ss[:, 1], c=Y_train, cmap="plasma")
        axes[1].set_title('PCA standard scaler')

        plt.show()

def SVD_tests(x_train, x_train_standard_scaler_sklearn):

    for n_dim in range(2, 4):
        svd = sklearn.decomposition.TruncatedSVD(n_components=n_dim)
        x_train_svd = svd.fit_transform(x_train)
        print(x_train_svd)

        svd = sklearn.decomposition.TruncatedSVD(n_components=n_dim)
        x_train_svd_ss = svd.fit_transform(x_train_standard_scaler_sklearn)
        print(x_train_svd_ss)

        fig, axes = plt.subplots(2, 2, figsize=(10, 12), subplot_kw={"projection": "3d"})

        axes[0, 1].scatter(x_train_svd[:, 0], x_train_svd[:, 1], c=Y_train, cmap="plasma")
        axes[0, 1].set_title('SVD')

        axes[1, 1].scatter(x_train_svd_ss[:, 0], x_train_svd_ss[:, 1], c=Y_train, cmap="plasma")
        axes[1, 1].set_title('SVD standard scaler')

        plt.show()

    # quadratic discriminal analisis -> later
    # linear discriminal analisis
def LDA_tests(x_train, x_train_standard_scaler_sklearn, y_train):

    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=3)
    x_train_lda = lda.fit_transform(x_train, y_train)

    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=3)
    x_train_lda_ss = lda.fit_transform(x_train_standard_scaler_sklearn, y_train)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"projection": "3d"})
    axes[0].scatter(x_train_lda[:, 0], x_train_lda[:, 1], x_train_lda[:, 2], c=y_train, cmap="plasma")
    axes[0].set_title('LDA')
    axes[1].scatter(x_train_lda_ss[:, 0], x_train_lda_ss[:, 1], x_train_lda_ss[:, 2], c=y_train, cmap="plasma")
    axes[1].set_title('LDA standard scaler')

    plt.show()

    # classificador

def compute_test(x, y, clfr, cv=10):
    KFold = sklearn.model_selection.KFold(n_splits=cv)
    scores = []
    for train, test in KFold.split(x):
        clfr.fit(x[train], y[train])
        pred_values = clfr.predict(X[test])
        scores.append(sklearn.metrics.accuracy_score(pred_values, y[test]))
    return scores

def classificator_test (x_test, y_test):

    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform')
    clf.fit(X_train, Y_train)
    accuracy = np.round((np.sum(clf.predict(x_test) == y_test) / len(y_test)) * 100, 2)

    print(accuracy)
    print(sklearn.metrics.classification_report(Y_test, clf.predict(X_test)))

def class_tests(x, y, x_train, y_train, x_test, y_test):

    #mirar les dades
    show_some_digit(x, y)
    show_mean_digits(x, y)
    show_std_var_digits(x, y)
    stratification (y, y_train, y_test)

    #fer tests
    x_train_standard_scaler_sklearn = normalization_z_score(X_train)

    PCA_tests(x_train, x_train_standard_scaler_sklearn)
    SVD_tests(x_train, x_train_standard_scaler_sklearn)
    LDA_tests(x_train, x_train_standard_scaler_sklearn, y_train)

    classificator_test(x_test, y_test)
    results = compute_test(X, Y, sklearn.neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform'), 10)
    print(results)
    results = compute_test(X, Y, pipe_clf, 10)
    print(results)

#grid search -> treure els fors (es millor)
grid_search = [
    {
        "reduce": ["passthrough"],
        "knn__n_neighbors": [1, 3, 5, 7, 9],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2]
    },
    {
        "reduce": [sklearn.decomposition.PCA()],
        "reduce__n_components": [2, 3, 5, 10],
        "knn__n_neighbors": [1, 3, 5, 7, 9, 11, 15],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2]
    },
    {
        "reduce": [sklearn.decomposition.TruncatedSVD()],
        "reduce__n_components": [2, 5, 10, 20, 30],
        "knn__n_neighbors": [1, 3, 5, 7, 9, 11, 15],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2]
    },
    {
        "reduce": [sklearn.discriminant_analysis.LinearDiscriminantAnalysis()],
        "reduce__n_components": [2, 5, 9],
        "knn__n_neighbors": [1, 3, 5, 7, 9, 11, 15],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2]
    }
]

# pipeline -> moduls

pipe_clf = sklearn.pipeline.Pipeline([
    ("reduce", "passthrough"),
    ('scaler', sklearn.preprocessing.StandardScaler()),
    ('lda', "passthrough"),
    ("projection", "passthrough"),
    ("knn", sklearn.neighbors.KNeighborsClassifier())
])

#la funcio compute_test pero amb el format correcte

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

if __name__ == '__main__':

    digits = sklearn.datasets.load_digits()
    X = digits.data
    Y = digits.target

    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
        X,
        Y,
        test_size=0.3,
        random_state=7,
        stratify=Y)

    print("Dades extretes...")

    search = GridSearchCV(
        pipe_clf,
        grid_search,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        error_score='raise'
    )

    print("Executant algorismes...")

    #els tests que vem fer a classe
    #class_tests(X, Y, X_train, Y_train, X_test, Y_test)

    search.fit(X_train, Y_train)
    best_model = search.best_estimator_
    Y_pred = best_model.predict(X_test)
    print("Best Accuracy:",
          sklearn.metrics.accuracy_score(Y_test, Y_pred))
    print(sklearn.metrics.classification_report(Y_test, Y_pred))

    #plots ->
    results = pd.DataFrame(search.cv_results_)

    print(results.columns)

    top10 = results.sort_values(
        by='mean_test_score',
        ascending=False
    ).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(
        range(len(top10)),
        top10['mean_test_score']
    )

    ax.set_yticks(range(len(top10)))
    #ax.set_yticklabels([str(p)[:60] for p in top10['params']])

    for i, score in enumerate(top10['mean_test_score']):
        ax.text(score + 0.0001, i, f"{score:.4f}", va='center')

    ax.set_xlim(
        top10['mean_test_score'].min() - 0.002,
        top10['mean_test_score'].max() + 0.002
    )

    labels = []
    for p in top10['params']:
        labels.append(
            f"{type(p['reduce']).__name__}, "
            f"N={p['knn__n_neighbors']}, "
            f"P={p['knn__p']}, "
            f"W={p['knn__weights']}, "
            f"COM={p['reduce__n_components']}"
        )

    ax.set_yticklabels(labels)

    ax.invert_yaxis()

    plt.xlabel('Accuracy')
    plt.subplots_adjust(left=0.4)
    plt.title('Top 10 configuracions')
    plt.gca().invert_yaxis()
    plt.show()

    cm = confusion_matrix(Y_test, Y_pred, labels=range(10))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=range(10)
    )
    disp.plot(cmap="Blues")
    plt.xlabel("Dígit predit")
    plt.ylabel("Dígit real")
    plt.show()

    sns.boxplot(
        data=results,
        x='param_knn__n_neighbors',
        y='mean_test_score'
    )

    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    plt.title('Impacte de k')
    plt.show()

    print(f"Millor accuracy CV: {search.best_score_ * 100:.2f}%")
    test_accuracy = sklearn.metrics.accuracy_score(Y_test, Y_pred)
    print(f"Accuracy test: {test_accuracy * 100:.2f}%")