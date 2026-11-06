
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
#navalla suissa per gràfiques
if __name__ == '__main__':

    digits = sklearn.datasets.load_digits()
    #print(digits.DESCR)
    X = digits.data
    Y = digits.target
    #print(X.shape, Y.shape)
    #el primer número és un 0
    plt.imshow( np.reshape(X[0], [8,8]), cmap="plasma")
    plt.show()

    plt.imshow(np.reshape(np.mean(X[Y==8], axis=0), [8,8]), cmap="plasma")
    plt.show()

    fig, axes = plt.subplots(3,4, figsize=(10,8))

    # flattening to 1D
    axes = axes.ravel()

    for digit in range(10):
        ax = axes[digit]
        ax.imshow(np.reshape(np.mean(X[Y == digit], axis=0), [8, 8]), cmap="plasma")
        ax.set_title(f"Digit {digit} MEAN")
        ax.axis('off')

    for i in range(10, 12):
        axes[i].axis('off')
    #mitjana
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(3,4, figsize=(10,8))

    # flattening to 1D
    axes = axes.ravel()

    for digit in range(10):
        ax = axes[digit]
        ax.imshow(np.reshape(np.std(X[Y == digit], axis=0), [8, 8]), cmap="plasma")
        ax.set_title(f"Digit {digit} MEAN")
        ax.axis('off')

    for i in range(10, 12):
        axes[i].axis('off')
    #maxima variabilitat std
    plt.tight_layout()
    plt.show()

    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
        X,
        Y,
        test_size=0.3,
        random_state=7,
        stratify=Y)

    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    #stratificacio
    for digit in range(10):
        print(f"Nomnbre d'imatges amb el digit {digit} es: {np.sum(Y == digit)}")
    for digit in range(10):
        print(f"Nomnbre d'imatges de train amb el digit {digit} es: {np.sum(Y_train == digit)}")
    for digit in range(10):
        print(f"Nomnbre d'imatges de test amb el digit {digit} es: {np.sum(Y_test == digit)}")

    #normalitzacio -> mitjana a 0
    #normalitzacio Z score
    # (X-u)/sd
    u_train = np.mean(X_train, axis=0)
    sd_train = np.std(X_train, axis=0)

    print(u_train, sd_train)
    X_train_standard_scaler = (X_train - u_train) / sd_train #z-score

    X_train_standard_scaler[np.where(np.isnan(X_train_standard_scaler))] = 0
    #nans -> 0
    print(X_train_standard_scaler)
    print(np.round(np.mean(X_train_standard_scaler, axis=0), 3))
    print(np.std(X_train_standard_scaler, axis=0))

    scaler= sklearn.preprocessing.StandardScaler().fit(X_train_standard_scaler)
    X_train_standard_scaler_sklearn = scaler.fit_transform(X_train)

    print(np.round(np.mean(X_train_standard_scaler_sklearn, axis=0), 3))
    print(np.round(np.std(X_train_standard_scaler_sklearn, axis=0), 3))

    #PCA en 2 dimensions. memoria: dif entre PCA i SVD
    n_dim = 2
    pca=sklearn.decomposition.PCA(n_components=n_dim)
    X_train_pca = pca.fit_transform(X_train)
    print(X_train_pca)

    svd=sklearn.decomposition.TruncatedSVD(n_components=n_dim)
    X_train_svd = svd.fit_transform(X_train)
    print(X_train_svd)

    pca=sklearn.decomposition.PCA(n_components=n_dim)
    X_train_pca_ss = pca.fit_transform(X_train_standard_scaler_sklearn)
    print(X_train_pca_ss)

    svd=sklearn.decomposition.TruncatedSVD(n_components=n_dim)
    X_train_svd_ss = svd.fit_transform(X_train_standard_scaler_sklearn)
    print(X_train_svd_ss)

    fig, axes = plt.subplots(2,2, figsize=(10,12))
    axes[0,0].scatter(X_train_pca[:,0], X_train_pca[:,1], c=Y_train, cmap="plasma")
    axes[0,0].set_title('PCA')
    axes[0,1].scatter(X_train_svd[:,0], X_train_svd[:,1], c=Y_train, cmap="plasma")
    axes[0,1].set_title('SVD')
    axes[1,0].scatter(X_train_pca_ss[:,0], X_train_pca_ss[:,1], c=Y_train, cmap="plasma")
    axes[1,0].set_title('PCA standard scaler')
    axes[1,1].scatter(X_train_svd_ss[:,0], X_train_svd_ss[:,1], c=Y_train, cmap="plasma")
    axes[1,1].set_title('SVD standard scaler')

    plt.show()
    #PCA en 3 dimensions

    n_dim = 3
    pca=sklearn.decomposition.PCA(n_components=n_dim)
    X_train_pca = pca.fit_transform(X_train)
    print(X_train_pca)

    svd=sklearn.decomposition.TruncatedSVD(n_components=n_dim)
    X_train_svd = svd.fit_transform(X_train)
    print(X_train_svd)

    pca=sklearn.decomposition.PCA(n_components=n_dim)
    X_train_pca_ss = pca.fit_transform(X_train_standard_scaler_sklearn)
    print(X_train_pca_ss)

    svd=sklearn.decomposition.TruncatedSVD(n_components=n_dim)
    X_train_svd_ss = svd.fit_transform(X_train_standard_scaler_sklearn)
    print(X_train_svd_ss)

    fig, axes = plt.subplots(2,2, figsize=(10,12), subplot_kw={"projection":"3d"})
    axes[0,0].scatter(X_train_pca[:,0], X_train_pca[:,1], X_train_pca[:,2], c=Y_train, cmap="plasma")
    axes[0,0].set_title('PCA')
    axes[0,1].scatter(X_train_svd[:,0], X_train_svd[:,1], X_train_svd[:,2], c=Y_train, cmap="plasma")
    axes[0,1].set_title('SVD')
    axes[1,0].scatter(X_train_pca_ss[:,0], X_train_pca_ss[:,1], X_train_pca_ss[:,2], c=Y_train, cmap="plasma")
    axes[1,0].set_title('PCA standard scaler')
    axes[1,1].scatter(X_train_svd_ss[:,0], X_train_svd_ss[:,1], X_train_svd_ss[:,2], c=Y_train, cmap="plasma")
    axes[1,1].set_title('SVD standard scaler')

    plt.show()

    #quadratic discriminal  analisis

    #linear discriminal  analisis
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=3)
    X_train_lda = lda.fit_transform(X_train, Y_train)

    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=3)
    X_train_lda_ss = lda.fit_transform(X_train_standard_scaler_sklearn, Y_train)

    print(X_train_lda)
    print(X_train_lda_ss)

    fig, axes = plt.subplots(1,2, figsize=(12,6), subplot_kw={"projection":"3d"})
    axes[0].scatter(X_train_lda[:,0], X_train_lda[:,1], X_train_lda[:,2], c=Y_train, cmap="plasma")
    axes[0].set_title('LDA')
    axes[1].scatter(X_train_lda_ss[:,0], X_train_lda_ss[:,1], X_train_lda_ss[:,2], c=Y_train, cmap="plasma")
    axes[1].set_title('LDA standard scaler')

    plt.show()

    #classificador

    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform')
    clf.fit(X_train, Y_train)
    #clf.fit(X_train_pca, Y_train)
    accuracy = np.round( (np.sum(clf.predict(X_test)==Y_test) / len(Y_test))*100, 2)
    print(accuracy)

    print(sklearn.metrics.classification_report(Y_test, clf.predict(X_test)))

    #variables lliures: neighbours, z-scores, n-components, pca, svd, lda, weights

    def compute_test (x, y, clfr, cv=10):
        KFold = sklearn.model_selection.KFold(n_splits=cv)
        scores = []
        for train, test in KFold.split(x):
            clfr.fit(x[train], y[train])
            pred_values = clfr.predict(X[test])
            scores.append(sklearn.metrics.accuracy_score(pred_values, y[test]))
        return scores

    results = compute_test(X, Y, sklearn.neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform'), 10)
    print(results)

    #pipeline -> moduls

    clf = sklearn.pipeline.Pipeline([
        ('scaler', sklearn.preprocessing.StandardScaler()),
        ('lda', sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=9)),
        ('knn', sklearn.neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform'))]
    )

    results = compute_test(X, Y, clf, 10)
    print(results)

    #falta posar grafiques
    #GridSearchCV
    #quina es la millor combinacio
    #optimitzar els fors -> grid search ->primera part de la pràctica 3.
    # distancia A -> B != B -> A. aixo es extrany
    # P(plou -> paraigues) != P(paraigues -> plou)
