import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


iris_df = pd.read_csv('Iris.csv')
iris_df = iris_df.drop(columns=['Id'])

print(iris_df.describe())

X_iris = iris_df.drop(columns=['Species'])
y_iris = pd.Categorical(iris_df['Species']).codes

X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris,
    y_iris,
    test_size=0.2,
    random_state=42,
    stratify=y_iris
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(class_weight='balanced'))
])

grid_linear = GridSearchCV(
    pipeline, {
        'svc__kernel': ['linear'], 
        'svc__C': [0.001, 0.1, 1, 10]}, 
        cv=5, 
        scoring='f1_weighted', 
        n_jobs=-1
)
grid_gausiano = GridSearchCV(
    pipeline, {
        'svc__kernel': ['rbf'], 
        'svc__C': [0.1, 1, 10], 
        'svc__gamma': [0.01, 0.1, 1]}, 
        cv=5, 
        scoring='f1_weighted', 
        n_jobs=-1
)

print("\nLinear")
grid_linear.fit(X_train_iris, y_train_iris)
best_model_linear = grid_linear.best_estimator_

print("Melhores parâmetros:", grid_linear.best_params_)

y_pred_linear = best_model_linear.predict(X_test_iris)

print("Acurácia:", accuracy_score(y_test_iris, y_pred_linear))
print("Precisão:", precision_score(y_test_iris, y_pred_linear, average='weighted'))
print("Revocação:", recall_score(y_test_iris, y_pred_linear, average='weighted'))
print("F1:", f1_score(y_test_iris, y_pred_linear, average='weighted'))
print("\nMatriz de confusão:")
print(confusion_matrix(y_test_iris, y_pred_linear))
print("\nRelatório de classificação:")
print(classification_report(y_test_iris, y_pred_linear))


print("\nGausiano")
grid_gausiano.fit(X_train_iris, y_train_iris)
best_model_gausiano = grid_gausiano.best_estimator_

print("Melhores parâmetros:", grid_gausiano.best_params_)

y_pred_gausiano = best_model_gausiano.predict(X_test_iris)

print("Acurácia:", accuracy_score(y_test_iris, y_pred_gausiano))
print("Precisão:", precision_score(y_test_iris, y_pred_gausiano, average='weighted'))
print("Revocação:", recall_score(y_test_iris, y_pred_gausiano, average='weighted'))
print("F1:", f1_score(y_test_iris, y_pred_gausiano, average='weighted'))
print("\nMatriz de confusão:")
print(confusion_matrix(y_test_iris, y_pred_gausiano))
print("\nRelatório de classificação:")
print(classification_report(y_test_iris, y_pred_gausiano))




scaler_viz = StandardScaler()
X_train_scaled = scaler_viz.fit_transform(X_train_iris)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

svm_linear_viz = SVC(kernel='linear', C=grid_linear.best_params_['svc__C'], class_weight='balanced')
svm_linear_viz.fit(X_pca, y_train_iris)

x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
Z = svm_linear_viz.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.25, cmap='RdYlBu')
plt.contour(xx, yy, Z, colors='k', linewidths=0.8)
plt.scatter(X_pca[y_train_iris == 0, 0], X_pca[y_train_iris == 0, 1], label='Setosa', alpha=0.6, s=30)
plt.scatter(X_pca[y_train_iris == 1, 0], X_pca[y_train_iris == 1, 1], label='Versicolor', alpha=0.6, s=30)
plt.scatter(X_pca[y_train_iris == 2, 0], X_pca[y_train_iris == 2, 1], label='Virginica', alpha=0.6, s=30)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'SVM Linear (C={grid_linear.best_params_["svc__C"]}) - Iris - Fronteira de decisão (PCA 2D)')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


svm_rbf_viz = SVC(kernel='rbf', C=grid_gausiano.best_params_['svc__C'], gamma=grid_gausiano.best_params_['svc__gamma'], class_weight='balanced')
svm_rbf_viz.fit(X_pca, y_train_iris)

Z = svm_rbf_viz.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.25, cmap='RdYlBu')
plt.contour(xx, yy, Z, colors='k', linewidths=0.8)
plt.scatter(X_pca[y_train_iris == 0, 0], X_pca[y_train_iris == 0, 1], label='Setosa', alpha=0.6, s=30)
plt.scatter(X_pca[y_train_iris == 1, 0], X_pca[y_train_iris == 1, 1], label='Versicolor', alpha=0.6, s=30)
plt.scatter(X_pca[y_train_iris == 2, 0], X_pca[y_train_iris == 2, 1], label='Virginica', alpha=0.6, s=30)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'SVM Gaussiano (C={grid_gausiano.best_params_["svc__C"]}, γ={grid_gausiano.best_params_["svc__gamma"]}) - Iris - Fronteira de decisão (PCA 2D)')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
