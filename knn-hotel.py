import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('hotel_bookings.csv')

df = df.drop(columns=['reservation_status', 'reservation_status_date', 'country'])

df['children'] = df['children'].fillna(0)
df['agent'] = df['agent'].fillna(0)
df['company'] = df['company'].fillna(0)

df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=['is_canceled'])
y = df['is_canceled']

X_train_hotel, X_test_hotel, y_train_hotel, y_test_hotel = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_amostra, _, y_amostra, _ = train_test_split(
    X_train_hotel,
    y_train_hotel,
    train_size=10000,
    random_state=42,
    stratify=y_train_hotel
)

pipeline_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

grid_knn = GridSearchCV(
    pipeline_knn,
    {'knn__n_neighbors': [3, 5, 7, 11, 15, 21]},
    cv=5,
    scoring='f1',
    n_jobs=-1
)

print("\nKNN")
grid_knn.fit(X_amostra, y_amostra)
best_model_knn = grid_knn.best_estimator_

print("Melhores parâmetros:", grid_knn.best_params_)

y_pred_knn_treino = best_model_knn.predict(X_amostra)

print("Acurácia treino:", accuracy_score(y_amostra, y_pred_knn_treino))
print("Precisão treino:", precision_score(y_amostra, y_pred_knn_treino))
print("Revocação treino:", recall_score(y_amostra, y_pred_knn_treino))
print("F1 treino:", f1_score(y_amostra, y_pred_knn_treino))
print("\nMatriz de confusão treino:")
print(confusion_matrix(y_amostra, y_pred_knn_treino))
print("\nRelatório de classificação treino:")
print(classification_report(y_amostra, y_pred_knn_treino))

y_pred_knn = best_model_knn.predict(X_test_hotel)

print("Acurácia teste:", accuracy_score(y_test_hotel, y_pred_knn))
print("Precisão teste:", precision_score(y_test_hotel, y_pred_knn))
print("Revocação teste:", recall_score(y_test_hotel, y_pred_knn))
print("F1 teste:", f1_score(y_test_hotel, y_pred_knn))
print("\nMatriz de confusão teste:")
print(confusion_matrix(y_test_hotel, y_pred_knn))
print("\nRelatório de classificação teste:")
print(classification_report(y_test_hotel, y_pred_knn))




scaler_viz = StandardScaler()
X_amostra_scaled = scaler_viz.fit_transform(X_amostra)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_amostra_scaled)

knn_viz = KNeighborsClassifier(n_neighbors=grid_knn.best_params_['knn__n_neighbors'])
knn_viz.fit(X_pca, y_amostra)

x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
Z = knn_viz.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.25, cmap='RdBu')
plt.contour(xx, yy, Z, colors='k', linewidths=0.8)
plt.scatter(X_pca[y_amostra == 0, 0], X_pca[y_amostra == 0, 1], label='Não cancelado', alpha=0.3, s=10)
plt.scatter(X_pca[y_amostra == 1, 0], X_pca[y_amostra == 1, 1], label='Cancelado', alpha=0.3, s=10)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'KNN (k={grid_knn.best_params_["knn__n_neighbors"]}) - Fronteira de decisão (PCA 2D)')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
