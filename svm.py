import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

plt.style.use("ggplot")

df = pd.read_csv("train.csv")
# =================================================
# Inciso 2: Exploracion de datos para transformacion
# =================================================
"""
print(df.head())
print(df.info())
print(df.describe())
#EDA
#var num relacion
num_vars = ["SalePrice", "LotArea", "GrLivArea", "TotalBsmtSF", "OverallQual"]
sb.pairplot(df[num_vars])
plt.suptitle("Relacion entre Variables Numericas", y=1.001)
plt.show()
#dist saleprice
plt.figure(figsize=(10, 6))
sb.histplot(df["SalePrice"], kde=True, bins=50)
plt.title("Distribucion de SalePrice")
plt.xlabel("Precio de Venta (USD)")
plt.show()
#correlacion saleprice
corr_matrix = df[num_vars].corr()
plt.figure(figsize=(10, 8))
sb.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Matriz de Correlacion")
plt.show()
""" 
#"limpieza"
# columnas con mas de 30% nulos
null_percent = df.isnull().mean()*100
high_null_cols = null_percent[null_percent>30].index
df_clean = df.drop(columns=high_null_cols)

df_clean["PriceCategory"] = pd.qcut(df_clean["SalePrice"], q=3, labels=["barata", "media", "cara"])

X = df_clean.drop(columns=["SalePrice", "Id", "PriceCategory"])
y = df_clean["PriceCategory"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Columnas numericas y categoricas
num_col = X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_col = X.select_dtypes(include=["object"]).columns.tolist()

# Pipelines de transformacion
numeric_transformer = Pipeline([
    ("imputer", IterativeImputer(random_state=42)),  # Imputación multivariable
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Preprocesador
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_col),
    ("cat", categorical_transformer, cat_col)
])
# =================================================
# Inciso 4: Varios modelos de SVM
# =================================================
#Modelo 1: kernel lineal
svm_linear = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="linear", random_state=42))
])
#Modelo 2: Kernel RBF
svm_rbf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="rbf", random_state=42))
])
#Modelo 3: kernel polinomial
svm_polinomial = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="poly", random_state=42))
])

param_grids = [
    {  #linal
        "classifier__C": [0.1, 1, 10],
        "classifier__kernel": ["linear"]
    },
    {  #rbf
        "classifier__C": [0.1, 1, 10, 100],
        "classifier__gamma": ["scale", "auto", 0.1, 1],
        "classifier__kernel": ["rbf"]
    },
    {  #polinomial
        "classifier__C": [0.1, 1, 10],
        "classifier__degree": [2, 3, 4],
        "classifier__gamma": ["scale", "auto"],
        "classifier__kernel": ["poly"]
    }
]

models = [svm_linear, svm_rbf, svm_polinomial]
best_models = []

for i, model in enumerate(models):
    print(f"\nEntrenando modelo {i+1} ({model.named_steps["classifier"].kernel} kernel)")
    grid_search = GridSearchCV(
        model,
        param_grids[i],
        cv=5,
        scoring="accuracy",
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_models.append(grid_search.best_estimator_)
    
    y_pred = grid_search.best_estimator_.predict(X_test)
    print(f"\nMejores parametros para modelo {i+1}:")
    print(grid_search.best_params_)
    print("\nReporte de clasificacion:")
    print(classification_report(y_test, y_pred))
    print("\nMatriz de confusion:")
    print(confusion_matrix(y_test, y_pred))

#comparacion final
print("\nComparacion final:")
for i, model in enumerate(best_models):
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Modelo {i+1} ({model.named_steps["classifier"].kernel} kernel): Accuracy = {acc:.4f}")

best_model_idx = np.argmax([accuracy_score(y_test, m.predict(X_test)) for m in best_models])
best_model = best_models[best_model_idx]
print(f"\nMejor modelo: {best_model.named_steps["classifier"].kernel} kernel")

#preddccion vs reales
y_pred_best = best_model.predict(X_test)
categorias = ["barata", "media", "cara"]

plt.figure(figsize=(10, 6))
sb.scatterplot(x=y_test, y=y_pred_best, hue=y_test, palette="viridis", s=100)
plt.xticks(ticks=range(len(categorias)), labels=categorias)
plt.yticks(ticks=range(len(categorias)), labels=categorias)
plt.xlabel("Categoria Real", fontsize=12)
plt.ylabel("Categoria Prediccion", fontsize=12)
plt.title(f"Resultados de Mejor Modelo (Kernel: {best_model.named_steps["classifier"].kernel})", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

# =================================================
# 5. Valor de la variable respuesta (Según yo se responde arriba. Esto es para después realmente)
# =================================================
predictions = {name: model.predict(X_test) for name, model in zip(model_names, best_models)}

# =================================================
# 6. Matrices de confusión
# =================================================
for name in model_names:
    cm = confusion_matrix(y_test, predictions[name], labels=categories)
    print(f"\nMatriz de confusión – Modelo {name}:")
    print(pd.DataFrame(
        cm,
        index=[f"real: {c}" for c in categories],
        columns=[f"pred: {c}" for c in categories]
    ))
    plt.figure(figsize=(6, 4))
    sb.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=categories,
        yticklabels=categories,
        cmap="Blues"
    )
    plt.title(f"Confusión – {name}")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()
# =================================================
# 7. Comparación train vs test para ver si está sobreajustado o no
# =================================================
print("\nComparación de Accuracy (train vs test):")
for name, model in zip(model_names, best_models):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test, predictions[name])
    print(f"{name}: train_acc = {train_acc:.4f}, test_acc = {test_acc:.4f}")

# ================================
# 8. Estadísticas SVM – efectividad, tiempo y errores
# ================================
svm_stats = []
for name, m in zip(model_names, best_models):
    t0 = time.time()
    m.fit(X_train, y_train)
    train_time = time.time() - t0

    t1 = time.time()
    y_pred = m.predict(X_test)
    pred_time = time.time() - t1

    cm = confusion_matrix(y_test, y_pred, labels=categories)
    mistakes = {categories[i]: int(cm[i].sum() - cm[i, i]) for i in range(3)}

    svm_stats.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "TrainTime_s": train_time,
        "PredTime_s": pred_time,
        "Mistakes": mistakes
    })

print("\n=== Inciso 8: Estadísticas SVM ===")
print(pd.DataFrame(svm_stats))

# ================================
# 9. Comparación con otros clasificadores
# ================================
other_classifiers = {
    "DecisionTree": Pipeline([("pre", preprocessor),
                              ("clf", DecisionTreeClassifier(random_state=42))]),
    "RandomForest": Pipeline([("pre", preprocessor),
                              ("clf", RandomForestClassifier(random_state=42))]),
    "NaiveBayes":   Pipeline([("pre", preprocessor),
                              ("clf", GaussianNB())]),
    "KNN":          Pipeline([("pre", preprocessor),
                              ("clf", KNeighborsClassifier())]),
    "Logistic":     Pipeline([("pre", preprocessor),
                              ("clf", LogisticRegression(max_iter=1000, random_state=42))])
}
other_stats = []
for name, pipe in other_classifiers.items():
    t0 = time.time()
    pipe.fit(X_train, y_train)
    tt = time.time() - t0

    t1 = time.time()
    y_pred = pipe.predict(X_test)
    pt = time.time() - t1

    other_stats.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "TrainTime_s": tt,
        "PredTime_s": pt
    })

print("\n=== Inciso 9: Otros clasificadores ===")
print(pd.DataFrame(other_stats))

# ================================
# 10. Modelo de regresión sobre SalePrice. Wooo
# ================================
y_reg = df_clean["SalePrice"]
X_tr_reg, X_te_reg, y_tr_reg, y_te_reg = train_test_split(
    X, y_reg, test_size=0.3, random_state=42
)

reg_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", SVR())
])
param_grid_reg = {
    "regressor__kernel": ["linear", "rbf"],
    "regressor__C": [0.1, 1, 10],
    "regressor__gamma": ["scale", "auto"]
}

grid_reg = GridSearchCV(
    reg_pipeline,
    param_grid_reg,
    cv=5,
    scoring="neg_root_mean_squared_error",
    verbose=1,
    n_jobs=-1
)
grid_reg.fit(X_tr_reg, y_tr_reg)

best_reg = grid_reg.best_estimator_
y_pred_reg = best_reg.predict(X_te_reg)

# Cálculo manual de RMSE, porque ya ni modo
mse = mean_squared_error(y_te_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2   = r2_score(y_te_reg, y_pred_reg)
mae = mean_absolute_error(y_te_reg, y_pred_reg) 

print("\n=== Inciso 10: SVR Regresión ===")
print("Mejores parámetros:", grid_reg.best_params_)
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}") 
print(f"R²:    {r2:.4f}")
