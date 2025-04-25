import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC, SVR

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