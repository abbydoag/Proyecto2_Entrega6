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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR

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
null_percent = df.isnull().mean() * 100
high_null_cols = null_percent[null_percent > 30].index
df_clean = df.drop(columns=high_null_cols)

X = df_clean.drop(columns=["SalePrice","Id"])
y = df_clean["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Columnas numericas y categoricas
num_col = X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_col = X.select_dtypes(include=["object"]).columns.tolist()

# Pipelines de transformacion
numeric_transformer = Pipeline([
    ("imputer", IterativeImputer(random_state=42)),  # Imputación multivariable
    ("scaler", StandardScaler())
])

# Fix typo in "most_frequent"
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Preprocesador
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_col),
    ("cat", categorical_transformer, cat_col)
])

svm_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", SVR(kernel="rbf"))
])

param_grid = {
    "regressor__C": [0.1, 1, 10, 100],
    "regressor__kernel": ["linear", "rbf"],
    "regressor__gamma": ["scale", "auto"]
}

grid_search = GridSearchCV(
    svm_pipe,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_svm = grid_search.best_estimator_
y_pred_best = best_svm.predict(X_test)

print(f"Parametros: {grid_search.best_params_}")
print(f"RMSE: {mean_squared_error(y_test, y_pred_best, squared=False):.2f}")
print(f"R²: {r2_score(y_test, y_pred_best):.4f}")

# Pred vs real
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.5, color="royalblue")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=2)
plt.xlabel("Precio Real (USD)", fontsize=12)
plt.ylabel("Precio Predicho (USD)", fontsize=12)
plt.title("Resultados del Modelo SVM Optimizado", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.3)
plt.show()