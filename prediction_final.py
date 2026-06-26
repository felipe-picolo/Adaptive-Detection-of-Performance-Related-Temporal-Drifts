import pandas as pd
import random
import os
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from glob import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
try:
    from tensorflow.keras import Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except Exception:
    # Fallback to standalone keras if tensorflow.keras is unavailable
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.callbacks import EarlyStopping


WINDOW=30
RESULT_PATH="prediction/prediction_rul"
os.makedirs(RESULT_PATH,exist_ok=True)

def create_rul(df):

    df = df.copy()
    df["RUL"] = 0

    failures = sorted(
        df.loc[df["Equipment_Failure"] == 1, "Trace"].tolist()
    )

    last_trace = df["Trace"].max()

    # Se o último trace não é uma falha,
    # adiciona uma falha virtual no final
    if len(failures) == 0 or failures[-1] != last_trace:
        failures.append(last_trace)

    current_failure = 0

    for i, trace in enumerate(df["Trace"]):

        while (
            current_failure < len(failures) - 1 and
            trace > failures[current_failure]
        ):
            current_failure += 1

        df.at[i, "RUL"] = failures[current_failure] - trace

    return df

def create_window(df):

    cols = [
        "Trace",
        "Machine_Operating",
        "Raw_Material_Loading",
        "Short_Downtime",
        "DriftD"
    ]

    data = {}

    for col in cols:
        for i in range(WINDOW):
            data[f"{col}_t{i}"] = df[col].shift(i)

    data["RUL"] = df["RUL"]

    out = pd.DataFrame(data)

    out = out.iloc[WINDOW-1:].reset_index(drop=True)

    return out

def process_matrix(file):
    df=pd.read_csv(file,sep=";")
    df=create_rul(df)
    df=create_window(df)
    return df

def load_dataset(files):
    dfs=[]
    for file in files:
        dfs.append(process_matrix(file))
    return pd.concat(dfs,ignore_index=True)

def split_files():

    dr_ms=sorted(glob("prediction/matrix_log/DR_MS_[0-9][0-9]_matrix.csv"))
    dr_ms_st=sorted(glob("prediction/matrix_log/DR_MS_ST_[0-9][0-9]_matrix.csv"))

    dr_train,dr_test=train_test_split(
        dr_ms,
        test_size=0.30,
        random_state=42,
        shuffle=True
    )

    st_train,st_test=train_test_split(
        dr_ms_st,
        test_size=0.30,
        random_state=42,
        shuffle=True
    )

    train_files=dr_train+st_train
    test_files=dr_test+st_test

    random.shuffle(train_files)
    random.shuffle(test_files)

    return train_files,test_files

def prediction_rul_prepare():

    train_files,test_files=split_files()

    train=load_dataset(train_files)
    test=load_dataset(test_files)

    features=[c for c in train.columns if c!="RUL"]

    X_train=train[features]
    y_train=train["RUL"]

    X_test=test[features]
    y_test=test["RUL"]

    print(f"Treino: {X_train.shape}")
    print(f"Teste : {X_test.shape}")
    print(f"Nº Features: {len(features)}")

    return X_train,y_train,X_test,y_test,features


def random_forest_rul():

    X_train,y_train,X_test,y_test,features=prediction_rul_prepare()

    model=RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train,y_train)

    pred=model.predict(X_test)

    mae=mean_absolute_error(y_test,pred)
    rmse=np.sqrt(mean_squared_error(y_test,pred))
    r2=r2_score(y_test,pred)

    print("="*60)
    print("RANDOM FOREST")
    print("="*60)
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")

    with open(f"{RESULT_PATH}/rf_metrics.txt","w") as f:
        f.write(f"MAE={mae}\n")
        f.write(f"RMSE={rmse}\n")
        f.write(f"R2={r2}\n")

    result=pd.DataFrame({
        "Real":y_test,
        "Predicted":pred,
        "Error":y_test-pred
    })

    result.to_csv(
        f"{RESULT_PATH}/rf_predictions.csv",
        sep=";",
        index=False
    )

    # ======================================================
    # 1 - REAL X PREVISTO
    # ======================================================

    plt.figure(figsize=(6,6))

    plt.scatter(y_test,pred,s=10,alpha=0.6)
    lim=max(y_test.max(),pred.max())
    plt.plot([0,lim],[0,lim],"r--",linewidth=2,label="Ideal")
    plt.title("Random Forest - Actual vs Predicted RUL",fontsize=14)
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{RESULT_PATH}/rf_real_vs_pred.png",dpi=300)
    plt.close()

    # ======================================================
    # 2 - CURVA RUL
    # ======================================================

    plt.figure(figsize=(14,5))
    plt.plot(y_test.values,label="Actual",linewidth=2)
    plt.plot(pred,label="Predicted",linewidth=2)
    plt.title("Random Forest - RUL Prediction",fontsize=14)
    plt.xlabel("Sample")
    plt.ylabel("Remaining Useful Life")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{RESULT_PATH}/rf_rul_curve.png",dpi=300)
    plt.close()

    # ======================================================
    # 3 - HISTOGRAMA DOS ERROS
    # ======================================================

    plt.figure(figsize=(7,5))
    plt.hist(y_test-pred,bins=40,edgecolor="black")
    plt.title("Random Forest - Prediction Error Distribution",fontsize=14)
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{RESULT_PATH}/rf_error_histogram.png",dpi=300)
    plt.close()

    # ======================================================
    # 4 - IMPORTÂNCIA DAS VARIÁVEIS
    # ======================================================

    importance=pd.DataFrame({
        "Feature":features,
        "Importance":model.feature_importances_
    })

    importance=importance.sort_values(
        by="Importance",
        ascending=False
    ).head(20)

    importance.to_csv(
        f"{RESULT_PATH}/rf_feature_importance.csv",
        sep=";",
        index=False
    )

    plt.figure(figsize=(9,7))
    plt.barh(
        importance["Feature"],
        importance["Importance"]
    )
    plt.gca().invert_yaxis()
    plt.title("Random Forest - Top 20 Feature Importances",fontsize=14)
    plt.xlabel("Importance")
    plt.grid(axis="x",alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{RESULT_PATH}/rf_feature_importance.png",dpi=300)
    plt.close()

    # ======================================================
    # 5 - IMPORTÂNCIA DAS VARIÁVEIS (NORMALIZADA)
    # ======================================================

    importance["Group"]=importance["Feature"].str.extract(
        r"(^.*)_t\d+"
    )
    grouped=importance.groupby("Group")["Importance"].sum()
    grouped=grouped.sort_values(ascending=False)
    plt.figure(figsize=(7,5))
    grouped.plot.bar()
    plt.title("Random Forest - Variable Importance by Signal",fontsize=14)
    plt.ylabel("Total Importance")
    plt.grid(axis="y",alpha=0.3)
    plt.tight_layout()  

    plt.savefig(f"{RESULT_PATH}/rf_feature_importance_grouped.png",dpi=300)
    plt.close()

    return model

# ==========================================================
# PART 3 - XGBOOST REGRESSOR
# ==========================================================

def xgboost_rul():

    X_train,y_train,X_test,y_test,features=prediction_rul_prepare()

    model=XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train,y_train)
    pred=model.predict(X_test)

    mae=mean_absolute_error(y_test,pred)
    rmse=np.sqrt(mean_squared_error(y_test,pred))
    r2=r2_score(y_test,pred)


    print("="*60)
    print("XGBOOST")
    print("="*60)
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")

    with open(f"{RESULT_PATH}/xgb_metrics.txt","w") as f:
        f.write(f"MAE={mae}\n")
        f.write(f"RMSE={rmse}\n")
        f.write(f"R2={r2}\n")

    result=pd.DataFrame({

        "Real":y_test,

        "Predicted":pred,

        "Error":y_test-pred

    })

    result.to_csv(

        f"{RESULT_PATH}/xgb_predictions.csv",

        sep=";",

        index=False

    )

    # ======================================================
    # 1 - REAL X PREVISTO
    # ======================================================

    plt.scatter(y_test,pred,s=10,alpha=0.6)
    lim=max(y_test.max(),pred.max())
    plt.plot([0,lim],[0,lim],"r--",linewidth=2,label="Ideal")
    plt.title("XGBoost - Actual vs Predicted RUL",fontsize=14)
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{RESULT_PATH}/xgb_real_vs_pred.png",dpi=300)
    plt.close()

    # ======================================================
    # 2 - CURVA RUL
    # ======================================================

    plt.figure(figsize=(14,5))
    plt.plot(y_test.values,label="Actual",linewidth=2)
    plt.plot(pred,label="Predicted",linewidth=2)
    plt.title("XGBoost - RUL Prediction",fontsize=14)
    plt.xlabel("Sample")
    plt.ylabel("Remaining Useful Life")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{RESULT_PATH}/xgb_rul_curve.png",dpi=300)
    plt.close()

    # ======================================================
    # 3 - HISTOGRAMA DOS ERROS
    # ======================================================

    plt.figure(figsize=(7,5))
    plt.hist(y_test-pred,bins=40,edgecolor="black")
    plt.title("XGBoost - Prediction Error Distribution",fontsize=14)
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{RESULT_PATH}/xgb_error_histogram.png",dpi=300)
    plt.close()

    # ======================================================
    # 4 - IMPORTÂNCIA DAS VARIÁVEIS
    # ======================================================

    importance=pd.DataFrame({
        "Feature":features,
        "Importance":model.feature_importances_
    })
    importance=importance.sort_values(
        by="Importance",
        ascending=False
    ).head(20)

    importance.to_csv(
        f"{RESULT_PATH}/xgb_feature_importance.csv",
        sep=";",
        index=False
    )

    plt.figure(figsize=(9,7))
    plt.barh(
        importance["Feature"],
        importance["Importance"]
    )
    plt.gca().invert_yaxis()
    plt.title("XGBoost - Top 20 Feature Importances",fontsize=14)
    plt.xlabel("Importance")
    plt.grid(axis="x",alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{RESULT_PATH}/xgb_feature_importance.png",dpi=300)
    plt.close()

    # ======================================================
    # 5 - IMPORTÂNCIA DAS VARIÁVEIS (NORMALIZADA)
    # ======================================================

    importance["Group"]=importance["Feature"].str.extract(
        r"(^.*)_t\d+"
    )
    grouped=importance.groupby("Group")["Importance"].sum()
    grouped=grouped.sort_values(ascending=False)
    plt.figure(figsize=(7,5))
    grouped.plot.bar()
    plt.title("XGBoost - Variable Importance by Signal",fontsize=14)
    plt.ylabel("Total Importance")
    plt.grid(axis="y",alpha=0.3)
    plt.tight_layout()  

    plt.savefig(f"{RESULT_PATH}/xgb_feature_importance_grouped.png",dpi=300)
    plt.close()

    return model


def neural_network_rul():

    X_train,y_train,X_test,y_test,features=prediction_rul_prepare()

    scaler=StandardScaler()

    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation="relu"),
        Dropout(0.20),
        Dense(64, activation="relu"),
        Dropout(0.20),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    early_stop=EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

    history=model.fit(

        X_train,

        y_train,

        validation_split=0.20,

        epochs=300,

        batch_size=64,

        callbacks=[early_stop],

        verbose=1

    )

    pred=model.predict(X_test).flatten()

    mae=mean_absolute_error(y_test,pred)
    rmse=np.sqrt(mean_squared_error(y_test,pred))
    r2=r2_score(y_test,pred)

    print("="*60)
    print("NEURAL NETWORK")
    print("="*60)
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")

    pd.DataFrame({
        "Real":y_test,
        "Predicted":pred,
        "Error":y_test-pred
    }).to_csv(
        f"{RESULT_PATH}/nn_predictions.csv",
        sep=";",
        index=False
    )

    plt.scatter(y_test,pred,s=10,alpha=0.6)
    lim=max(y_test.max(),pred.max())
    plt.plot([0,lim],[0,lim],"r--",linewidth=2,label="Ideal")
    plt.title("Neural Network - Actual vs Predicted RUL",fontsize=14)
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"{RESULT_PATH}/nn_real_vs_pred.png",dpi=300)
    plt.close()

    plt.figure(figsize=(14,5))
    plt.plot(y_test.values,label="Actual",linewidth=2)
    plt.plot(pred,label="Predicted",linewidth=2)
    plt.title("Neural Network - RUL Prediction",fontsize=14)
    plt.xlabel("Sample")
    plt.ylabel("Remaining Useful Life")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{RESULT_PATH}/nn_rul_curve.png",dpi=300)
    plt.close()

    plt.figure(figsize=(7,5))
    plt.hist(y_test-pred,bins=40,edgecolor="black")
    plt.title("Neural Network - Prediction Error Distribution",fontsize=14)
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{RESULT_PATH}/nn_error_histogram.png",dpi=300)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(history.history["loss"],label="Train")
    plt.plot(history.history["val_loss"],label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{RESULT_PATH}/nn_training.png",dpi=300)
    plt.close()

    return model

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print("="*70)
    print("PREDICTION OF REMAINING USEFUL LIFE (RUL)")
    print("="*70)

    print("\nRunning Random Forest...\n")
    rf_model = random_forest_rul()

    print("\nRunning XGBoost...\n")
    xgb_model = xgboost_rul()

    print("\nRunning Neural Network...\n")
    nn_model = neural_network_rul()

    print("\nFinished!")
    print(f"Results saved in: {RESULT_PATH}")