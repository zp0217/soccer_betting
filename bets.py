import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import GlorotUniform, RandomNormal
from tensorflow.keras.utils import to_categorical
import seaborn as sns
import keras_tuner as kt
import joblib
from keras_tuner import HyperParameters


df = pd.read_csv("data/dat.csv")
df.drop(columns=[col for col in ['B365>2.5', 'B365<2.5', 'Unnamed: 0'] if col in df.columns], inplace=True)


odds_columns = ['B365H', 'B365D', 'B365A']
odds_df = df[odds_columns].copy()


excluded_columns = ['FTR', 'FTR_encoded', 'FTR_A', 'FTR_H', 'FTR_D']
columns_to_standardize = [col for col in df.select_dtypes(include=['float64', 'int64']).columns
                          if col not in odds_columns and col not in excluded_columns and 'norm_diff' not in col.lower()]
scaler = StandardScaler()
df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

le = LabelEncoder()
y = le.fit_transform(df['FTR'])
y_cat = to_categorical(y, num_classes=3)

features = [
   'HST', 'AST', 'HC', 'AC', 'Home_xG', 'Away_xG',
   'ShotDiff', 'CornerDiff', 'FoulDiff', 'xGDiff',
   'Norm_ShotDiff', 'Norm_CornerDiff', 'Norm_FoulDiff', 'Norm_xGDiff'
]
x = df[features].values

x_indices = np.arange(len(x))
x_train, x_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
    x, y_cat, x_indices, test_size=0.2, random_state=10
)
x_val, x_test, y_val, y_test, idx_val, idx_test = train_test_split(
    x_temp, y_temp, idx_temp, test_size=0.5, random_state=10
)


class M1:
    def __init__(self, input_shape, num_classes, hyperparam, visualize=True):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hyperparams = hyperparam
        self.visualize = visualize
        self.histories = []
        self.y_preds = []
        self.y_trues = []

    def m2(self):
        model = Sequential()

        if self.hyperparams['weight_init'] == 'xavier':
            initializer = GlorotUniform()
        else:
            initializer = RandomNormal()
        
        for units in self.hyperparams['hidden_layer']:
         if self.hyperparams['activation'] == 'leaky_relu':
            model.add(Dense(units,
                            kernel_regularizer=l1_l2(self.hyperparams['l1'], self.hyperparams['l2']),
                            kernel_initializer=initializer))
            model.add(LeakyReLU(alpha=0.01))  
         else:
            model.add(Dense(units,
                            activation=self.hyperparams['activation'],
                            kernel_regularizer=l1_l2(self.hyperparams['l1'], self.hyperparams['l2']),
                            kernel_initializer=initializer))

        model.add(Dropout(self.hyperparams['dropout']))

        model.add(Dense(self.num_classes, activation='softmax'))

        optimizer_choices = {
            'adam': Adam(learning_rate=self.hyperparams['learning_rate']),
            'sgd': SGD(learning_rate=self.hyperparams['learning_rate']),
            'rmsprop': RMSprop(learning_rate=self.hyperparams['learning_rate'])
        }
        optimizer = optimizer_choices[self.hyperparams['optimizer']]

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        return model

    def train(self, x, y):
        
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y if y.ndim == 1 else None
        )

        
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val if y_train_val.ndim == 1 else None
        )

        model = self.m2()

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=self.hyperparams['epochs'],
            batch_size=self.hyperparams['batch_size'],
            verbose=1,
            callbacks=[early_stop]
        )

       
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        val_loss = min(history.history['val_loss'])
        train_loss = min(history.history['loss'])

        print(f"\nTest Accuracy: {test_acc:.4f}")

        y_pred = np.argmax(model.predict(x_test), axis=1)
        y_actual = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test

        self.histories = [history]
        self.y_preds = y_pred
        self.y_trues = y_actual

        if self.visualize:
            self.plot_training_curves(self.histories)
            self.plot_confusion_matrix(self.y_trues, self.y_preds)

        return train_loss, val_loss, test_acc

    def plot_training_curves(self, histories):
        for i, history in enumerate(histories):
            plt.plot(history.history['loss'], label=f'Train Loss')
            plt.plot(history.history['val_loss'], label=f'Validation Loss', linestyle='dashed')
            plt.title('Training & Validation Loss(nueral network)')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=range(self.num_classes),
                    yticklabels=range(self.num_classes))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix(nueral network)')
        plt.show()


def build_model(hp):
    hyperparam = {
        'hidden_layer': [hp.Int(f'units_{i}', min_value=32, max_value=128, step=16) for i in range(hp.Int('num_layers', 1, 3))],
        'activation': hp.Choice('activation', ['relu','tanh','leaky_relu']),
        'optimizer': hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd']),
        'learning_rate': hp.Float('learning_rate', 1e-4, 1e-2, sampling='log'),
        'dropout': hp.Float('dropout', 0.1, 0.5, step=0.01),
        'l1': hp.Float('l1', 0.0, 0.01, step=0.001),
        'l2': hp.Float('l2', 0.0, 0.01, step=0.001),
        'weight_init': hp.Choice('weight_init', ['xavier', 'random']),
        'epochs': 50,
        'batch_size': hp.Choice('batch_size', [32, 64, 128])
    }

    model_instance = M1(input_shape=x.shape[1], num_classes=3, hyperparam=hyperparam, visualize=False)
    model = model_instance.m2()
    return model




tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=30,
    executions_per_trial=1,
)

tuner.search_space_summary()
tuner.search(x_train, y_train, validation_data=(x_val, y_val), epochs=50, callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

for param in best_hp.values:
    print(f"{param}: {best_hp.get(param)}")
test_loss, test_acc = best_model.evaluate(x_test, y_test)
print(f"\nBest Model Test Accuracy: {test_acc:.4f}")

best_params = {
    'hidden_layer': [best_hp.get(f'units_{i}') for i in range(best_hp.get('num_layers'))],
    'activation': best_hp.get('activation'),
    'optimizer': best_hp.get('optimizer'),
    'learning_rate': best_hp.get('learning_rate'),
    'dropout': best_hp.get('dropout'),
    'l1': best_hp.get('l1'),
    'l2': best_hp.get('l2'),
    'weight_init': best_hp.get('weight_init'),
    'epochs': 50,
    'batch_size': best_hp.get('batch_size')
}

best_model = M1(input_shape=x.shape[1], num_classes=3, hyperparam=best_params, visualize=True)

final_train_loss, final_val_loss, final_test_acc = best_model.train(x, y_cat)
print(f"\nFinal Results: Avg Train Loss = {final_train_loss:.4f}, "
              f"Avg Val Loss = {final_val_loss:.4f}, Avg Test Accuracy = {final_test_acc:.4f}")

# create hyperparameter table
top_row_data = {
    "Layers": ["3"],
    "Activation": ["Leaky ReLU"],
    "Optimizer": ["RMSprop"],
    "Learning Rate": ["0.00115"],
}

middle_row_data = {
    "Units 0": ["80"],
    "units 1": ['80'],
    "units 2": ["128"],
    "Batch Size": ["32"]
}

bottom_row_data = {
    "Dropout": ["0.49"],
    "L1 Regularizer": ["0.005"],
    "L2 Regularizer": ["0.009"],
    "Weight Initialization": ["Xavier"],
}

df_top = pd.DataFrame(top_row_data)
df_middle = pd.DataFrame(middle_row_data)
df_bottom = pd.DataFrame(bottom_row_data)

fig, axs = plt.subplots(3, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 1, 1]})
fig.suptitle("Neural Network Model Hyperparameters", fontsize=13, weight='bold', y=1.02)

# Top row
axs[0].axis('off')
table_top = axs[0].table(
    cellText=df_top.values,
    colLabels=df_top.columns,
    cellLoc='center',
    loc='center',
    colColours=["#fce5cd"] * len(df_top.columns)
)
table_top.auto_set_font_size(False)
table_top.set_fontsize(10.5)
table_top.scale(1.1, 1.5)

# Middle row
axs[1].axis('off')
table_middle = axs[1].table(
    cellText=df_middle.values,
    colLabels=df_middle.columns,
    cellLoc='center',
    loc='center',
    colColours=["#fce5cd"] * len(df_middle.columns)
)
table_middle.auto_set_font_size(False)
table_middle.set_fontsize(10.5)
table_middle.scale(1.1, 1.5)

# Bottom row
axs[2].axis('off')
table_bottom = axs[2].table(
    cellText=df_bottom.values,
    colLabels=df_bottom.columns,
    cellLoc='center',
    loc='center',
    colColours=["#fce5cd"] * len(df_bottom.columns)
)
table_bottom.auto_set_font_size(False)
table_bottom.set_fontsize(10.5)
table_bottom.scale(1.1, 1.5)

plt.tight_layout()
plt.subplots_adjust(top=0.9, hspace=0.6)
plt.show()



#betting simulation
final_model = best_model.m2()
final_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, batch_size=32, verbose=0)

probs = final_model.predict(x_test)
y_true = np.argmax(y_test, axis=1)

X_test_betting = pd.DataFrame(x_test, columns=features)
X_test_betting["B365H"] = df.iloc[idx_test]["B365H"].values
X_test_betting["B365D"] = df.iloc[idx_test]["B365D"].values
X_test_betting["B365A"] = df.iloc[idx_test]["B365A"].values

X_test_betting["B365H_prob"] = 1 / X_test_betting["B365H"]
X_test_betting["B365D_prob"] = 1 / X_test_betting["B365D"]
X_test_betting["B365A_prob"] = 1 / X_test_betting["B365A"]
total = X_test_betting[["B365H_prob", "B365D_prob", "B365A_prob"]].sum(axis=1)
X_test_betting["B365H_prob"] /= total
X_test_betting["B365D_prob"] /= total
X_test_betting["B365A_prob"] /= total



final_model.save("final_model.keras") 
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)
np.save('idx_test.npy', idx_test)
df.to_csv('df.csv', index=False)

