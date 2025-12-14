# bayesian_optimization_gpyopt.py
import os
from tensorflow.keras import layers, regularizers, callbacks, models, optimizers
from tensorflow.keras.datasets import mnist
import GPyOpt
import matplotlib.pyplot as plt

MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 5
RESULTS_DIR = "checkpoints"
REPORT_FILE = "bayes_opt.txt"
os.makedirs(RESULTS_DIR, exist_ok=True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

x_train, x_val = x_train[:-10000], x_train[-10000:]
y_train, y_val = y_train[:-10000], y_train[-10000:]

def objective_function(params):
    lr, units, dropout, l2_reg, batch_size = params[0]

    units = int(units)
    batch_size = int(batch_size)

    model = models.Sequential([
        layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
            input_shape=(784,)
        ),
        layers.Dropout(dropout),
        layers.Dense(10, activation="softmax")
    ])

    optimizer = optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint_name = (
        f"lr={lr:.5f}_units={units}_dropout={dropout:.2f}"
        f"_l2={l2_reg:.5f}_bs={batch_size}.keras"
    )

    checkpoint_path = os.path.join(RESULTS_DIR, checkpoint_name)

    cb_early = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True
    )

    cb_checkpoint = callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=batch_size,
        verbose=0,
        callbacks=[cb_early, cb_checkpoint]
    )

    best_val_acc = max(history.history["val_accuracy"])

    with open(REPORT_FILE, "a") as f:
        f.write(
            f"LR={lr}, Units={units}, Dropout={dropout}, "
            f"L2={l2_reg}, Batch={batch_size}, "
            f"Val_Acc={best_val_acc}\n"
        )
    return -best_val_acc

domain = [
    {"name": "learning_rate", "type": "continuous", "domain": (1e-4, 1e-2)},
    {"name": "units", "type": "discrete", "domain": (64, 128, 256, 512)},
    {"name": "dropout", "type": "continuous", "domain": (0.1, 0.5)},
    {"name": "l2", "type": "continuous", "domain": (1e-5, 1e-2)},
    {"name": "batch_size", "type": "discrete", "domain": (32, 64, 128)}
]

optimizer = GPyOpt.methods.BayesianOptimization(
    f=objective_function,
    domain=domain,
    acquisition_type="EI",
    exact_feval=True
)

optimizer.run_optimization(max_iter=30)

optimizer.plot_convergence()
plt.title("Bayesian Optimization Convergence")
plt.show()

with open(REPORT_FILE, "a") as f:
    f.write("\nBest parameters found:\n")
    f.write(str(optimizer.x_opt))
    f.write("\nBest validation accuracy:\n")
    f.write(str(-optimizer.fx_opt))
