import torch
import torch.nn as nn
import optuna

from optuna.trial import Trial
from BasicCNN import BasicCNN
from train import train_on_mnist

N_TRIALS = 30

def objective(trial: Trial):
    """
    This function is called by Optuna for each trial.
    It builds, trains, and validates a model, returning its accuracy.
    """
    
    # MNIST starts at 28x28. Each MaxPool halves the size.
    # 1-layer -> 14x14
    # 2-layer -> 7x7
    # 3-layer -> 3x3
    # 4-layer -> 1x1
    num_conv_layers = trial.suggest_int('num_conv_layers', 1, 4)
    
    initial_channels = trial.suggest_categorical('initial_channels', [8, 16, 32])
    channel_multiplier = trial.suggest_float('channel_multiplier', 1.0, 2.5, step=0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BasicCNN(
        num_conv_layers=num_conv_layers,
        initial_channels=initial_channels,
        channel_multiplier=channel_multiplier,
        num_classes=10
    ).to(device)

    _model, final_accuracy = train_on_mnist(model=model, trial=trial)

    return final_accuracy
    
if __name__ == '__main__':
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner()
    )
    
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "="*30)
    print("Optuna Study Finished")
    print(f"Number of finished trials: {len(study.trials)}")
    
    print("\nBest trial:")
    trial = study.best_trial
    
    print(f" Accuracy: {trial.value:.4f}")
    print("  Best Model Size:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig1.write_image("optimization_history.png")
    fig1.show()
    
    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.write_image("param_importances.png")
    fig2.show()