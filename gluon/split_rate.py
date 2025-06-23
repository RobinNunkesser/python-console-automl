from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import numpy as np

# Zufallsdaten erzeugen
np.random.seed(42)
for n_samples in [
    150,303,4898,569,48842
]:
    X = np.random.randn(n_samples, 5)
    y = np.random.randint(0, 2, size=n_samples)

    df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(5)])
    df['output'] = y

    # Als CSV speichern
    df.to_csv(f'../data/random_classification_data_{n_samples}.csv', index=False)
    train_data = TabularDataset(f'../data/random_classification_data_{n_samples}.csv')
    label = 'output'
    time_limit = 20  # for quick demonstration only, you should set this to longest time you are willing to wait (in seconds)
    metric = 'f1'  # specify your evaluation metric here
    preset = 'medium'
    predictor = TabularPredictor(label=label, problem_type='binary', eval_metric=metric).fit(train_data,
                                                                                                 time_limit=time_limit,
                                                                                                 presets=preset)
# 100: Automatically generating train/validation split with holdout_frac=0.2, Train Rows: 80, Val Rows: 20
# 1000: Automatically generating train/validation split with holdout_frac=0.2, Train Rows: 800, Val Rows: 200
# 10000: Automatically generating train/validation split with holdout_frac=0.1, Train Rows: 8000, Val Rows: 2000
# 10000: Automatically generating train/validation split with holdout_frac=0.025, Train Rows: 8000, Val Rows: 2000