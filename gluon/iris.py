from autogluon.tabular import TabularDataset, TabularPredictor
train_data = TabularDataset(f'/Users/nunkesser/repos/work/artifacts/nuget-ml-ucimlr/Italbytz.ML.UCIMLR/Italbytz.ML.UCIMLR/Data/Iris.csv')
#tuning_data = TabularDataset(f'/Users/nunkesser/repos/work/artifacts/nuget-ml-ucimlr/Italbytz.ML.UCIMLR/Italbytz.ML.UCIMLR/Data/Iris.csv')
test_data = TabularDataset(f'/Users/nunkesser/repos/work/artifacts/nuget-ml-ucimlr/Italbytz.ML.UCIMLR/Italbytz.ML.UCIMLR/Data/Iris.csv')
print(train_data.head())
label = 'class'
print(train_data[label].describe())
time_limit = 20  # for quick demonstration only, you should set this to longest time you are willing to wait (in seconds)
metric = 'balanced_accuracy'  # specify your evaluation metric here
preset = 'best_quality'
#preset = 'medium'
#predictor = TabularPredictor(label=label,problem_type='multiclass',eval_metric=metric).fit(train_data,tuning_data=tuning_data,time_limit=time_limit,presets=preset)
predictor = TabularPredictor(label=label,problem_type='multiclass',eval_metric=metric).fit(train_data,time_limit=time_limit,presets=preset)
y_pred = predictor.predict(test_data.drop(columns=[label]))
print(y_pred.head())
print(predictor.evaluate(test_data, silent=True))
print(predictor.leaderboard(test_data))