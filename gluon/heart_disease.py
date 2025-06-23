from autogluon.tabular import TabularDataset, TabularPredictor
# with open('/Users/nunkesser/repos/work/articles/logicgp/data/ucimlrepo/HeartDisease/AutoGluon.csv', 'w') as f:
#     f.write(f"\"x\"\n")
# with open('/Users/nunkesser/repos/work/articles/logicgp/data/ucimlrepo/HeartDisease/AutoGluon_seeds.csv', 'w') as f:
#     f.write(f"\"seed\"\n")
# for seed in [
#         42, 7, 13, 99, 256, 1024, 73, 3, 17, 23,
#         5, 11, 19, 29, 31, 37, 41, 43, 47, 53,
#         59, 61, 67, 71, 79, 83, 89, 97, 101, 103,
#         107, 109, 113, 127, 131, 137, 139, 149, 151, 157,
#         163, 167, 173, 179, 181, 191, 193, 197, 199, 211,
#         223, 227, 229, 233, 239, 241, 251, 257, 263, 269,
#         271, 277, 281, 283, 293, 307, 311, 313, 317, 331,
#         337, 347, 349, 353, 359, 367, 373, 379, 383, 389,
#         397, 401, 409, 419, 421, 431, 433, 439, 443, 449,
#         457, 461, 463, 467, 479, 487, 491, 499, 503, 509
#     ]:
for seed in [
    359, 367, 373, 379, 383, 389,
    397, 401, 409, 419, 421, 431, 433, 439, 443, 449,
    457, 461, 463, 467, 479, 487, 491, 499, 503, 509
]:
    baccvals = []
    train_data = TabularDataset(f'../data/heart_disease_train_validate_seed{seed}.csv')
    #tuning_data = TabularDataset(f'/Users/nunkesser/repos/work/artifacts/nuget-ml-ucimlr/Italbytz.ML.UCIMLR/Italbytz.ML.UCIMLR/Data/Iris.csv')
    test_data = TabularDataset(f'../data/heart_disease_test_seed{seed}.csv')
    print(train_data.head())
    label = 'num'
    print(train_data[label].describe())
    time_limit = 60  # for quick demonstration only, you should set this to longest time you are willing to wait (in seconds)
    metric = 'f1_macro'  # specify your evaluation metric here
    preset = 'best_quality'
    #preset = 'medium'
    #predictor = TabularPredictor(label=label,problem_type='multiclass',eval_metric=metric).fit(train_data,tuning_data=tuning_data,time_limit=time_limit,presets=preset)
    predictor = TabularPredictor(label=label,problem_type='multiclass',eval_metric=metric).fit(train_data,time_limit=time_limit,presets=preset)
    y_pred = predictor.predict(test_data.drop(columns=[label]))
    print(y_pred.head())
    predictor.evaluate(test_data, display=True,detailed_report=True)
    bacc = predictor.evaluate(test_data, silent=True)['f1_macro']
    with open('/Users/nunkesser/repos/work/articles/logicgp/data/ucimlrepo/HeartDisease/AutoGluon.csv', 'a') as f:
        f.write(f"{bacc}\n")
    with open('/Users/nunkesser/repos/work/articles/logicgp/data/ucimlrepo/HeartDisease/AutoGluon_seeds.csv', 'a') as f:
        f.write(f"{seed}\n")
    baccvals.append(bacc)
#    for val in baccvals:
#        f.write(f"{val}\n")
#print(','.join(map(str, baccvals)))


