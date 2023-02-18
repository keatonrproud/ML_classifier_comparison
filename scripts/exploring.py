import model_comp as mc
import pandas as pd


def view_outcome_distr(list_of_dfs: list):
    for i in list_of_dfs:
        print(i)
        print(i['Outcome'].value_counts(normalize=True))


def get_baseline_data(train_features, train_target, evals):
    # run models w/ no hyperparameters on the training data and get results
    baseline_results = mc.train_models([mods.lm.LogisticRegression(max_iter=1000), mods.da.LinearDiscriminantAnalysis(),
                                        mods.nbr.KNeighborsClassifier()], features=train_features, target=train_target,
                                       evals=evals)

    return baseline_results


def check_feature_importance(model, features, target):
    transformer, oversampler = mc.preprocessing_fns()

    pipeline = mc.imblearn.pipeline.make_pipeline(transformer, oversampler, model)

    pipeline.fit(features, target)

    print('============= Feature Importances =============')
    print(pipeline.steps[2][1].feature_importances_)
    print(pipeline.steps[0][1].get_feature_names_out())


def grid_search(models_and_param_grid, features, target, eval, verbose: int = 0):
    for model in models_and_param_grid:
        transformer, oversampler = mc.preprocessing_fns()

        pipeline = mc.imblearn.pipeline.Pipeline(steps=[('transformer', transformer),
                                                        ('oversampler', oversampler),
                                                        ('model', model[0])])

        split_method = mc.sklms.StratifiedKFold(n_splits=10)

        gs = mc.sklms.GridSearchCV(estimator=pipeline,
                                   param_grid=model[1],
                                   scoring=eval,
                                   n_jobs=-1,
                                   cv=split_method,
                                   refit=True,
                                   verbose=verbose)  # allows parameters to be adjusted to see new predictions

        gs.fit(features, target)

        return gs.best_params_, pd.DataFrame(gs.cv_results_)
