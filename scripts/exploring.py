"""Exploring the data and models"""

import model_comp as mc
import pandas as pd


def view_outcome_distr(list_of_dfs: list):
    """View the distribution of variables for each column in the dataframe

    :param list_of_dfs: a list of DataFrames to explore
    """
    for i in list_of_dfs:
        print(i)
        print(i['Outcome'].value_counts(normalize=True))


def get_baseline_data(train_features, train_target, evals):
    """View the distribution of variables for each column in the dataframe

    :param train_features: features to train on
    :param train_target: features to test on
    :param evals: metrics to create

    :return: the baseline results from training the models on the given data
    """
    # run models w/ no hyperparameters on the training data and get results
    return mc.train_models([mods.lm.LogisticRegression(max_iter=1000), mods.da.LinearDiscriminantAnalysis(),
                                        mods.nbr.KNeighborsClassifier()], features=train_features, target=train_target,
                                       evals=evals)


def check_feature_importance(model, features, target):
    """Evaluate importance of each feature given the features and model

    :param model: model to use
    :param features: features to use
    :param target: target values to use

    Variables:
        transformer: the encoders
        oversampler: used for oversampling the imbalanced data
        pipeline: encoder, oversampler, and model combined

    :return: a DataFrame of the feature names and their importances
    """
    transformer, oversampler = mc.preprocessing_fns()

    pipeline = mc.imblearn.pipeline.make_pipeline(transformer, oversampler, model)

    pipeline.fit(features, target)

    return pd.DataFrame(pipeline.steps[2][1].feature_importances_, pipeline.steps[0][1].get_feature_names_out())


def grid_search(models_and_param_grid, features, target, eval, verbose: int = 0):
    """Complete a grid search for the given model(s) and parameters

    :param models_and_param_grid: 1 or multiple dictionaries for comparing models and model parameters
    :param features: df of features
    :param target: df of target values
    :param eval: metric to optimize
    :param verbose: detail/quantity of text to be printed for each round of the grid search

    Variables:
        transformer: the encoders
        oversampler: used for oversampling the imbalanced data
        pipeline: encoder, oversampler, and model combined
        split_method: 10-fold stratified split of the data for each grid search
        gs: the grid search object

    :return: Best Parameters from the grid search, Results from the grid search
    """
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
