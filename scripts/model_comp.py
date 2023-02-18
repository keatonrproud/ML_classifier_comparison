"""Comparison of all ML algos across training and test data."""
import imblearn.pipeline
import sklearn as skl
import sklearn.model_selection as sklms
import sklearn.compose as sklcmp
import imblearn.over_sampling as imbovr
import numpy as np


def preprocessing_fns():
    categorical_encoder = skl.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    agegroup_levels = ['0-4', '5-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90+']
    sumcom_levels = [0, 1, 2, 3, 4, 5, 6]
    sumsym_levels = [0, 1, 2, 3, 4, 5, 6, 7]

    ordinal_encoder = skl.preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value',
                                                       unknown_value=-1,
                                                       categories=[agegroup_levels, sumcom_levels, sumsym_levels])

    categorical_fts = ['CriteriaConfirmation', 'Sex', 'ColorRace', 'Education', 'Fever',
                       'BreathingDifficulty', 'Cough', 'RunnyNose', 'SoreThroat', 'Diarrhea', 'Headache',
                       'ComorbidityPulmonary', 'ComorbidityCardiac', 'ComorbidityRenal', 'ComorbidityDiabetes',
                       'ComorbiditySmoking', 'ComorbidityObesity', 'Hospitalized', 'TravelBrasil',
                       'TravelInternational', 'OverSixty']
    ordinal_fts = ['AgeGroup', 'SumComorbidities', 'SumSymptoms']
    transformer = sklcmp.make_column_transformer((categorical_encoder, categorical_fts),
                                                 (ordinal_encoder, ordinal_fts))

    oversampler = imbovr.RandomOverSampler(sampling_strategy=1)

    return transformer, oversampler


def focused_preprocessing_fns():
    categorical_encoder = skl.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    agegroup_levels = ['0-4', '5-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90+']
    sumcom_levels = [0, 1, 2, 3, 4, 5, 6]
    sumsym_levels = [0, 1, 2, 3, 4, 5, 6, 7]

    ordinal_encoder = skl.preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value',
                                                       unknown_value=-1,
                                                       categories=[agegroup_levels, sumcom_levels, sumsym_levels])

    categorical_fts = ['Education', 'BreathingDifficulty', 'SoreThroat', 'Headache', 'ComorbidityDiabetes',
                       'Hospitalized', 'OverSixty']
    ordinal_fts = ['AgeGroup', 'SumComorbidities', 'SumSymptoms']
    transformer = sklcmp.make_column_transformer((categorical_encoder, categorical_fts),
                                                 (ordinal_encoder, ordinal_fts))

    oversampler = imbovr.RandomOverSampler(sampling_strategy=1)

    return transformer, oversampler


def train_test_split_strat(features, target, train_size=0.7):
    data = sklms.train_test_split(features,
                                  target,
                                  train_size=train_size,
                                  stratify=target)

    return data


def x_validate(model, features, target, evals: list, focused=False):
    """Run test w/ stratified 10-fold cross-validation"""

    if not focused:
        transformer, oversampler = preprocessing_fns()

    else:
        transformer, oversampler = focused_preprocessing_fns()

    pipeline = imblearn.pipeline.make_pipeline(transformer, oversampler, model)

    pipeline.fit(features, target)

    split_method = sklms.StratifiedKFold(n_splits=10)


    output_all = skl.model_selection.cross_validate(estimator=pipeline,
                                                    X=features,
                                                    y=target,
                                                    scoring=evals,
                                                    cv=split_method,
                                                    n_jobs=-1)  # uses multiple processors for training

    print(f'{model} results done')


    target_predictions = skl.model_selection.cross_val_predict(estimator=pipeline,
                                                               X=features,
                                                               y=target,
                                                               cv=split_method,
                                                               n_jobs=-1)

    print(f'{model} predicts done')

    target_preds_real = [target_predictions.ravel(), target.values.ravel()]

    print(f'{model} is completed')

    return output_all, target_preds_real


def train_models(models, features, target, evals: list, focused=False):
    """Runs all algos on the given dataset(s) and returns all outputs as a list"""

    outputs = []
    for model in models:
        results, target_preds_real = x_validate(model, features, target, evals, focused)
        preds, real = target_preds_real
        output = {
            'Model': model,
            'Accuracy': np.mean(results['test_accuracy']),
            'ROC AUC': np.mean(results['test_roc_auc']),
            'Average Precision': np.mean(results['test_average_precision']),
            'Precision': np.mean(results['test_precision']),
            'Recall': np.mean(results['test_recall']),
            'F1': np.mean(results['test_f1']),
            'Predictions': preds,
            'Real Outcomes': real
        }
        outputs.append(output)

    return outputs


def test_models(models, train_features, train_target, test_features, test_target, focused):

    if not focused:
        transformer, oversampler = preprocessing_fns()
    else:
        transformer, oversampler = focused_preprocessing_fns()

    test_results = []

    for model in models:
        pipeline = imblearn.pipeline.make_pipeline(transformer, oversampler, model)

        pipeline.fit(train_features, train_target)

        preds = pipeline.predict(test_features)
        real = test_target.values

        probs = pipeline.predict_proba(test_features)[:, 1]
        precision, recall, _ = skl.metrics.precision_recall_curve(real, probs)

        result = {
            'Model': model,
            'Accuracy': skl.metrics.accuracy_score(real, preds),
            'ROC AUC': skl.metrics.roc_auc_score(real, preds),
            'Average Precision': skl.metrics.average_precision_score(real, probs),
            'Precision': skl.metrics.precision_score(real, preds),
            'Recall': skl.metrics.recall_score(real, preds),
            'F1': skl.metrics.f1_score(real, preds),
            'Predictions': preds,
            'Real Outcomes': real
        }
        test_results.append(result)

    return test_results


def lists_of_runs_to_lists_of_model_means(all_dfs: list):
    all_mean_results = []
    for data in all_dfs:
        data_by_model = []

        run_num = 0

        for run in data:
            run_num += 1
            for new_model_run in run:
                if new_model_run.get('Model') in [model_runs.get('Model') for model_runs in data_by_model]:
                    for existing in data_by_model:
                        if new_model_run.get('Model') == existing.get('Model'):
                            for key in existing.keys():
                                if key not in ('Model', 'Predictions', 'Real Outcomes'):
                                    existing[key] += new_model_run[key]
                            for key in ('Predictions', 'Real Outcomes'):
                                existing[key] = np.append(existing[key], new_model_run[key])

                else:
                    data_by_model.append(new_model_run)

        # divide by number of runs
        for model in data_by_model:
            for key in model.keys():
                if key not in ('Model', 'Predictions', 'Real Outcomes'):
                    model[key] = model[key] / run_num

        all_mean_results.append(data_by_model)

    return all_mean_results
