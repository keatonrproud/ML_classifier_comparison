# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
# ---

import cleaning as cln
import exploring as xpl
import visualizations as vis
import model_comp as mc
import sklearnex
from sklearnex import patch_sklearn

patch_sklearn()
import sklearn.linear_model as lm
import sklearn.discriminant_analysis as da
import sklearn.neighbors as nbr
import sklearn.tree as tree
import sklearn.svm as svm
import xgboost as xgb
import sklearn.feature_selection as sklfs

def run_exp(models, train_features, train_target, test_features, test_target, evals, rpt: int = 33, focused=False):
    train_all = []
    test_all = []
    for num in range(rpt):
        train_results = mc.train_models(models, train_features, train_target, evals, focused)

        test_results = mc.test_models(models, train_features, train_target, test_features, test_target, focused)

        train_all.append(train_results)
        test_all.append(test_results)

        print(f'Round {num + 1} completed')

    return [train_all, test_all]


# ------ Clean the Data and build Test / Training sets for each Exp
# # creates a df for each experiment, in one list called dfs, from the main() method in cleaning module
pr10_exp1_df, pr10_exp2_df, all_data = cln.main()
dfs = [pr10_exp1_df, pr10_exp2_df, all_data]
#
# # ------- EXP1 ------
# # # creates the exp1 train/test datasets
# exp1_features = pr10_exp1_df.iloc[:, 1:]
# exp1_targets = pr10_exp1_df.iloc[:, 0]
# exp1_f_train, exp1_f_test, exp1_t_train, exp1_t_test = mc.train_test_split_strat(
#     features=exp1_features,  # features are all columns after the target column
#     target=exp1_targets)  # target is first column -- originally 'Outcome'
#
# # set the tuned models to test for the experiments
# tuned_models = [
#     lm.LogisticRegression(max_iter=1500, C=3.0, fit_intercept=True, tol=1e-05),
#     da.LinearDiscriminantAnalysis(store_covariance=True),
#     nbr.KNeighborsClassifier(algorithm='kd_tree', leaf_size=10, n_neighbors=10, p=1),
#     tree.DecisionTreeClassifier(max_features=None, max_leaf_nodes=10, class_weight='balanced',
#                                 criterion='log_loss', min_samples_leaf=3),
#     xgb.XGBClassifier(colsample_bytree=0.5, learning_rate=0.01, max_depth=3, n_estimators=1000, subsample=0.6),
#     svm.SVC(kernel='rbf', C=10, gamma=0.0001, probability=True)
# ]
#
# # # ------ Grid Search to find the best hyperparameters
# # # Grid Search to find the best Hyperparameters
# params, results = xpl.grid_search([[lm.LogisticRegression(),
#                                     [{"model__max_iter": (1000, 1500),
#                                       "model__C": (2, 3),
#                                       "model__tol": (1e-04, 1e-05)}]]],
#                                   features=exp1_f_train,
#                                   target=exp1_t_train,
#                                   eval='roc_auc',
#                                   verbose=3)
#
# print(f'params\n'
#       f'results')
#
# # Run Experiment 1 and create Training and Test Results
# all_exp1_dfs = run_exp(tuned_models,
#                        train_features=exp1_f_train,
#                        train_target=exp1_t_train,
#                        test_features=exp1_f_test,
#                        test_target=exp1_t_test,
#                        evals=['accuracy', 'roc_auc', 'average_precision', 'precision', 'recall', 'f1'],
#                        rpt=3)
#
# # Get means of test results for each run
# exp1_train_means, exp1_test_means = mc.lists_of_runs_to_lists_of_model_means(all_exp1_dfs)
#
# # -------- EXP2 -------
# # creates the exp2 train/test datasets
# exp2_f_train = exp1_features
# exp2_t_train = exp1_targets
# exp2_f_test = pr10_exp2_df.iloc[:, 1:]
# exp2_t_test = pr10_exp2_df.iloc[:, 0]
#
# # Run Experiment 2 and create Training and Test Results
# all_exp2_dfs = run_exp(tuned_models,
#                        exp2_f_train,
#                        exp2_t_train,
#                        exp2_f_test,
#                        exp2_t_test,
#                        evals=['accuracy', 'roc_auc', 'average_precision', 'precision', 'recall', 'f1'],
#                        rpt=1)
#
# # Get means of test results for each run
# exp2_train_means, exp2_test_means = mc.lists_of_runs_to_lists_of_model_means(all_exp2_dfs)
#
# # ------ Create Demographic and Model Performance Visualizations
# # create and save demographics data for each experiment
# data_names = ['exp1', 'exp2', 'all_data']
#
# for data_w_name in zip(dfs, data_names):
#     vis.build_all_demographics(data_w_name[0], data_w_name[1])
#
# # create correct data format for confusion matrices
# all_exp2_test_models, all_exp2_test_real_and_preds = vis.create_confusion_matrix_info(exp2_test_means)
#
# # create confusion matrices
# vis.build_conf_mtrx_grid(all_exp2_test_models, all_exp2_test_real_and_preds,
#                          filename='conf_mtrx_exp2_test',
#                          title='Confusion Matrices from Exp2 Test Data')
#
# # Barplots of Model Results in each Phase (Baseline, Training, Test) for each Exp, for Acc / ROC AUC / PR AUC
# barplot_metrics = ['Accuracy', 'ROC AUC', 'Average Precision', 'Precision', 'Recall', 'F1']  # choose metrics for the barplots
#
# # set data, filename, and title for the barplot images
# barplot_info = [
#     [exp1_train_means, 'model_metrics_exp1_train', 'Exp1 Training Model Metrics'],
#     [exp1_test_means, 'model_metrics_exp1_test', 'Exp1 Test Model Metrics'],
#     [exp2_train_means, 'model_metrics_exp2_train', 'Exp2 Training Model Metrics'],
#     [exp2_test_means, 'model_metrics_exp2_test', 'Exp2 Test Model Metrics']]
#
# # create  a barplot for each list in barplot_info
# for info in barplot_info:
#     vis.model_metric_bar_grid(data=info[0], metrics=barplot_metrics, filename=info[1], title=info[2])
#
# # create comparison table of key metrics for each model in each exp for training and test
# print(f'{vis.build_comparison_tables(exp2_train_means)}\n'
#       f'{vis.build_comparison_tables(exp2_test_means)}')
#
# # ------- Additional RQ Exploration
# # ### Histogram evaluating all data of cures vs deaths for each symptom
# # done above
#
#
# #### evaluate the most important variables for the model
# best_mod = xgb.XGBClassifier(colsample_bytree=0.5, learning_rate=0.01, max_depth=3, n_estimators=1000,
#                                   subsample=0.6)
#
# xpl.check_feature_importance(xgb.XGBClassifier(), exp1_f_train, exp1_t_train)
#
# # keep education, breathingdifficulty, sorethroat, headache, hospitalization, age, over60, sumsymptoms, sumcomorbidites
# # evaluate model w/ fewer features on Exp2 data and compare to full model performance
# # for XGB, led to nearly identical Accuracy, ROC AUC, and Recall, and barely lower Precision
#
# # remove the columns -- keep only above features
# focused_exp1 = pr10_exp1_df.copy()
# focused_exp2 = pr10_exp2_df.copy()
# for df in focused_exp1, focused_exp2:
#     df.drop(columns=['CriteriaConfirmation', 'Sex', 'ColorRace', 'Fever', 'Cough', 'RunnyNose', 'Diarrhea',
#                      'ComorbidityPulmonary', 'ComorbidityCardiac', 'ComorbidityRenal',
#                      'ComorbiditySmoking', 'ComorbidityObesity', 'TravelBrasil', 'TravelInternational'],
#             inplace=True)
#
# focused_f_train = focused_exp1.iloc[:, 1:]
# focused_t_train = focused_exp1.iloc[:, 0]
# focused_f_test = focused_exp2.iloc[:, 1:]
# focused_t_test = focused_exp2.iloc[:, 0]
#
# focused_dfs = run_exp([best_mod],
#                       focused_f_train,
#                       focused_t_train,
#                       focused_f_test,
#                       focused_t_test,
#                       evals=['accuracy', 'roc_auc', 'average_precision', 'precision', 'recall', 'f1'],
#                       rpt=1,
#                       focused=True)
#
# # Get means of test results for each run
# focused_train_means, focused_test_means = mc.lists_of_runs_to_lists_of_model_means(focused_dfs)
#
# # create correct data format for confusion matrices
# focused_models, focused_real_and_preds = vis.create_confusion_matrix_info(focused_test_means)
#
# # create confusion matrices
# vis.build_conf_mtrx_grid(focused_models, focused_real_and_preds,
#                          filename='conf_mtrx_focused_test',
#                          title='Confusion Matrices from Focused Test Data')
#
# # Barplots of Model Results in each Phase (Baseline, Training, Test) for each Exp, for Acc / ROC AUC / PR AUC
# barplot_metrics = ['Accuracy', 'ROC AUC', 'Average Precision', 'Precision', 'Recall',
#                    'F1']  # choose metrics for the barplots
#
# # set data, filename, and title for the barplot images
# barplot_info = [[focused_test_means, 'model_metrics_focused_test', 'Focused Test Model Metrics']]
#
# # create  a barplot for each list in barplot_info
# for info in barplot_info:
#     vis.model_metric_bar_grid(data=info[0], metrics=barplot_metrics, filename=info[1], title=info[2])

##### use the one most effective model (XGB) trained on Exp1+2 data on all 2021 cases...
# show confusion matrix + model metrics for comparison of effectiveness of the XGB model
train_pre_2020_june = all_data[
    (all_data['NotificationDate'] < "2020-06-01")].copy()
test_recent = all_data[
    (all_data['NotificationDate'] > "2020-12-31")].copy()

for df in train_pre_2020_june, test_recent:
    del df['NotificationDate']

train_pre_2020_june_f = train_pre_2020_june.iloc[:, 1:]
train_pre_2020_june_t = train_pre_2020_june.iloc[:, 0]
test_recent_f = test_recent.iloc[:, 1:]
test_recent_t = test_recent.iloc[:, 0]

# recent_dfs = run_exp([best_mod],
#                      train_pre_2020_june_f,
#                      train_pre_2020_june_t,
#                      test_recent_f,
#                      test_recent_t,
#                      evals=['accuracy', 'roc_auc', 'average_precision', 'precision', 'recall', 'f1'],
#                      rpt=1)
#
# # Get means of test results for each run
# train_pre_2020_june_means, test_recent_means = mc.lists_of_runs_to_lists_of_model_means(recent_dfs)
#
# # create correct data format for confusion matrices
# recent_data_models, recent_data_real_and_preds = vis.create_confusion_matrix_info(test_recent_means)
#
# # create confusion matrices
# vis.build_conf_mtrx_grid(recent_data_models, recent_data_real_and_preds,
#                          filename='conf_mtrx_recent_test',
#                          title='Confusion Matrices from Recent Test Data')
#
# # Barplots of Model Results in each Phase (Baseline, Training, Test) for each Exp, for Acc / ROC AUC / PR AUC
# barplot_metrics = ['Accuracy', 'ROC AUC', 'Average Precision', 'Precision', 'Recall',
#                    'F1']  # choose metrics for the barplots
#
# # set data, filename, and title for the barplot images
# barplot_info = [[test_recent_means, 'model_metrics_recent_test', 'Recent Test Model Metrics']]
#
# # create  a barplot for each list in barplot_info
# for info in barplot_info:
#     vis.model_metric_bar_grid(data=info[0], metrics=barplot_metrics, filename=info[1], title=info[2])
#
#
# #### implement ensemble learning -- Voting Classifier -- with the 7 models from the paper
# import sklearn.ensemble as sklen
#
# voting = sklen.VotingClassifier(estimators=
#                                 [('LR', lm.LogisticRegression(max_iter=1500, C=3.0,
#                                                               fit_intercept=True, tol=1e-05)),
#                                  ('LDA', da.LinearDiscriminantAnalysis(store_covariance=True)),
#                                  ('KNN', nbr.KNeighborsClassifier(algorithm='kd_tree', leaf_size=10,
#                                                                   n_neighbors=10, p=1)),
#                                  ('XGB', xgb.XGBClassifier(colsample_bytree=0.5, learning_rate=0.01, max_depth=3,
#                                                            n_estimators=1000, subsample=0.6)),
#                                  ('DT', tree.DecisionTreeClassifier(max_features=None, max_leaf_nodes=10,
#                                                                     class_weight='balanced',
#                                                                     criterion='log_loss', min_samples_leaf=3)),
#                                  ('SVC', svm.SVC(kernel='rbf', C=10, gamma=0.0001, probability=True))],
#                                 voting='soft')
#
# voting_dfs = run_exp([voting],
#                      exp2_f_train,
#                      exp2_t_train,
#                      exp2_f_test,
#                      exp2_t_test,
#                      evals=['accuracy', 'roc_auc', 'average_precision', 'precision', 'recall', 'f1'],
#                      rpt=1)
#
# # Get means of test results for each run
# voting_train_means, voting_test_means = mc.lists_of_runs_to_lists_of_model_means(voting_dfs)
#
# # create correct data format for confusion matrices
# voting_models, voting_real_and_preds = vis.create_confusion_matrix_info(voting_test_means)
#
# # create confusion matrices
# vis.build_conf_mtrx_grid(voting_models, voting_real_and_preds,
#                          filename='conf_mtrx_voting_test',
#                          title='Confusion Matrices from Voting Classifier on Exp2 Test Data')

##### speed improvements using sklearnex to make processing large datasets more quickly
# compare times to complete for the above task with and without sklearnex
# XGBoost isn't affected, so test with LogisticRegression and KNN
import time

patched_start = time.time()
run_exp([lm.LogisticRegression(max_iter=1500, C=3.0, fit_intercept=True, tol=1e-05)],
        train_pre_2020_june_f,
        train_pre_2020_june_t,
        test_recent_f,
        test_recent_t,
        evals=['accuracy', 'roc_auc', 'average_precision', 'precision', 'recall', 'f1'],
        rpt=2)
patched_end = time.time() - patched_start

sklearnex.unpatch_sklearn()
from sklearn.linear_model import LogisticRegression

unpatched_start = time.time()
run_exp([LogisticRegression(max_iter=1500, C=3.0, fit_intercept=True, tol=1e-05)],
        train_pre_2020_june_f,
        train_pre_2020_june_t,
        test_recent_f,
        test_recent_t,
        evals=['accuracy', 'roc_auc', 'average_precision', 'precision', 'recall', 'f1'],
        rpt=2)
unpatched_end = time.time() - unpatched_start

print(f'Train Data # of rows = {len(train_pre_2020_june_f.index)}\n'
      f'Test Data # of rows = {len(test_recent_f.index)}\n'
      f'Seconds Taken for Normal Sklearn Model: {unpatched_end}\n'
      f'Seconds Taken for Sklearnex Model: {patched_end}\n'
      f'% of Time Saved: {round((1 - (patched_end / unpatched_end)) * 100)}%')
