"""Building the visualizations for each model and experiment"""

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import os

# set display theme to basic seaborn
sb.set_theme()

# set figure directory
figure_path = os.path.dirname(os.getcwd()) + "\\figures"

def add_spaces(word):
    """Add spaces before all upper case letters after the first

    :param word: some word with no spaces in it

    Variables:
        string: the first letter of the word submitted, to avoid checking it when looking for upper case chars

    :return: word with spaces
    """
    string = [word[0], ]
    for letter in word[1:]:
        if letter.isupper():
            string.append(f" {letter}")
        else:
            string.append(letter)
    return ''.join(string)


def histo_bar_plot(df, parameter: str, grouping: str = 'Outcome', axis=None):
    """Makes a histogram bar plot to be later added into a grid of plots

    :param df: data to create the bar plot from
    :param parameter: parameter to compare from the data
    :param grouping: how to split the data within the parameter; default is Outcome
    :param axis: which axis in the grid to place the plot

    Variables:
        plot: the plot produced
        xlabel: the x axis label
    """
    # set as categorical variable, sorted from highest to lowest count
    df[parameter] = pd.Categorical(df[parameter], categories=df[parameter].value_counts().index)

    # build plot
    plot = sb.countplot(data=df,
                        x=parameter,
                        hue=grouping,
                        ax=axis)

    # make xlabel by adding space before second (or greater) capital letter(s) in parameter name
    xlabel = add_spaces(parameter)

    # remove count from ylabel, set xlabel as the item in xlabels list at same item as current parameter
    plot.set(ylabel=None, xlabel=xlabel)


def build_plot_grid(df, parameters: list, grouping: str = 'Outcome', filename: str = 'no_name',
                    title: str = "", w: int = 3):
    """Building the grid of histo plots and saving it as a figure

    :param df: data to create the bar plot from
    :param parameter: parameter to compare from the data
    :param grouping: how to split the data within the parameter; default is Outcome
    :param filename: name to save the plot grid as
    :param title: title of the plot grid
    :param w: number of columns for the plot grid

    Variables:
        h: number of rows
        figure: the figure containing the grid
        grid: the grid of plots
        row: the current row
        col: the current col
        sum_tick_chars: the total # of characters in all the x tick labels
        last_col_ind: the col index of the last plot that has data in it and isn't an empty axis
    """
    # set height of grid as # of parameters / 3, unless there's remainder then add extra row
    h = len(parameters) // w if len(parameters) % w == 0 else len(parameters) // w + 1
    figure, grid = plt.subplots(h, 3, figsize=(15, 10))
    figure.suptitle(title)

    # set first plot location
    row, col = 0, 0

    for var in parameters:
        # if height is 1, then set as one-dimensional grid. Otherwise, set us two-dimensional
        if not h == 1:
            histo_bar_plot(df, parameter=var, grouping=grouping, axis=grid[row, col])
            # remove legend for all charts that aren't the first
            if [row, col] != [0, 0]:
                grid[row, col].get_legend().remove()

        else:
            histo_bar_plot(df, parameter=var, grouping=grouping, axis=grid[col])
            if [col] != [0]:
                grid[col].get_legend().remove()

        # rotate tick sizes if sum of all len of xticklabels is greater than 20
        sum_tick_chars = 0
        for value in df[var].value_counts().index:
            sum_tick_chars += len(str(value))
        if sum_tick_chars > 20:
            plt.setp(grid[row, col].get_xticklabels(), rotation=45,
                     horizontalalignment='right', rotation_mode='anchor')

        # if current graph is at last column in row, set next axis to be first column of next row
        if not col == w - 1:
            col += 1
        else:
            col, row = 0, row + 1

        # make charts with no data not visible
        last_col_ind = len(parameters) % w - 1
        if last_col_ind >= 0:
            for i in range(last_col_ind + 1, w):
                if h > 1:
                    grid[h - 1, i].set_axis_off()
                else:
                    grid[i].set_axis_off()

    # align everything properly given the rotated labels
    plt.tight_layout()

    # save figure, show figure, and clear the plot for future plots
    plt.savefig(f"{figure_path}\\{filename}.png")
    plt.clf()


def build_all_demographics(df, name):
    """Builds and saves a plot grid for the descriptors, symptoms, and comorbidities

    :param df: the data to build graphs from
    :param name: part of the name to save the file as
    """
    build_plot_grid(df=df,
                    parameters=['CriteriaConfirmation', 'AgeGroup', 'Education', 'ColorRace', 'Sex', 'OverSixty',
                                'Hospitalized', 'TravelBrasil', 'TravelInternational', 'SumSymptoms',
                                'SumComorbidities'],
                    filename=f"{name}_descriptors",
                    title='Descriptors in Training Set')

    build_plot_grid(df=df,
                    parameters=['Fever', 'BreathingDifficulty', 'Cough', 'RunnyNose', 'SoreThroat', 'Diarrhea',
                                'Headache'],
                    filename=f"{name}_symptoms",
                    title='Symptoms in Training Set')

    build_plot_grid(df=df,
                    parameters=['ComorbidityPulmonary', 'ComorbidityCardiac', 'ComorbidityRenal',
                                'ComorbidityDiabetes', 'ComorbiditySmoking', 'ComorbidityObesity'],
                    filename=f"{name}_comorbidities",
                    title='Comorbidities in Training Set')


def create_confusion_matrix_info(data):
    """Puts the total predictions and real outcomes into a format usable for a confusion matrix

    :param data: data to use

    Variables:
        all_models: list of all the model names
        all_real_and_preds: list with two items; the list of real outcomes, and the list of predictions

    :return: the list of model names, and the list of two lists; the real outcomes and the predictions
    """
    all_models = [type(model).__name__ for model in [d['Model'] for d in data]]
    all_real_and_preds = []
    for d in data:
        real_and_preds = [d['Real Outcomes'], d['Predictions']]
        all_real_and_preds.append(real_and_preds)

    return all_models, all_real_and_preds


def build_confusion_matrix(model, real: list, preds: list, axis=None):
    """Creates a confusion matrix

    :param model: the model, used for the title
    :param real: list of real outcomes
    :param preds: list of predictions
    :param axis: axis to place the confusion matrix in the grid

    Variables:
        mtrx: the confusion matrix
        htmp: the heatmap for the confusion matrix
    """
    mtrx = confusion_matrix(real, preds)
    htmp = sb.heatmap(mtrx / np.sum(mtrx), annot=True, fmt='.1%', cmap='Reds', ax=axis)

    htmp.set_xlabel('Predictions')
    htmp.xaxis.set_ticklabels(['Cured', 'Death'])

    htmp.set_ylabel('Real Outcomes')
    htmp.yaxis.set_ticklabels(['Cured', 'Death'])

    htmp.set_title(f'{model}')


def build_conf_mtrx_grid(models: list, all_real_and_preds: list, filename: str, title: str = '', w: int = 3):
    """Building the grid of confusion matrices and saving it as a figure

    :param models: list of model names for titles
    :param all_real_and_preds: list of two lists; real outcomes and predictions
    :param filename: name to save the plot grid as
    :param title: title of the plot grid
    :param w: number of columns for the plot grid

    Variables:
        h: number of rows
        figure: the figure containing the grid
        grid: the grid of plots
        row: the current row
        col: the current col
        last_col_ind: the col index of the last plot that has data in it and isn't an empty axis
    """
    # set height of grid as # of parameters / 3, unless there's remainder then add extra row
    h = len(all_real_and_preds) // w if len(all_real_and_preds) % w == 0 else len(all_real_and_preds) // w + 1
    figure, grid = plt.subplots(h, 3, figsize=(15, 8))
    figure.suptitle(title)

    # set first plot location
    row, col = 0, 0

    for i in range(len(all_real_and_preds)):
        # if height is 1, then set as one-dimensional grid. Otherwise, set us two-dimensional
        if not h == 1:
            build_confusion_matrix(models[i], all_real_and_preds[i][0], all_real_and_preds[i][1], axis=grid[row, col])

        else:
            build_confusion_matrix(models[i], all_real_and_preds[i][0], all_real_and_preds[i][1], axis=grid[col])

        # if current graph is at last column in row, set next axis to be first column of next row
        if not col == w - 1:
            col += 1
        else:
            col, row = 0, row + 1

        # make charts with no data not visible
        last_col_ind = len(all_real_and_preds) % w - 1
        if last_col_ind >= 0:
            for i in range(last_col_ind + 1, w):
                if h > 1:
                    grid[h - 1, i].set_axis_off()
                else:
                    grid[i].set_axis_off()

    # align everything properly given the rotated labels
    plt.tight_layout()

    # save figure, show figure, and clear the plot for future plots
    plt.savefig(f"{figure_path}\\{filename}.png")
    plt.clf()


def model_metric_bar_plot(data, metric, axis=None):
    """Building a bar plot to compare performances of models along a certain metric

    :param data: data to compare using
    :param metric: metric to compare along
    :param axis: axis to place the bar plot in the grid

    Variables:
        model_shortform: shortforms for the models when creating titles
        models: list of model shortforms
        metric_list: list of metrics to compare total
        df: dataframe of model names and metrics to use
        bp: barplot created
    """
    model_shortform = {'LogisticRegression': 'LogReg',
                       'LinearDiscriminantAnalysis': 'LDA',
                       'KNeighborsClassifier': 'KNN',
                       'DecisionTreeClassifier': 'DT',
                       'XGBClassifier': 'XGB',
                       'SVC': 'SVC'}
    models = []
    metric_list = []
    for dat in data:
        models.append(model_shortform.get(str(type(dat['Model']).__name__)))
        metric_list.append(dat[metric])

    df = pd.DataFrame(zip(models, metric_list),
                      columns=['Model', metric])

    bp = sb.barplot(x='Model',
                    y=metric,
                    data=df,
                    ax=axis)

    bp.set_ylabel(None)
    bp.set_xlabel(f'Mean {metric}')
    bp.set_ylim(.2, 1)


def model_metric_bar_grid(data, metrics: list, filename: str, title: str='', w: int=3):
    """Building the grid of model comparisons and saving it as a figure

    :param data: data to create the grid from
    :param metrics: metrics to compare with
    :param filename: name to save the plot grid as
    :param title: title of the plot grid
    :param w: number of columns for the plot grid

    Variables:
        h: number of rows
        figure: the figure containing the grid
        grid: the grid of plots
        row: the current row
        col: the current col
        last_col_ind: the col index of the last plot that has data in it and isn't an empty axis
    """
    # set height of grid as # of parameters / 3, unless there's remainder then add extra row
    h = len(metrics) // w if len(metrics) % w == 0 else len(metrics) // w + 1
    figure, grid = plt.subplots(h, 3, figsize=(15, 10))
    figure.suptitle(title)

    # set first plot location
    row, col = 0, 0

    for i in range(len(metrics)):
        # if height is 1, then set as one-dimensional grid. Otherwise, set us two-dimensional
        if h > 1:
            model_metric_bar_plot(data, metrics[i], axis=grid[row, col])
        else:
            model_metric_bar_plot(data, metrics[i], axis=grid[col])

        # if current graph is at last column in row, set next axis to be first column of next row
        if not col == w - 1:
            col += 1
        else:
            col, row = 0, row + 1

        # make charts with no data not visible
        last_col_ind = len(metrics) % w - 1
        if last_col_ind >= 0:
            for x in range(last_col_ind + 1, w):
                if h > 1:
                    grid[h - 1, x].set_axis_off()
                else:
                    grid[x].set_axis_off()


    # align everything properly given the rotated labels
    plt.tight_layout()

    # save figure, show figure, and clear the plot for future plots
    plt.savefig(f"{figure_path}\\{filename}.png")
    plt.clf()


def build_comparison_tables(data):
    """Building a table to compare metric scores for each model

    :param data: data to use for comparing models

    Variables:
        all_metrics: list of metrics to compare
        all_models: all model names
        all_acc: all accuracy scores
        all_roc: all ROC AUC scores
        all_avg_prc: all PR AUC scores
        all_prc: all precision scores
        all_rcl: all recall scores
        all_F1: all F1 scores
        df: final dataframe with all the scores

    :return: a dataframe with each metric and each score
    """
    all_metrics = ['Accuracy', 'ROC AUC', 'PR AUC', 'Precision', 'Recall', 'F1']
    all_models = [type(model).__name__ for model in [d['Model'] for d in data]]
    all_acc = [d['Accuracy'] for d in data]
    all_roc = [d['ROC AUC'] for d in data]
    all_avg_prc = [d['Average Precision'] for d in data]
    all_prc = [d['Precision'] for d in data]
    all_rcl = [d['Recall'] for d in data]
    all_F1 = [d['F1'] for d in data]

    df = pd.DataFrame(data=[all_acc, all_roc, all_avg_prc, all_prc, all_rcl, all_F1], columns=all_models)

    df.insert(loc=0, column='', value=pd.Series(all_metrics))

    return df
