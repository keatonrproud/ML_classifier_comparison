"""Processes and prepares data for upcoming analyses"""

import os
import pandas as pd
import numpy as np

# increase max columns when viewing the dataset printed in the console
pd.set_option('display.max_columns', 30)


def load_data(filepath_from_script: str = '\data\project_data.csv'):
    """Load data from csv -- only use columns
    
    :param filepath_from_script: the filepath to the data
    
    Variables:
        cols: columns to read in

    :return: DataFrame processed from the data
    """
    # columns to read in
    cols = ("DataNotificacao", "Evolucao", "CriterioConfirmacao", "StatusNotificacao", "FaixaEtaria", "Sexo", "RacaCor",
        "Escolaridade", "Febre", "DificuldadeRespiratoria", "Tosse", "Coriza", "DorGarganta", "Diarreia", "Cefaleia",
        "ComorbidadePulmao", "ComorbidadeCardio", "ComorbidadeRenal", "ComorbidadeDiabetes", "ComorbidadeTabagismo",
        "ComorbidadeObesidade", "FicouInternado", "ViagemBrasil", "ViagemInternacional")

    print('loading data...')

    return pd.read_csv("https://bi.s3.es.gov.br/covid19/MICRODADOS.csv", delimiter=";", encoding='cp1252',
                       usecols=lambda x: x in cols)  # only loads the required columns from the dataset


def filter_df(data):
    """Remove rows with data that don't help analyses

    :param data: the data to filter

    Variables:
        new_data: the new data after changes
        keep_outcomes: list of outcomes to keep in the data

    :return: the new data
    """
    # only keep rows that are closed and have a final outcome of cured or a death
    new_data = data[data['StatusNotificacao'] == 'Encerrado']  # only keep rows marked as closed (cured or dead)

    # only keep outcomes marked as cured, COVID death, or other death
    keep_outcomes = ["Cura", 'Óbito por outras causas', 'Óbito pelo COVID-19']
    new_data = new_data[(new_data['Evolucao'].isin(keep_outcomes))]

    return new_data


def replace_portuguese(data):
    """Convert Portuguese to English for easier analysis and exploration

    :param data: the data to convert to English
    """
    # rename column headers
    data.columns = ['NotificationDate', 'Outcome', 'CriteriaConfirmation', 'AgeGroup', 'Sex', 'ColorRace', 'Education',
                    'Fever', 'BreathingDifficulty', 'Cough', 'RunnyNose', 'SoreThroat', 'Diarrhea', 'Headache',
                    'ComorbidityPulmonary', 'ComorbidityCardiac', 'ComorbidityRenal', 'ComorbidityDiabetes',
                    'ComorbiditySmoking', 'ComorbidityObesity', 'Hospitalized', 'TravelBrasil',
                    'TravelInternational']

    # replace Portuguese with English words, and No or Yes with 0 or 1
    data.replace(
        {"Não": "No", "Sim": "Yes", "I": "N/A", "-": "N/A", "Ignorado": "N/A", "Não se aplica": "N/A",
         "Não Informado": "N/A", "Cura": 0, "Óbito por outras causas": 1,
         "Preta": "Black", "Branca": "White", "Parda": "Brown", "Indigena": "Indigenous", "Amarela": "Yellow",
         "Clinico": "Clinic", "Laboratorial": "Lab", "Clinico Epdemiologico": "Epidem.",
         "Ensino médio completo (antigo colegial ou 2º grau ) ": "HS",
         "5ª à 8ª série incompleta do EF (antigo ginásio ou 1º grau)": "I Gr. 8",
         "4ª série completa do EF (antigo primário ou 1º grau)": "Gr. 4",
         "Ensino médio incompleto (antigo colegial ou 2º grau )": "I HS",
         "1ª a 4ª série incompleta do EF (antigo primário ou 1º grau)": "I Gr. 4",
         "Educação superior incompleta ": "I Higher",
         "Educação superior completa": "Higher Ed.",
         "Ensino fundamental completo (antigo ginásio ou 1º grau) ": "Gr. 1",
         "Analfabeto": "None",
         "Óbito pelo COVID-19": 1, "0 a 4 anos": "0-4", "05 a 9 anos": "5-9",
         "10 a 19 anos": "10-19", "20 a 29 anos": "20-29", "30 a 39 anos": "30-39",
         "40 a 49 anos": "40-49", "50 a 59 anos": "50-59", "60 a 69 anos": "60-69",
         "70 a 79 anos": "70-79", "80 a 89 anos": "80-89",
         "90 anos ou mais": "90+"},
        inplace=True)


def set_dtypes(data):
    """Set dtypes and set missing values for int columns

    :param data: the data to change
    """
    for i in data.columns:
        # if column has no information or a single dash, mark as N/A
        data[i].replace({'No information': "N/A",
                         '-': "N/A"},
                        inplace=True)
        data[i].astype(str)


def clean_data(data):
    """Prepare data for further analysis

    :param data: the data to clean

    Variables:
        new_data: the updated data

    :return: the new data after cleaning

    """
    # filter out rows missing required information or not closed
    new_data = filter_df(data)

    # remove StatusNotification column from dataset
    new_data.drop(columns='StatusNotificacao',
                  inplace=True)

    # rename headers and replace commonly used Portuguese words with English for easier analysis and exploration
    replace_portuguese(new_data)

    # set dtypes of columns and add 999 as missing number for integer columns
    set_dtypes(new_data)

    return new_data


def sum_true_columns(data, new_col: object, cols_to_sum: list):
    """Adds 1 to the new_col for each col_to_sum that is equal to Yes

    :param data: the data to sum columns in
    :param new_col: the column to sum to
    :param cols_to_sum: the columns to sum from

    Variables:
        data[new_col]: the new column
    """
    data[new_col] = 0
    for i in cols_to_sum:
        data[new_col] += data[i] == "Yes"


def add_summary_variables(data):
    """Add new variables -- sum of comorbidities, sum of symptoms

    :param data: the data to filter

    Variables:
        over60_ages: the ages included in the 'over60' group from the AgeGroup column
        data['OverSixty']: the OverSixty column in the data
    """

    # create sum of comorbidities and symptoms, and for each, add 1 if the column at that row is equal to 1 (is true)
    sum_true_columns(data,
                     new_col='SumComorbidities',
                     cols_to_sum=['ComorbidityPulmonary', 'ComorbidityCardiac', 'ComorbidityRenal',
                                  'ComorbidityDiabetes', 'ComorbiditySmoking', 'ComorbidityObesity'])

    sum_true_columns(data,
                     new_col='SumSymptoms',
                     cols_to_sum=['Fever', 'BreathingDifficulty', 'Cough', 'RunnyNose',
                                  'SoreThroat', 'Diarrhea', 'Headache'])

    # create new column if first two characters of that row's AgeGroup (as an int) is greater than or equal to 60
    over60_ages = ['60-69', '70-79', '80-89', '90+']
    data['OverSixty'] = np.where(data['AgeGroup'].isin(over60_ages),
                                 "Yes", "No")


def create_exp_datasets(all_data):
    """Split datasets for analyses matching the paper's datasets

    :param all_data: all the data to use for the two experiments

    Variables:
        exp1: the Exp1 data
        exp2: the Exp2 data

    :return: the Exp1 and the Exp2 DataFrames
    """
    # create exp_1 dataset (Feb 28 - May 23)
    exp1 = all_data[
        (all_data['NotificationDate'] < "2020-05-24") & (all_data['NotificationDate'] > "2020-02-28")].copy()

    # create exp2 dataset (May 24 - May 31)
    exp2 = all_data[
        (all_data['NotificationDate'] < "2020-06-01") & (all_data['NotificationDate'] > "2020-05-23")].copy()

    return exp1, exp2


def main():
    """Run overarching functions

    Variables:
        df: the entire DataFrame
        exp1: the Exp1 DataFrame
        exp2: the Exp2 DataFrame

    :return: Exp1 data, Exp2 data, and all the data
    """

    df = load_data()

    df = clean_data(df)

    add_summary_variables(df)

    exp1, exp2 = create_exp_datasets(df)

    for d in exp1, exp2:
        del d['NotificationDate']

    return exp1, exp2, df
