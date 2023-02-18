"""Processes and prepares data for upcoming analyses"""
import os
import pandas as pd
import numpy as np

# list of columns to read in
COLS = ("DataNotificacao", "Evolucao", "CriterioConfirmacao", "StatusNotificacao", "FaixaEtaria", "Sexo", "RacaCor",
        "Escolaridade", "Febre", "DificuldadeRespiratoria", "Tosse", "Coriza", "DorGarganta", "Diarreia", "Cefaleia",
        "ComorbidadePulmao", "ComorbidadeCardio", "ComorbidadeRenal", "ComorbidadeDiabetes", "ComorbidadeTabagismo",
        "ComorbidadeObesidade", "FicouInternado", "ViagemBrasil", "ViagemInternacional")

# increase max columns when viewing the dataset printed in the console
pd.set_option('display.max_columns', 30)


def load_data(filepath_from_script: str = '\data\project_data.csv', cols_to_load: list = COLS):
    """Load data from csv -- only use columns"""
    return pd.read_csv("https://bi.s3.es.gov.br/covid19/MICRODADOS.csv",
                       delimiter=";",
                       encoding='cp1252',
                       usecols=lambda x: x in cols_to_load)  # only loads the required columns from the dataset

import time

start = time.time()
print(load_data())
print(time.time() - start)


def filter_df(data):
    """Remove rows with data that will not help analyses"""
    # only keep rows that are closed and have a final outcome of cured or a death
    new_data = data[
        data['StatusNotificacao'] == 'Encerrado']  # only keep rows that have been marked as close (cured or dead)

    # only keep outcomes marked as cured, COVID death, or other death
    keep_outcomes = ["Cura", 'Óbito por outras causas', 'Óbito pelo COVID-19']
    new_data = new_data[(new_data['Evolucao'].isin(keep_outcomes))]

    return new_data


def replace_portuguese(data):
    """Convert Portuguese to English for easier analysis and exploration"""
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
    """Set dtypes and set missing values for int columns"""
    for i in data.columns:
        # if column has no information or a single dash, mark as N/A
        data[i].replace({'No information': "N/A",
                         '-': "N/A"},
                        inplace=True)
        data[i].astype(str)


def clean_data(data):
    """Prepare data for further analysis"""
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
    """Adds columns from cols_to_sum that are equal to 1 (i.e. true) and avoids missing values and 0s"""
    data[new_col] = 0
    for i in cols_to_sum:
        data[new_col] += data[i] == "Yes"


def add_summary_variables(data):
    """Add new variables -- sum of comorbidities, sum of symptoms"""

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
    """Split datasets for analyses matching the paper's datasets"""
    # create exp_1
    exp1 = all_data[
        (all_data['NotificationDate'] < "2020-05-24") & (all_data['NotificationDate'] > "2020-02-28")].copy()

    # create exp2 dataset (May 24 - May 31)
    exp2 = all_data[
        (all_data['NotificationDate'] < "2020-06-01") & (all_data['NotificationDate'] > "2020-05-23")].copy()

    return exp1, exp2


def main():
    """Run overarching functions"""
    df = load_data()

    df = clean_data(df)

    add_summary_variables(df)

    exp1, exp2 = create_exp_datasets(df)

    for d in exp1, exp2:
        del d['NotificationDate']

    return exp1, exp2, df
