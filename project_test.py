"""
Author: Joshua Yap
This file tests the functions implemented in project main using
datasets 'mini_data' and 'tiny_data' in data_files.
"""
from cse163_utils import assert_equals
from cse163_utils import check_approx_equals
import part1 # Name of main file
import math
import pandas as pd
import csv

# Generate smaller dataset from 2019_data_driving
def reduce_dataset(df):
    keep_vars = ['Driving Distance - (AVG.)', 'Driving Accuracy Percentage - (%)', 'SG: Off-the-Tee - (AVERAGE)',
                 'SG: Off-the-Tee - (TOTAL SG:OTT)']
    data = df.loc[df['Variable'].isin(keep_vars)]
    names = ['Ryan Armour', 'Jim Furyk', 'Brian Gay', 'Ollie Schneiderjans', 'Chez Reavie',
             'Phil Mickelson', 'Cameron Champ', 'Rory McIlroy', 'Ryan Moore','Michael Kim']
    data = data.loc[data['Player Name'].isin(names)]
    return data


def test_merge_data(tiny_data):
    player_1 = tiny_data[tiny_data['Player Name'] == 'Player1']
    d = {'Player Name': ['Player1'], 'Date': ['8/25/2019'], 'Statistic_x': ['Driving Accuracy Percentage'], 
         'Variable_x': ['Driving Accuracy Percentage - (%)'], 'Accuracy (%)': [80], 'Statistic_y': ['Driving Distance'], 
         'Variable_y': ['Driving Distance - (AVG.)'], 'Distance': [290]}
    check_approx_equals(d, part1.merge_data(player_1).to_dict('list'))


def test_plot_driving(merged):
    part1.plot_driving(merged)


def test_course_difficulty(mini_data):
    check_approx_equals(-0.0226, part1.course_difficulty('8/18/2019', mini_data))


def test_plot_difficulty_and_type(mini_data):
    part1.plot_difficulty_and_type(mini_data, 'difficult', 'short_accurate')


def test_compute_profile_performance(mini_data):
    print(part1.compute_profile_performance(mini_data, 'short_accurate').head())

def main():
    mini_data = pd.read_csv('data_files/mini_data.csv')
    tiny_data = pd.read_csv('data_files/tiny_data.csv')
    #data = reduce_dataset(df)
    #data.to_csv('data_files/mini_data.csv', index=False)
    #test_merge_data(tiny_data)
    merged_mini = part1.merge_data(mini_data)
    merged_tiny = part1.merge_data(tiny_data)
    #test_plot_driving(merged_mini)
    #test_plot_driving(merged_tiny)
    #test_course_difficulty(mini_data)
    #test_plot_difficulty_and_type(mini_data)
    test_compute_profile_performance(mini_data)

if __name__ == '__main__':
    main()