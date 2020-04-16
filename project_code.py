"""
Authors: Joshua Yap and Keagan Anderson
This program manipulates PGA Tour data to produce relevant plots of driving performance
using '2019_data' from data_files.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['axes.grid'] = True

sns.set()
 
 
class Player:
    """
    Takes a data set and a Player’s name. Keeps track of the player’s
	stats in the data set and plots the driving distance vs the 
	accuracy of the player.
    """
    def __init__(self, data, name):
        self._data = data.loc[data["Player Name"] == name]
        self._data = merge_data(self._data)
 
    def plot_driving_ratio(self):
        self._data = plot_driving(self._data)
 
 
def plot_driving(data):
    """
    Takes a data set and plots the driving accuracy vs the driving
	distance for the dataset.
    """
    sns.lmplot(x='Distance', y='Accuracy (%)', data=data)
    plt.title('Driving Accuracy against Distance')
    plt.savefig('driving_plpot.png', bbox_inches='tight')
    plt.show()
 
 
def merge_data(data):
    """
    Saves rows of driving distance and accuracy variables from given dataframe as
    separate dataframes and merges them on player name and date.
    """
    driving_dist = data['Variable'] == 'Driving Distance - (AVG.)'
    driving_acc = data['Variable'] == 'Driving Accuracy Percentage - (%)'
    dist_data = data[driving_dist].rename(columns={'Value': 'Distance'})
    acc_data = data[driving_acc].rename(columns={'Value': 'Accuracy (%)'})
    merged = acc_data.merge(dist_data, left_on=['Player Name', 'Date'],
                            right_on=['Player Name', 'Date'])
    return merged
 
 
def course_difficulty(date, df):
    """
    Returns SG:OTT average for the field at a given tournament.
    (Date format yyyy-mm-dd)
    """
    is_sgott = df['Variable'] == 'SG: Off-the-Tee - (TOTAL SG:OTT)'
    filtered = df.loc[is_sgott]
    grouped = filtered.groupby("Date")["Value"].mean()
    date1 = grouped.index.get_loc(date)
    date2 = grouped.index.get_loc(date) - 1
    if date1 == 0:
        finished = grouped[date1] / 4
    else:
        finished = (grouped[date1] - grouped[date2]) / 4
    return finished
 
 
def filter_profile(data, profile):
    """
    Take a dataframe after it has been processed by merge_data, and a profile
    and returns a filtered dataframe of players with that profile. Valid profiles
    are 'long_inaccurate', 'long_accurate', 'short_inaccurate', 'short_inaccurate'.
    """
    player_aggregates = data[data['Date'] == '2019-08-25']
    if profile == 'long_inaccurate':
        acc_thresh = 55
        dist_thresh = 300
        criteria = (player_aggregates['Distance'] > dist_thresh) & (player_aggregates['Accuracy (%)'] < acc_thresh)

    elif profile == 'long_accurate':
        acc_thresh = 65
        dist_thresh = 300
        criteria = (player_aggregates['Distance'] > dist_thresh) & (player_aggregates['Accuracy (%)'] > acc_thresh)
    elif profile == 'short_inaccurate':
        acc_thresh = 60
        dist_thresh = 290
        criteria = (player_aggregates['Distance'] < dist_thresh) & (player_aggregates['Accuracy (%)'] < acc_thresh)
    elif profile == 'short_accurate':
        acc_thresh = 70
        dist_thresh = 280
        criteria = (player_aggregates['Distance'] < dist_thresh) & (player_aggregates['Accuracy (%)'] > acc_thresh)
    else:
        return 'Error: enter valid profile'
    filtered = player_aggregates[criteria]
    result = data[data['Player Name'].isin(filtered['Player Name'])]
    return result


def compute_profile_performance(df, profile):
    """
    Takes a dataset and a profile type and sorts the data set
	To compare the given profile type to different difficulty
	Courses. Returns a dataset with Adjusted Strokes Gained per player
    """
    data = merge_data(df)
    filtered = filter_profile(data, profile)
    filtered = filtered[filtered["Date"] != "2019-03-24"]
    filtered["difficulty"] = filtered['Date'].apply(course_difficulty, df=df)
    merged = df.merge(filtered, left_on=["Player Name", "Date"], right_on=["Player Name", "Date"], how="right")
    is_sgott = merged['Variable'] == 'SG: Off-the-Tee - (TOTAL SG:OTT)'
    filtered_sgott = merged[is_sgott]
    temp_sgott = filtered_sgott[['Player Name', 'Date', 'Value']]
    temp_sgott = temp_sgott.sort_values(by='Date')
    # filter by name
    temp = temp_sgott.groupby('Player Name')['Value'].diff()
    merged_temp = temp_sgott.merge(temp, left_index=True, right_index=True, how='left')
    merged_sgott = filtered_sgott.merge(merged_temp, left_on=['Player Name', 'Date'], right_on=['Player Name', 'Date'], how='left')
    merged_sgott.fillna(0, inplace=True)
    merged_sgott['Value_y'] = merged_sgott['Value_y'] / 4
    merged_sgott = merged_sgott[merged_sgott['Value_y'] != 0]
    merged_sgott['Adjusted_SG'] = merged_sgott['Value_y'] - merged_sgott['difficulty']
    return merged_sgott


def plot_difficulty_and_type(df, profile):
    """
    Plots a profile type against all the different course
	difficulties.
    """
    sg_data = compute_profile_performance(df, profile)
    # filter out dates where players did not play
    print(sg_data.head())
    sns.relplot(x='difficulty', y='Adjusted_SG', data=sg_data, kind='scatter', hue='Player Name')
    plt.show()


def combined_plots(df):
    """
    Plots an assortment of graphs to display the difference in
	Driving accuracy and Driving Distance Vs Course difficulty
    """
    short_acc = compute_profile_performance(df, "short_accurate")
    short_acc_x = short_acc['difficulty']
    short_acc_y = short_acc['Adjusted_SG']

    short_inacc = compute_profile_performance(df, "short_inaccurate")
    short_inacc_x = short_inacc['difficulty']
    short_inacc_y = short_inacc['Adjusted_SG']

    long_acc = compute_profile_performance(df, "long_accurate")
    long_acc_x = long_acc['difficulty']
    long_acc_y = long_acc['Adjusted_SG']

    long_inacc = compute_profile_performance(df, "long_inaccurate")
    long_inacc_x = long_inacc['difficulty']
    long_inacc_y = long_inacc['Adjusted_SG']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle('Relative Strokes Gained against Difficulty for each Profile')

    sns.regplot(x='difficulty', y='Adjusted_SG', data=short_acc, ax=ax1)
    ax1.set_title('Short Accurate')
    ax1.set_xlim(0.15, -0.025)
    ax1.set_ylabel('Relative SG: OTT')
    ax1.set_xlabel('')
    sns.regplot(x='difficulty', y='Adjusted_SG', data=short_inacc, ax=ax2)
    ax2.set_title('Short Inaccurate')
    ax2.set_xlim(0.15, -0.025)
    ax2.set_ylabel('')
    ax2.set_xlabel('')
    sns.regplot(x='difficulty', y='Adjusted_SG', data=long_acc, ax=ax3)
    ax3.set_title('Long Accurate')
    ax3.set_xlim(0.15, -0.025)
    ax3.set_xlabel('Course difficulty')
    ax3.set_ylabel('Relative SG: OTT')
    sns.regplot(x='difficulty', y='Adjusted_SG', data=long_inacc, ax=ax4)
    ax4.set_title('Long Inaccurate')
    ax4.set_xlim(0.15, -0.025)
    ax4.set_xlabel('Course difficulty')
    ax4.set_ylabel('')

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)
    plt.savefig('tile_profiles.png', bbox_inches='tight')
    plt.show()

    plt.subplots()
    plt.scatter(short_acc_x, short_acc_y, s=12, label='Short Accurate')
    plt.scatter(short_inacc_x, short_inacc_y, s=12, label='Short Inaccurate')
    plt.scatter(long_acc_x, long_acc_y, s=12, label='Long Accurate')
    plt.scatter(long_inacc_x, long_inacc_y, s=12, label='Long Inaccurate')
    plt.legend()
    plt.title('Combined plots of profiles')
    plt.xlim(0.15, -0.025)
    plt.xlabel('Course difficulty')
    plt.ylabel('Relative SG: OTT')
    plt.savefig('combined_profiles.png', bbox_inches='tight')
    plt.show()

    # Aggregate SG of each profile for each course difficulty
    agg_short_acc = short_acc.groupby('difficulty')['Adjusted_SG'].mean()
    agg_short_acc = agg_short_acc.reset_index(name='Adjusted_SG')
    agg_short_acc['Profile'] = 'Short Accurate'
    agg_short_inacc = short_inacc.groupby('difficulty')['Adjusted_SG'].mean()
    agg_short_inacc = agg_short_inacc.reset_index(name='Adjusted_SG')
    agg_short_inacc['Profile'] = 'Short Inaccurate'
    agg_long_acc = long_acc.groupby('difficulty')['Adjusted_SG'].mean()
    agg_long_acc = agg_long_acc.reset_index(name='Adjusted_SG')
    agg_long_acc['Profile'] = 'Long Accurate'
    agg_long_inacc = long_inacc.groupby('difficulty')['Adjusted_SG'].mean()
    agg_long_inacc = agg_long_inacc.reset_index(name='Adjusted_SG')
    agg_long_inacc['Profile'] = 'Long Inaccurate'

    # Combine into 1 dataframe
    agg_all = pd.concat([agg_short_acc, agg_short_inacc, agg_long_acc, agg_long_inacc], ignore_index=True)
    with sns.axes_style('whitegrid'):
        sns.lmplot(x='difficulty', y='Adjusted_SG', data=agg_all, hue='Profile')
        plt.title('Aggregate Strokes Gained for each Profile at varying Difficulty')
        plt.xlim(0.15, -0.025)
        plt.ylabel('Relative Strokes Gained')
        plt.xlabel('Difficulty')
        plt.savefig('agg_profiles.png', bbox_inches='tight')
        plt.show()


def main():
    df = pd.read_csv('data_files/2019_data.csv')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    #data = merge_data(df)
    #print(data)
    #print(course_difficulty('2019-08-25', df))
    #plot_driving(data[data['Date'] == '2019-08-25'])
    #plot_driving(filter_profile(data, 'short_accurate'))
    #test = Player(df, "Jim Furyk")
    #test.plot_driving_ratio()
    #plot_difficulty_and_type(df, "long_accurate")
    #compute_profile_performance(df, 'long_accurate').head()
    combined_plots(df)
 
if __name__ == '__main__':
    main()
