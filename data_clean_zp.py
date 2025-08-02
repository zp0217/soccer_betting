#converting date for 2016- this season is the only one that has different date format 

#import pandas as pd
#t1 = pd.read_csv('data/2016.csv')
#t1['Date'] = pd.to_datetime(t1['Date'], format='%d/%m/%y')
#t1['Date'] = t1['Date'].dt.strftime('%d/%m/%Y')
#t1.to_csv('data/2016.csv')

#getting all seasons and merge it to one csv
import pandas as pd
import glob
import os
from pathlib import Path
path = Path("data")

csv_files = glob.glob(os.path.join(path, '*.csv'))
csv_files.sort()
for i, file in enumerate(csv_files, start=1):
    globals()[f'df{i}'] = pd.read_csv(file)

df_list = [globals()[f'df{i}'] for i in range(1, len(csv_files) + 1)]

bet = pd.concat(df_list, ignore_index=True)

#changing date format to yyyy-mm-dd
bet['Date'] = pd.to_datetime(bet['Date'], dayfirst=True).dt.strftime('%Y-%m-%d')
bet[['Date']].head()
bet = bet.rename(columns={
    'HomeTeam': 'Home',
    'AwayTeam': 'Away'
})

m1 = pd.read_csv('raw/matches.csv')
df_filtered = m1[m1["Season"] >= "2015/2016"]
m1 = df_filtered[df_filtered["Date"].notna() & (df_filtered["Date"].str.strip() != "")]
m1 =  m1.drop(columns=["Unnamed: 0","Unnamed: 0.1","Attendance","Venue","Season"], errors="ignore")
m1 = m1.rename(columns={
    'xG': 'Home_xG',
    'xG.1': 'Away_xG',
    'Home Goals':'home_goals',
    'Away Goals': 'away_goals'
})
m1.to_csv('m1.csv')

m2 = pd.read_csv('raw/xg_2024.csv')

# Rename m2 columns to match m1
m2_renamed = m2.rename(columns={
    'HomeGoals': 'home_goals',
    'AwayGoals': 'away_goals'
})

# Reorder m2 columns to match m1
m2_reordered = m2_renamed[m1.columns]

# Concatenate the two DataFrames
seasons = pd.concat([m1, m2_reordered], ignore_index=True)
seasons = seasons.sort_values(by='Date').reset_index(drop=True)
#team name is different
team_name_mapping = {
    'Newcastle': 'Newcastle Utd',
    'Man City': 'Manchester City',
    'Man United': 'Manchester Utd',
    'Sheffield Utd': 'Sheffield United',
    'Spurs': 'Tottenham',
    'Wolves': 'Wolverhampton Wanderers',
    'West Brom': 'West Bromwich Albion',
    'West Ham Utd': 'West Ham',
    'Brighton': 'Brighton and Hove Albion',
    'Leeds': 'Leeds United',
    'Leicester': 'Leicester City',
    'Norwich': 'Norwich City',
    'Cardiff': 'Cardiff City',
    'Hull': 'Hull City',
    'QPR': 'Queens Park Rangers',
    'Blackpool': 'Blackpool FC',
    'Birmingham': 'Birmingham City',
    'Swansea': 'Swansea City',
    'Stoke': 'Stoke City',
    'Bolton': 'Bolton Wanderers',
    'Wigan': 'Wigan Athletic',
    'Barnsley': 'Barnsley FC',
    'Huddersfield': 'Huddersfield Town',
    'Luton': 'Luton Town',
    "Ipswich": "Ipswich Town",
    "Nott'm Forest": "Nottingham Forest",
    "Nott'ham Forest":"Nottingham Forest"
}
bet['Home'] = bet['Home'].replace(team_name_mapping)
bet['Away'] = bet['Away'].replace(team_name_mapping)
seasons['Home'] = seasons['Home'].replace(team_name_mapping)
seasons['Away'] = seasons['Away'].replace(team_name_mapping)
seasons = seasons.drop(columns=['home_goals','away_goals'], errors="ignore")

total = pd.merge(bet, seasons, on=["Date", "Home", "Away"], how="inner")
all_seasons = total[['FTHG','FTAG','FTR','HST','AST','HC','AC','B365H',
               'B365H','B365A','HC','AC','HF','AF','B365>2.5',
               'B365<2.5','Date','Home_xG','Away_xG']]

from sklearn.preprocessing import LabelEncoder
all_seasons = all_seasons.copy()
le = LabelEncoder()
all_seasons['FTR_encoded'] = le.fit_transform(all_seasons['FTR'])
one_hot_encoded = pd.get_dummies(all_seasons['FTR'], prefix='FTR')
betting_result = pd.concat([all_seasons, one_hot_encoded], axis=1)
one_hot_columns = ['FTR_A', 'FTR_D', 'FTR_H']
betting_result[one_hot_columns] = betting_result[one_hot_columns].astype(int)
betting_result.head()

# xg_goals: 15/16,16/17 season missing -remove it 
batting_result= betting_result.dropna(subset=['Home_xG', 'Away_xG'])
batting_result.to_csv('fianl_data.csv')