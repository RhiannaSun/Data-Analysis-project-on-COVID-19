# Yiling Sun
# This code to perform various analytical operations on
# data of COVID and SARS from JHU database, includes
# data visualization
import pandas as pd 
import numpy as np

import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sns.set()

"""
COVID
"""
def load_in_global_data():
    """
    This function load in the global map
    Only keep geometry, the name and population of countries
    """
    global_map = gpd.read_file('ne_10m_admin_0_countries.shp')
    global_map = global_map[['NAME','geometry','POP_EST']]
    return global_map

def load_in_China_data():
    """
    This function load in the China map
    Only kepp geometry, the name of province
    Change the name of some provinces to match with further data
    """
    china_map = gpd.read_file('CHGIS_V4_1997_PROV_PGN.shp')
    china_map = china_map[['NAME_PY','geometry']]
    china_map = china_map.replace('Chongqing Shi', 'Chongqing')
    china_map = china_map.replace('Macau', 'Macao')
    china_map = china_map.replace('Neimenggu','Inner Mongolia')
    china_map = china_map.replace('Taiwan','Tibet')
    return china_map


def load_in_COVID_data():
    """
    load in and clean the COVID data
    """
    df = pd.read_csv('time_series_19-covid-Confirmed.csv')
    df = clean_df(df)
    return df


def merge_COVID_data(df, global_map):
    """
    This function keep only country-wise information 
    and merge with the geometry data 
    delete all NAN and keep only data before 3/13/20
    """
    filtered = df.drop(['Province/State', 'Lat', 'Long'], axis=1)
    grouped = filtered.groupby(by='Country/Region').sum().reset_index()
    COVID_merged = global_map.merge(grouped, right_on = 'Country/Region', left_on ='NAME',how='right')
    COVID_merged = COVID_merged[COVID_merged['geometry'] != None]
    COVID_merged = COVID_merged.loc[:,'NAME':'3/13/20']
    return COVID_merged    


def clean_df(df):
    """
    This funtion change names of df to match with further data
    """
    df = df.replace('US','United States of America')
    df = df.replace('UK','United Kingdom')
    df = df.replace('Viet Nam', 'Vietnam')
    df = df.replace('North Macedonia', 'Macedonia')
    df = df.replace('Mainland China', 'China')
    df = df.replace('Czech Republic', 'Czechia')
    df = df.replace ('Saint Martin', 'St-Martin')
    df = df[df['Country/Region'] != 'St. Martin']
    df = df.replace ('Iran (Islamic Republic of)','Iran')
    df = df.replace('Vatican City','Vatican')
    df = df.replace('occupied Palestinian territory','Palestine')
    df = df.replace('Macau','Macao')
    df = df[df['Country/Region'] != 'Macao SAR']
    df = df.replace('Hong Kong SAR','Hong Kong')
    df = df.replace('Taipei and environs','Taiwan')
    df = df.replace('Russian Federation','Russia')
    df = df.replace('Bosnia and Herzegovina','Bosnia and Herz.')
    df = df.replace('Dominican Republic','Dominican Rep.')
    df = df.replace('Saint Barthelemy','St-BarthÃƒÂ©lemy')
    df = df.replace('Faroe Islands','Faeroe Is.')
    df = df.replace('Korea, South','South Korea')
    df = df.replace('Republic of Moldova','Moldova')
    df = df.replace('Saint Barthelemy','St-BarthÃƒÂ©lemy')  
    return df


def plot_COVID_global(global_map, COVID_merged):
    """
    This function plot the global situation of COVID-19
    """
    fig, ax = plt.subplots(1, figsize=(15,7))
    global_map.plot(ax=ax, figsize=(15,7), color='#EEEEEE')
    COVID_merged.plot(ax=ax, column = '3/13/20', figsize=(15,7), legend=True, vmin=1, vmax= 15000,
            cmap = sns.cm.rocket_r)
    plt.title('Current Infection of COVID-19 World Wide')
    fig.savefig('COVID-19_global.png')


def select_top10_COVID_progress(COVID_merged):
    """
    This function select the top 10 serious country at 3/13/2020
    select the data from 1/22/2020 to 3/13/2020
    prepare data for further plotting
    """
    top10 = COVID_merged.sort_values('3/13/20',ascending=False)['NAME'].head(10)

    selected = COVID_merged[COVID_merged['NAME'].isin(top10)]
    filtered = selected.drop(['geometry', 'Country/Region','POP_EST'], axis = 1)
    filtered = filtered.transpose()
    filtered.columns = filtered.iloc[0]
    filtered = filtered.drop(filtered.index[0])

    countries = list()
    for i in top10:
        subdf = filtered[[i]]
        subdf.columns = ['num']
        subdf['country'] = i
        subdf['date'] = pd.to_datetime(pd.date_range("1 22 2020", periods=52, freq="D"))
        countries.append(subdf)
    countries = pd.concat(countries)
    countries = countries[['num','country','date']]
    countries['num'] = countries['num'].astype(int)
    return countries 


def plot_top10_COVID_progress(countries):
    """
    This function plots the progress of the top10 serious countries 
    from 1/22/2020 to 3/13/2020.
    It generates two plot. another one is w/o China
    """
    fig, [ax1, ax2] = plt.subplots(2, figsize=(10,10), ncols=1)
    sns.lineplot(ax = ax1, x = 'date', y = 'num', hue = 'country', data = countries)
    ax1.set_title('Growth of Confirmed Cases of COVID-19 in Top 10 Countries')
    sns.lineplot(ax = ax2, x = 'date', y = 'num', hue = 'country', data = countries[(countries['country'] != 'China') & (countries['date']>'2020-2-22')])
    plt.xticks(rotation=-45)
    fig.savefig('COVID-19_top10.png')


def merge_China_data(china_map, COVID_data):
    """
    This function selects data in China 
    and merge with the geometry data
    """
    china_confirmed = COVID_data[COVID_data['Country/Region'] == 'China']
    china_COVID_merged = china_map.merge(china_confirmed,left_on = 'NAME_PY', 
                                         right_on = 'Province/State', how='right')

    return china_COVID_merged


def plot_China_COVID(china_map, china_COVID_merged):
    """
    This function plot the number of infectious in each provinces in China
    at 3/13/20
    """
    fig, ax = plt.subplots(1, figsize=(15,7))
    china_map.plot(ax=ax, figsize=(15,7), color='#EEEEEE')
    china_COVID_merged.plot(ax=ax, column = '3/13/20', figsize=(15,7), legend=True, 
                            vmin=1, vmax= 15000,cmap = sns.cm.rocket_r)
    plt.title('Confirmed cases of COVID-19 in China')
    fig.savefig('COVID-19_China.png')

def load_clean_china_COVID_ratio():
    """
    load and clean the time series data of confirmed, death and recover
    in each provinces in China
    calculate the death and recover ratio 
    """
    confirmed = pd.read_csv('time_series_19-covid-Confirmed.csv')
    confirmed = confirmed[confirmed['Country/Region'] == 'China']
    confirmed = confirmed[['Province/State','3/13/20']]

    death = pd.read_csv('time_series_19-covid-Deaths.csv')
    death = death[death['Country/Region'] == 'China']
    death = death[['Province/State','3/13/20']]

    recover = pd.read_csv('time_series_19-covid-Recovered.csv')
    recover = recover[recover['Country/Region'] == 'China']
    recover = recover[['Province/State','3/13/20']]

    total = confirmed.merge(death, on = 'Province/State')
    total = total.merge(recover, on = 'Province/State')
    total.columns = ['Province/State', 'total', 'death', 'recover']
    total['death_ratio'] = total['death'] / total['total']
    total['recover_ratio'] = total['recover'] / total['total']
    return total
    

def plot_COVID_china_ratio(total):
    """
    This function plot the bar plot of death rate and recover rate
    """
    top_death = total.sort_values('death_ratio',ascending=False).head(15)
    top_recover = total.sort_values('recover_ratio').head(15)

    sns.set()
    sns.catplot(x='Province/State', y= 'death_ratio', kind='bar', height=3, aspect=3, data=top_death)
    plt.xticks(rotation=-45)
    plt.title('Death Ratio in Top 10 provinces in China')
    plt.savefig('COVID-19_china_death_ratio')

    sns.catplot(x='Province/State', y= 'recover_ratio', kind='bar', height=3, aspect=3, data=top_recover)
    plt.xticks(rotation=-45)
    plt.title('Recover Ratio in Least 10 provinces in China')
    plt.savefig('COVID-19_china_recover_ratio')
    
def load_clean_COVID_gender():
    """
    This funtion load individual data with gender information
    Prepared the gender info for further plotting
    """
    COVID19_gender_data = pd.read_csv('COVID19_open_line_list.csv')
    COVID19_gender_data.dropna(subset=['sex'])
    COVID19_gender_data = pd.get_dummies(COVID19_gender_data, columns=['sex'])
    COVID19_gender_data = COVID19_gender_data.groupby('country')[['sex_female', 'sex_male']].sum()
    COVID19_gender_data['sex_sum'] = COVID19_gender_data['sex_male'] + COVID19_gender_data['sex_female']
    top10_gender = COVID19_gender_data.nlargest(10, 'sex_sum')
    return top10_gender

def plot_COVID_gender(top10_gender):
    """
    This function plot the gender info of COVID
    """
    plt.figure(figsize=(10,7))
    top10_gender[['sex_male', 'sex_female']].plot(kind='bar')
    plt.title('Gender of Infectious People in Top 10 Country')
    plt.xticks(rotation=45)
    plt.savefig('COVID-19_gender')
    
    
def load_clean_COVID_age():
    """
    This functions load the individual data with age and death info 
    prepare data for further plotting
    """
    line_df = pd.read_csv('COVID19_line_list_data.csv')
    filtered = line_df[['age', 'death']]
    filtered = filtered[~filtered['age'].isna()]
    filtered = filtered[~filtered['death'].isna()] 
    filtered.death[filtered.death != '0'] = 1
    filtered.death[filtered.death == '0'] = 0
    return filtered

def plot_COVID_age(age_data):
    """
    This functions plot the age info of COVID
    plot one hisotagram and two box plot 
    """
    fig, [ax0, ax1, ax2] = plt.subplots(1, figsize=(13,5), ncols=3)
    sns.boxplot(ax=ax1, y=age_data["age"])
    ax1.set_title('Age of Infectious people')
    sns.boxplot(ax=ax2, y=age_data['age'], x=age_data['death'])
    ax2.set_title('Age of Infectious people(Alive/Death)')
    sns.distplot(ax=ax0, a=age_data['age'], bins=10)
    ax0.set_title('Age Distribution of Infectious people')
    fig.savefig('COVID-10_Age')
    

def plot_global_progress_COVID():
    daily_data = pd.read_csv('covid_19_data.csv')
    daily_data = clean_df(daily_data)
    daily_data['Last Update'] = pd.to_datetime(daily_data['Last Update'])
    dates = daily_data['Last Update'].dt.floor('D')
    daily_data = daily_data.groupby(dates)['Confirmed'].sum().reset_index()
    fig, ax = plt.subplots(1, figsize=(12, 7))
    sns.lineplot(x='Last Update', y='Confirmed', data=daily_data, ax=ax)
    plt.xticks(rotation=-45)
    plt.title('Global Infection Progress COVID-19')
    plt.savefig('Global Infection Progress COVID-19')


"""
SARS
"""
def load_in_SARS_cumulative():
    """
    This function load in the SARS data 
    change countries name to match further data
    """
    SARS_cumulative_data = pd.read_csv('2003_07_11.csv')
    SARS_cumulative_data = SARS_cumulative_data.replace('China^5', 'China')
    SARS_cumulative_data = SARS_cumulative_data.replace('Canada^4', 'Canada')
    SARS_cumulative_data = SARS_cumulative_data.replace('China, Hong Kong Special Administrative Region^6', 'Hong Kong')
    SARS_cumulative_data = SARS_cumulative_data.replace('China, Macao Special Administrative Region', 'Macao')
    SARS_cumulative_data = SARS_cumulative_data.replace('China, Taiwan ', 'Taiwan')
    SARS_cumulative_data = SARS_cumulative_data.replace('United States^7', 'United States of America')
    SARS_cumulative_data = SARS_cumulative_data.replace('Viet Nam', 'Vietnam')
    return SARS_cumulative_data


def merge_SARS(global_map, SARS_cumulative_data):
    """
    This function merge the SARS data with global geometry
    """
    SARS_merged = global_map.merge(SARS_cumulative_data, left_on='NAME', right_on='Country', how='left')
    return SARS_merged


def plot_SARS_global(global_map, SARS_merged):
    """
    This function plots the global situation of SARS
    """
    fig, ax = plt.subplots(1, figsize=(15, 7))
    global_map.plot(ax=ax, figsize=(15, 7), color='#EEEEEE')
    SARS_merged.plot(ax=ax, column='Cumulative number of case(s)^2', figsize=(15, 7), legend=True, vmin=0, vmax=2500,
                 cmap = sns.cm.rocket_r)
    plt.title('Infection Number of SARS World Wide')
    fig.savefig('SARS_global')

def plot_global_progress_SARS():
    SARS_daily_data = pd.read_csv('sars_final1.csv')
    fig, ax = plt.subplots(1, figsize=(12, 7))
    ax1 = sns.lineplot(x='Date', y='Infected', data=SARS_daily_data, ax=ax)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.xticks(rotation=-45)
    plt.title('Global Infection Progress SARS')
    plt.savefig('Global Infection Progress SARS')


def main():
    global_map = load_in_global_data()
    china_map = load_in_China_data()

    # Question 1
    COVID_data = load_in_COVID_data()
    COVID_merged = merge_COVID_data(COVID_data, global_map)
    plot_COVID_global(global_map, COVID_merged)
    
    SARS_cumulative_data = load_in_SARS_cumulative()
    SARS_merged = merge_SARS(global_map, SARS_cumulative_data)
    plot_SARS_global(global_map, SARS_merged)
    
    # Question 2
    plot_global_progress_SARS()
    plot_global_progress_COVID()

    countries = select_top10_COVID_progress(COVID_merged)
    plot_top10_COVID_progress(countries)
    
    # Question 3
    china_COVID_merged = merge_China_data(china_map, COVID_data)
    plot_China_COVID(china_map, china_COVID_merged)
    
    total = load_clean_china_COVID_ratio()
    plot_COVID_china_ratio(total)
    
    # Question 4
    top10_gender = load_clean_COVID_gender()
    plot_COVID_gender(top10_gender)
    
    # Question 5
    age_data = load_clean_COVID_age()
    plot_COVID_age(age_data)

if __name__ == '__main__':
    main()