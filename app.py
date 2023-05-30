import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
from collections import deque

import pandas as pd
import numpy as np
import plotly.express as px
import os #чтение файлов /мб не пригодится
from dash import (
    Dash, html,
    dash_table,
    dcc, callback,
    Input, Output
)

# ЭТО НЕ СВЯЗАНО С ДЭШ ЭТО МУТКИ С ДАТАСЕТОМ!!!
df1 = pd.read_csv('/Users/viktoria.nam/Downloads/archive/Highest_victim_count.csv')  #читаем три датасета
df2 = pd.read_csv('/Users/viktoria.nam/Downloads/archive/15_to_30_victim_count.csv')
df3 = pd.read_csv('/Users/viktoria.nam/Downloads/archive/5_to_14_victim_count.csv')

df = pd.concat([df1, df2, df3])
print(df.columns)
df.reset_index(drop=True, inplace=True)
# print(df.shape())
#чистим данные

df.rename(columns={'Years active': 'Years_active', 'Proven victims': 'Proven_victims', 'Possible victims': 'Possible_victims'}, inplace=True)
#POSSIBLE

df.Possible_victims.fillna(df.Proven_victims,inplace=True)
df.Possible_victims.value_counts(dropna=False)

df.loc[df.Possible_victims.isin(['Unknown', '-']), 'Possible_victims'] = df.Proven_victims

df = df.apply(lambda x: x.apply(str) if x.dtype == 'object' else x)
df = df.apply(lambda x: x.str.rstrip('+') if x.dtype == 'object' else x)
df = df.apply(lambda x: x.str.rstrip('?') if x.dtype == 'object' else x)
df = df.apply(lambda x: x.str.lstrip('~') if x.dtype == 'object' else x)

df['Possible_victims'] = [str(x).split('-')[-1] for x in df['Possible_victims']]
df['Possible_victims'] = [str(x).split('–')[-1] for x in df['Possible_victims']]
df.Possible_victims = pd.to_numeric(df.Possible_victims, errors='coerce')
# print(df.Possible_victims.value_counts(dropna=False))
#PROVEN
df.Proven_victims.value_counts(dropna=False)

df['Proven_victims'] = [str(x).split('–')[-1] for x in df['Proven_victims']]

print(df.Proven_victims.value_counts(dropna=False))
df.Proven_victims = pd.to_numeric(df.Proven_victims, errors='coerce')
#country

df.Country = df.Country.str.replace('\(suspected\)', "", regex = True)
df.Country = df.Country.str.replace('\(alleged\)', "", regex = True)
df.Country = df.Country.str.replace('\(claimed\)', "", regex = True)


df.Country = df.Country.str.replace('West Germany', "Germany")
df.Country = df.Country.str.replace('East Germany', "Germany")
df.Country = df.Country.str.replace('German Empire', "Germany")
df.Country = df.Country.str.replace('Allied-occupied Germany', "Germany")
df.Country = df.Country.str.replace('Portuguese Angola', "Angola")
df.Country = df.Country.str.replace('Ottoman Empire', "Turkey")
df.Country = df.Country.str.replace('Kingdom of Romania', "Romania")
df.Country = df.Country.str.split('\r\n', expand=True)[0]

# print(df.Country.value_counts())

#years

df.Years_active = df.Years_active.str.replace(' and earlier', '', regex = False)
df.Years_active = df.Years_active.str.replace(' to present', '', regex = False)
df.Years_active = df.Years_active.str.replace('30 June 1983', '1983', regex = False)
df.Years_active = df.Years_active.str.replace(' to 23 July 1983', '', regex = False)
df.Years_active = df.Years_active.str.replace('s', '', regex = False)
df.Years_active = df.Years_active.str.replace('?', '', regex = False)
df.Years_active = df.Years_active.str.replace('c.', '', regex = False)
df.Years_active = df.Years_active.str.replace('late ', '', regex = False)

df['From_Date'] = np.nan
df['To_Date'] = np.nan

df[['From_Date','To_Date']] = df['Years_active'].str.split('to',expand=True)
df.To_Date.fillna(df.From_Date,inplace=True)
# print(df.To_Date.isna().sum())
df.To_Date = pd.to_numeric(df.To_Date, errors='coerce')
df.From_Date = pd.to_numeric(df.From_Date, errors='coerce')

df['Active_Years'] = df.To_Date.values - df.From_Date.values
df.Active_Years.replace({0:1}, inplace=True)
df.Active_Years = pd.to_numeric(df.Active_Years, errors='coerce')



#DF1!!!!!

df1.reset_index(drop=True, inplace=True)
# print(df1.shape())
#чистим данные

df1.rename(columns={'Years active': 'Years_active', 'Proven victims': 'Proven_victims', 'Possible victims': 'Possible_victims'}, inplace=True)
#POSSIBLE

df1.Possible_victims.fillna(df1.Proven_victims,inplace=True)
df1.Possible_victims.value_counts(dropna=False)

df1.loc[df1.Possible_victims.isin(['Unknown', '-']), 'Possible_victims'] = df1.Proven_victims

df1 = df1.apply(lambda x: x.apply(str) if x.dtype == 'object' else x)
df1 = df1.apply(lambda x: x.str.rstrip('+') if x.dtype == 'object' else x)
df1 = df1.apply(lambda x: x.str.rstrip('?') if x.dtype == 'object' else x)
df1 = df1.apply(lambda x: x.str.lstrip('~') if x.dtype == 'object' else x)

df1['Possible_victims'] = [str(x).split('-')[-1] for x in df1['Possible_victims']]
df1['Possible_victims'] = [str(x).split('–')[-1] for x in df1['Possible_victims']]
df1.Possible_victims = pd.to_numeric(df1.Possible_victims, errors='coerce')
# print(df1.Possible_victims.value_counts(dropna=False))
#PROVEN
df1.Proven_victims.value_counts(dropna=False)

df1['Proven_victims'] = [str(x).split('–')[-1] for x in df1['Proven_victims']]

print(df1.Proven_victims.value_counts(dropna=False))
df1.Proven_victims = pd.to_numeric(df1.Proven_victims, errors='coerce')
#country

df1.Country = df1.Country.str.replace('\(suspected\)', "", regex = True)
df1.Country = df1.Country.str.replace('\(alleged\)', "", regex = True)
df1.Country = df1.Country.str.replace('\(claimed\)', "", regex = True)


df1.Country = df1.Country.str.replace('West Germany', "Germany")
df1.Country = df1.Country.str.replace('East Germany', "Germany")
df1.Country = df1.Country.str.replace('German Empire', "Germany")
df1.Country = df1.Country.str.replace('Allied-occupied Germany', "Germany")
df1.Country = df1.Country.str.replace('Portuguese Angola', "Angola")
df1.Country = df1.Country.str.replace('Ottoman Empire', "Turkey")
df1.Country = df1.Country.str.replace('Kingdom of Romania', "Romania")
df1.Country = df1.Country.str.split('\r\n', expand=True)[0]

# print(df1.Country.value_counts())

#years

df1.Years_active = df1.Years_active.str.replace(' and earlier', '', regex = False)
df1.Years_active = df1.Years_active.str.replace(' to present', '', regex = False)
df1.Years_active = df1.Years_active.str.replace('30 June 1983', '1983', regex = False)
df1.Years_active = df1.Years_active.str.replace(' to 23 July 1983', '', regex = False)
df1.Years_active = df1.Years_active.str.replace('s', '', regex = False)
df1.Years_active = df1.Years_active.str.replace('?', '', regex = False)
df1.Years_active = df1.Years_active.str.replace('c.', '', regex = False)
df1.Years_active = df1.Years_active.str.replace('late ', '', regex = False)

df1['From_Date'] = np.nan
df1['To_Date'] = np.nan

df1[['From_Date','To_Date']] = df1['Years_active'].str.split('to',expand=True)
df1.To_Date.fillna(df1.From_Date,inplace=True)
# print(df1.To_Date.isna().sum())
df1.To_Date = pd.to_numeric(df1.To_Date, errors='coerce')
df1.From_Date = pd.to_numeric(df1.From_Date, errors='coerce')

df1['Active_Years'] = df1.To_Date.values - df1.From_Date.values
df1.Active_Years.replace({0:1}, inplace=True)
df1.Active_Years = pd.to_numeric(df1.Active_Years, errors='coerce')

#DF2!!!df2.reset_index(drop=True, inplace=True)
# print(df2.shape())
#чистим данные

df2.rename(columns={'Years active': 'Years_active', 'Proven victims': 'Proven_victims', 'Possible victims': 'Possible_victims'}, inplace=True)
#POSSIBLE

df2.Possible_victims.fillna(df2.Proven_victims,inplace=True)
df2.Possible_victims.value_counts(dropna=False)

df2.loc[df2.Possible_victims.isin(['Unknown', '-']), 'Possible_victims'] = df2.Proven_victims

df2 = df2.apply(lambda x: x.apply(str) if x.dtype == 'object' else x)
df2 = df2.apply(lambda x: x.str.rstrip('+') if x.dtype == 'object' else x)
df2 = df2.apply(lambda x: x.str.rstrip('?') if x.dtype == 'object' else x)
df2 = df2.apply(lambda x: x.str.lstrip('~') if x.dtype == 'object' else x)

df2['Possible_victims'] = [str(x).split('-')[-1] for x in df2['Possible_victims']]
df2['Possible_victims'] = [str(x).split('–')[-1] for x in df2['Possible_victims']]
df2.Possible_victims = pd.to_numeric(df2.Possible_victims, errors='coerce')
# print(df2.Possible_victims.value_counts(dropna=False))
#PROVEN
df2.Proven_victims.value_counts(dropna=False)

df2['Proven_victims'] = [str(x).split('–')[-1] for x in df2['Proven_victims']]

print(df2.Proven_victims.value_counts(dropna=False))
df2.Proven_victims = pd.to_numeric(df2.Proven_victims, errors='coerce')
#country

df2.Country = df2.Country.str.replace('\(suspected\)', "", regex = True)
df2.Country = df2.Country.str.replace('\(alleged\)', "", regex = True)
df2.Country = df2.Country.str.replace('\(claimed\)', "", regex = True)


df2.Country = df2.Country.str.replace('West Germany', "Germany")
df2.Country = df2.Country.str.replace('East Germany', "Germany")
df2.Country = df2.Country.str.replace('German Empire', "Germany")
df2.Country = df2.Country.str.replace('Allied-occupied Germany', "Germany")
df2.Country = df2.Country.str.replace('Portuguese Angola', "Angola")
df2.Country = df2.Country.str.replace('Ottoman Empire', "Turkey")
df2.Country = df2.Country.str.replace('Kingdom of Romania', "Romania")
df2.Country = df2.Country.str.split('\r\n', expand=True)[0]

# print(df2.Country.value_counts())

#years

df2.Years_active = df2.Years_active.str.replace(' and earlier', '', regex = False)
df2.Years_active = df2.Years_active.str.replace(' to present', '', regex = False)
df2.Years_active = df2.Years_active.str.replace('30 June 1983', '1983', regex = False)
df2.Years_active = df2.Years_active.str.replace(' to 23 July 1983', '', regex = False)
df2.Years_active = df2.Years_active.str.replace('s', '', regex = False)
df2.Years_active = df2.Years_active.str.replace('?', '', regex = False)
df2.Years_active = df2.Years_active.str.replace('c.', '', regex = False)
df2.Years_active = df2.Years_active.str.replace('late ', '', regex = False)

df2['From_Date'] = np.nan
df2['To_Date'] = np.nan

df2[['From_Date','To_Date']] = df2['Years_active'].str.split('to',expand=True)
df2.To_Date.fillna(df2.From_Date,inplace=True)
# print(df2.To_Date.isna().sum())
df2.To_Date = pd.to_numeric(df2.To_Date, errors='coerce')
df2.From_Date = pd.to_numeric(df2.From_Date, errors='coerce')

df2['Active_Years'] = df2.To_Date.values - df2.From_Date.values
df2.Active_Years.replace({0:1}, inplace=True)
df2.Active_Years = pd.to_numeric(df2.Active_Years, errors='coerce')


#DF3!!!

df3.reset_index(drop=True, inplace=True)
# print(df3.shape())
#чистим данные

df3.rename(columns={'Years active': 'Years_active', 'Proven victims': 'Proven_victims', 'Possible victims': 'Possible_victims'}, inplace=True)
#POSSIBLE

df3.Possible_victims.fillna(df3.Proven_victims,inplace=True)
df3.Possible_victims.value_counts(dropna=False)

df3.loc[df3.Possible_victims.isin(['Unknown', '-']), 'Possible_victims'] = df3.Proven_victims

df3 = df3.apply(lambda x: x.apply(str) if x.dtype == 'object' else x)
df3 = df3.apply(lambda x: x.str.rstrip('+') if x.dtype == 'object' else x)
df3 = df3.apply(lambda x: x.str.rstrip('?') if x.dtype == 'object' else x)
df3 = df3.apply(lambda x: x.str.lstrip('~') if x.dtype == 'object' else x)

df3['Possible_victims'] = [str(x).split('-')[-1] for x in df3['Possible_victims']]
df3['Possible_victims'] = [str(x).split('–')[-1] for x in df3['Possible_victims']]
df3.Possible_victims = pd.to_numeric(df3.Possible_victims, errors='coerce')
# print(df3.Possible_victims.value_counts(dropna=False))
#PROVEN
df3.Proven_victims.value_counts(dropna=False)

df3['Proven_victims'] = [str(x).split('–')[-1] for x in df3['Proven_victims']]

print(df3.Proven_victims.value_counts(dropna=False))
df3.Proven_victims = pd.to_numeric(df3.Proven_victims, errors='coerce')
#country

df3.Country = df3.Country.str.replace('\(suspected\)', "", regex = True)
df3.Country = df3.Country.str.replace('\(alleged\)', "", regex = True)
df3.Country = df3.Country.str.replace('\(claimed\)', "", regex = True)


df3.Country = df3.Country.str.replace('West Germany', "Germany")
df3.Country = df3.Country.str.replace('East Germany', "Germany")
df3.Country = df3.Country.str.replace('German Empire', "Germany")
df3.Country = df3.Country.str.replace('Allied-occupied Germany', "Germany")
df3.Country = df3.Country.str.replace('Portuguese Angola', "Angola")
df3.Country = df3.Country.str.replace('Ottoman Empire', "Turkey")
df3.Country = df3.Country.str.replace('Kingdom of Romania', "Romania")
df3.Country = df3.Country.str.split('\r\n', expand=True)[0]

# print(df3.Country.value_counts())

#years

df3.Years_active = df3.Years_active.str.replace(' and earlier', '', regex = False)
df3.Years_active = df3.Years_active.str.replace(' to present', '', regex = False)
df3.Years_active = df3.Years_active.str.replace('30 June 1983', '1983', regex = False)
df3.Years_active = df3.Years_active.str.replace(' to 23 July 1983', '', regex = False)
df3.Years_active = df3.Years_active.str.replace('s', '', regex = False)
df3.Years_active = df3.Years_active.str.replace('?', '', regex = False)
df3.Years_active = df3.Years_active.str.replace('c.', '', regex = False)
df3.Years_active = df3.Years_active.str.replace('late ', '', regex = False)

df3['From_Date'] = np.nan
df3['To_Date'] = np.nan

df3[['From_Date','To_Date']] = df3['Years_active'].str.split('to',expand=True)
df3.To_Date.fillna(df3.From_Date,inplace=True)
# print(df3.To_Date.isna().sum())
df3.To_Date = pd.to_numeric(df3.To_Date, errors='coerce')
df3.From_Date = pd.to_numeric(df3.From_Date, errors='coerce')

df3['Active_Years'] = df3.To_Date.values - df3.From_Date.values
df3.Active_Years.replace({0:1}, inplace=True)
df3.Active_Years = pd.to_numeric(df3.Active_Years, errors='coerce')
# КОНЕЦ ИЗДЕВАТЕЛЬСТВ НАД ДАТАСЕТОМ

# df.info() #тут можно вывести инфу о датасете и типах данных

# print(df.columns) #названия колоночек




app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.H1(children="Serial Killers by number of victims"),
        html.H2(children = "Select the range of the number of victims"),
        html.Hr(),
        dcc.RadioItems(options=['5-14', '15-30', '30+', 'all'], style = {'color': '5e5557b8', 'font-family': 'sans-serif', 'font-weight': '200'}, value='15-30', id='input_data'),
        html.H2(children = "Select a priority for sorting data in a table"),
        dcc.Dropdown(['Proven_victims', 'From_Date', 'Name'], 'Proven_victims',style = {'color': '5e5557b8', 'font-family': 'sans-serif', 'font-weight': '200'}, multi=True, id = 'input_sorting'),
        dash_table.DataTable(style_data={
        'whiteSpace': 'normal',
        'height': 'auto',
        'backgroundColor': 'rgba(99, 92, 93, 0.25)'
         }, style_header={
        'backgroundColor': 'rgba(99, 92, 93, 0.25)'
        },
        id = 'output_data', page_size=5, ),
        html.H2(children = "Number of serrial killers victims by date in a whole world"),
        dcc.Graph(figure= {}, id = 'scatter_output'),
        # dcc.Graph(figure = px.line(df, x = "From_Date", y = "Proven_victims")),
        dcc.Graph(figure = {}, id = 'bar_output'),
        html.H2(children = "Select which dependency you want to display on the tree map "),
        dcc.RadioItems(options=['Difference', 'Ratio'],
                         value='difference',style = {'color': '5e5557b8', 'font-family': 'sans-serif', 'font-weight': '200'},
                         id='tm_input_id'),
        dcc.Graph(figure = {}, id = 'tm_output_id'),
        html.H2(children = "Select which sum you want to display on the histogram"),
        dcc.RadioItems(options=['Proven_victims', 'Possible_victims'],
                       value='Proven_victims',style = {'color': '5e5557b8', 'font-family': 'sans-serif', 'font-weight': '200'},
                       id='hist_input_id'),
        dcc.Graph(figure = {}, id = 'hist_output_id'),
    ]
)

@callback(
    Output(component_id='output_data', component_property='data'),
    Input(component_id='input_data', component_property='value'),
    Input(component_id = 'input_sorting', component_property='value')
)

def update_data(data_chosen, sort_chosen):
    if data_chosen == '5-14':
        data = df3
    if data_chosen == '15-30':
        data = df2
    if data_chosen == '30+':
        data = df1
    if data_chosen == 'all':
        data = df
    data = data.sort_values(by = sort_chosen).to_dict('records')
    return data

@callback(
    Output(component_id='scatter_output', component_property='figure'),
    Input(component_id='input_data', component_property='value')
)
def update_scatter(data_chosen):
    if data_chosen == '5-14':
        data = df3
    if data_chosen == '15-30':
        data = df2
    if data_chosen == '30+':
        data = df1
    if data_chosen == 'all':
        data = df
    data = data.sort_values(by = 'From_Date') #!!!! сортировка не хочет работать :(
    fig = px.scatter(data, x = 'From_Date', y = 'Proven_victims',  color = 'Country', size = 'Possible_victims',)
    fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    return fig

@callback(
    Output(component_id='bar_output', component_property='figure'),
    Input(component_id='input_data', component_property='value')
)
def update_bar(data_chosen):
    if data_chosen == '5-14':
        data = df3
    if data_chosen == '15-30':
        data = df2
    if data_chosen == '30+':
        data = df1
    if data_chosen == 'all':
        data = df

    fig = px.bar(
            data, x="Country", y="Proven_victims", color = 'Country',
            animation_frame= 'From_Date',
            range_y=[0, 168])
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    return fig

@callback(
    Output(component_id ='tm_output_id', component_property= 'figure'),
    Input(component_id='tm_input_id', component_property='value'),
    Input(component_id = 'input_data', component_property = 'value')
)
def update_graph(choosen_value, data_chosen):
    if data_chosen == '5-14':
        data = df3
    if data_chosen == '15-30':
        data = df2
    if data_chosen == '30+':
        data = df1
    if data_chosen == 'all':
        data = df

    if choosen_value == 'difference':
        fig = px.treemap(data, path=[px.Constant(data['Proven_victims'].sum()), 'Country'], values='Proven_victims',
                         color=data['Possible_victims'] - data['Proven_victims'])
    else:
        fig = px.treemap(data, path=[px.Constant(data['Proven_victims'].sum()), 'Country'], values='Proven_victims',
                         color=data['Possible_victims']/data['Proven_victims'])
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    return fig

@callback(
    Output(component_id ='hist_output_id', component_property= 'figure'),
    Input(component_id='hist_input_id', component_property='value'),
    Input(component_id = 'input_data', component_property = 'value')
)

def update_graph(choosen_victims, data_chosen):
    if data_chosen == '5-14':
        data = df3
    if data_chosen == '15-30':
        data = df2
    if data_chosen == '30+':
        data = df1
    if data_chosen == 'all':
        data = df

    if choosen_victims == 'Proven_victims':
        fig = px.histogram(data, x="From_Date", y="Proven_victims", color="Country", hover_data=df.columns)
    else:
        fig = px.histogram(data, x="From_Date", y="Possible_victims", color="Country", hover_data=df.columns)
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)