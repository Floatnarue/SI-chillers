import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
import dash
import dash_daq as daq
from dash.dependencies import Input, Output, State
from dash import Dash, dcc, html, Input, Output  # pip install dash (version 2.0.0 or higher)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error


# ------------------------------------------------------------------------------
# Import all trained model 
mymodel = tf.saved_model.load('/Users/floatnarue/Desktop/Get to acheivement stuff/DL:NLP stuff/Altotech')
myLinear = joblib.load('/Users/floatnarue/Desktop/Get to acheivement stuff/DL:NLP stuff/Altotech/linearmodel/linear_regression_model.pkl')
# ------------------------------------------------------------------------------

app = Dash(__name__)
app.title= "System identification of chiller plants model evaluation dashboard"

# ------------------------------------------------------------------------------

## Read the csv data ##
chdata = pd.read_csv('/Users/floatnarue/Desktop/Get to acheivement stuff/DL:NLP stuff/Altotech/chiller_plant_data.csv',index_col= False)
cinfo = pd.read_csv('/Users/floatnarue/Desktop/Get to acheivement stuff/DL:NLP stuff/Altotech/columns_info.csv')


chdata['datetime'] = pd.to_datetime(chdata['datetime'],format='%Y-%m-%d %H:%M:%S')
cor = chdata.corr("pearson")
cor_target = abs(cor["plant_efficiency"])
rel_feature = cor_target[cor_target>=0.3]
column_name = []
for col in chdata.columns:
    if col in rel_feature :
        column_name.append(col)
column_name.remove("plant_efficiency")


min_date = datetime(2020, 8, 13, 0, 3, 0)
max_date = datetime(2020, 12, 31, 23, 55, 0)


# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    html.H1("SI model evaluation", style={'text-align': 'center'}),


    html.Div([
        html.Label("Initial Date:"),
        dcc.DatePickerSingle(
            id='initial-date-picker',
            display_format='YYYY-MM-DD',  # Date format
            initial_visible_month=min_date,  # Initial visible month
            date=min_date,  # Default date
            min_date_allowed=min_date,  # Set minimum allowed date
            max_date_allowed=max_date  # Set maximum allowed date
        ),
    ]),
    
    html.Div([
        html.Label("Final Date:"),
        dcc.DatePickerSingle(
            id='final-date-picker',
            display_format='YYYY-MM-DD',  # Date format
            initial_visible_month=max_date,  # Initial visible month
            date=max_date,  # Default date
            min_date_allowed=min_date,  # Set minimum allowed date
            max_date_allowed=max_date  # Set maximum allowed date
        ),
    ]),

    dcc.Dropdown(
        id='hour-dropdown',
        options=[{'label': f'{hour:02}', 'value': f'{hour:02}'} for hour in range(24)],
        value='00',  # Default hour
    ),
    
    dcc.Dropdown(
        id='minute-dropdown',
        options=[{'label': f'{minute:02}', 'value': f'{minute:02}'} for minute in range(60)],
        value='00',  # Default minute
    ),

    dcc.Dropdown(
        id='second-dropdown',
        options=[{'label': f'{second:02}', 'value': f'{second:02}'} for second in range(60)],
        value='00',  # Default second
    ),


    # Dropdown for selecting a machine learning model => Compare the model to others algorithm
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'Linear regression', 'value': 'linearregression'},
            {'label': 'MLP', 'value': 'MLP'},
        ],
        value='MLP',  # Default model
    ), 
    

    html.Button('Evaluate Model', id='evaluate-button', n_clicks=0),
    
    # Display model accuracy
    html.Div(id='rmse-output'),
    # Visualize y_predict and y_actual in time period
    dcc.Graph(id='actual-plot', config={'displayModeBar': False}),
    
    dcc.Graph(id='predicted-plot', config={'displayModeBar': False}),
    
    
])

# Define callback to update model accuracy and scatter plot
@app.callback(
    [Output('rmse-output', 'children'),
     Output('actual-plot', 'figure'),
     Output('predicted-plot', 'figure')],
    [Input('evaluate-button', 'n_clicks')],
    [State('initial-date-picker', 'date'),
     State('final-date-picker', 'date'),
     State('hour-dropdown', 'value'),
     State('minute-dropdown', 'value'),
     State('second-dropdown', 'value'),
     State('model-dropdown', 'value')],
     prevent_initial_call=True
)
def perform_model_evaluation(n_clicks, initial_date, final_date, selected_hour, selected_minute, selected_second, selected_model):
    if not n_clicks :
        return None,go.Figure(),go.Figure # Evaluation hasn't started yet
    #--------------------------------------------------------------------------------------------------------
    initial_datetime = f"{initial_date} {selected_hour}:{selected_minute}:{selected_second}"
    final_datetime = f"{final_date} {selected_hour}:{selected_minute}:{selected_second}"

    # Convert to datetime objects
    initial_datetime = datetime.strptime(initial_datetime, '%Y-%m-%d %H:%M:%S')
    final_datetime = datetime.strptime(final_datetime, '%Y-%m-%d %H:%M:%S')
    
    # Filter data based on selected time period
    filtered_data = chdata[(chdata['datetime'] >= initial_datetime) & (chdata['datetime'] <= final_datetime)]
    
    time = filtered_data['datetime'].copy()
    y_actual = filtered_data['plant_efficiency'].values

    filtered_data.drop(['datetime'], axis=1)


    X = filtered_data[column_name]
    ## To keep the y_pred empty while the button still not clicked.
    y_pred = None 
    
                                                        
    
    # Select the machine learning model based on user's choice
    if selected_model == 'linearregression':
        y_pred = myLinear.predict(X)
    
            
    elif selected_model == 'MLP':
        y_pred = mymodel(X)
        
    

    if y_pred is not None:
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

        ## Check if data type if the same
        if type(y_pred) != type(y_actual):
            y_pred = pd.Series(y_pred.numpy().flatten())
        actual_plot = go.Figure(data=[
            go.Scatter(x=time, y=y_actual, mode='lines+markers', name='Actual')
        ])
        actual_plot.update_layout(title='Actual Values')
        
        predicted_plot = go.Figure(data=[
            go.Scatter(x=time, y=y_pred, mode='lines+markers', name='Predicted', line=dict(color='red'))
        ])
        predicted_plot.update_layout(title='Predicted Values')
        return f"Root mean square error (RMSE): {rmse:.2f}", actual_plot, predicted_plot
    return None, go.Figure(), go.Figure()
    

if __name__ == '__main__':
    app.run_server(debug=True)