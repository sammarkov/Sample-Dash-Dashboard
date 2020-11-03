# -*- coding: utf-8 -*-
"""
Author: Samantha J Corrado

Description: Surrogate model generation.
"""


import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sub
import dash
import dash_core_components as dcc
import dash_html_components as html
import colorlover as cl
import pickle
from sklearn import linear_model, metrics, model_selection

# Load data
data = pd.read_csv('dashboard_data.csv', header=0)
# Set inputs and outputs
outputs = ['out_1', 'out_2']
inputs = ['dv_1', 'dv_2', 'dv_3', 'dv_4', 'dv_5', 'dv_6', 'dv_7', 'dv_8']

# Folder to save models and print model fit figures and toggle printing/saving on or off
to_print=True
save_models=True
models_folder = 'models'
model_fit_plots_folder = 'model_fit_plots'
if save_models:
    if not os.path.exists(models_folder):
        os.mkdir(models_folder) 
if to_print:
    if not os.path.exists(model_fit_plots_folder):
        os.mkdir(model_fit_plots_folder)

# Colors
colors = cl.scales['12']['qual']['Paired']


def fit(X, y, save_models=False, save_models_path=None):
    # Split data into training and testing
    X_tr, X_test, y_tr, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=1) 
    # Create list of models to potentially fit (only use linear regression now)
    models = [linear_model.LinearRegression()]
    model_types = ['Linear Regression']
    model_labels = ['LR']
    n = len(models)
    a_v_p_title = ['Actual vs. Predicted' for i in range(n)]
    r_v_p_title = ['Residual vs. Predicted' for i in range(n)]
    titles = a_v_p_title + r_v_p_title
    scatter = {'type': 'scatter'}
    table = {'type': 'table'}
    fig = sub.make_subplots(rows=3, cols=n,
                            specs=[[scatter]*n,
                                   [scatter]*n,
                                   [table]*n],
                            subplot_titles=(titles))
    for i in range(n):
        fig = fit_model(X_tr, X_test, y_tr, y_test, models[i], model_types[i], model_labels[i], output, fig, i+1, save_models)
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=18)
    fig.update_layout(title={'text': y.name, 'font': {'size':20}},
                      plot_bgcolor='rgba(0,0,0,0)',
                      margin={'t': 55, 'b': 0},
                      height=600,
                      width=2000)
    # Writing images if True
    if to_print:
        path = model_fit_plots_folder + '/' + y.name + '.jpeg'
        if os.path.exists(path):
            os.remove(path)
        fig.write_image(path)    
    return fig

def fit_model(X_tr, X_test, y_tr, y_test, model, model_type, model_label, output, fig, i, save_models):
    # Fit model
    model.fit(X_tr, y_tr)
    # Predict
    y_pred_tr = model.predict(X_tr)
    y_pred_test = model.predict(X_test)
    # Computing fit statistics
    r_2_tr = metrics.r2_score(y_tr, y_pred_tr)
    r_2_val = metrics.r2_score(y_test, y_pred_test)
    mean_squared_error_tr = metrics.mean_squared_error(y_tr, y_pred_tr)
    mean_squared_error_val = metrics.mean_squared_error(y_test, y_pred_test)
    # Creating figure to visualize model fit
    fig.add_trace(go.Scatter(x=y_pred_tr, y=y_tr,
                             name='Training',
                             mode='markers',
                             marker=dict(color=colors[0]), 
                             showlegend=True), row=1, col=i)
    fig.add_trace(go.Scatter(x=y_pred_test, y=y_test,
                             name='Testing',
                             mode='markers',
                             marker=dict(color=colors[1]), 
                             showlegend=True), row=1, col=i)
    fig.add_trace(go.Scatter(x=y_pred_tr, y=(y_tr - y_pred_tr),
                             name='Training',
                             mode='markers',
                             marker=dict(color=colors[0]),
                             showlegend=False), row=2, col=i)
    fig.add_trace(go.Scatter(x=y_pred_test, y=(y_test - y_pred_test),
                             name='Testing',
                             mode='markers',
                             marker=dict(color=colors[1]),
                             showlegend=False), row=2, col=i)
    fig.add_trace(go.Table(header=dict(values=['Fit Statistic', 'Value']),
                           cells=dict(values=[['R-Squared Training',
                                               'R-Squared Validation',
                                               'Mean Squared Error Training',
                                               'Mean Squared Error Testing'],
                                              [np.round(r_2_tr, 3),
                                               np.round(r_2_val, 3),
                                               np.round(mean_squared_error_tr, 3),
                                               np.round(mean_squared_error_val, 3)]])),
                  row=3, col=i)  
    fig.update_xaxes(title_text=model_type+' Predicted',
                     gridcolor='gray',
                     zerolinecolor='gray',
                     row=1, col=i)
    fig.update_xaxes(title_text=model_type+' Predicted',
                     gridcolor='gray',
                     zerolinecolor='gray',
                     row=2, col=i)
    fig.update_yaxes(title_text='Actual',
                     gridcolor='gray',
                     zerolinecolor='gray',
                     row=1, col=i)
    fig.update_yaxes(title_text='Residual',
                     gridcolor='gray',
                     zerolinecolor='gray',
                     row=2, col=i)
    if save_models:
        save_models_path = os.path.join(models_folder, model_label)
        if not os.path.exists(save_models_path):
            os.mkdir(save_models_path)   
        path = os.path.join(save_models_path, output + '.pickle') 
        if os.path.exists(path):
            os.remove(path)
        pickle.dump(model, open(path, 'wb'))
    return fig   

def create_dashboard_service(figs_dict):
    display = [html.Div(html.H5('Model Fits Assessment'))]
    for key in figs_dict:
        display.append(html.Div(children=dcc.Graph(figure=figs_dict[key],
                                                   style={'width': '100%', 'height': '100%'}),
        className='row'))
    return display


## Train model ##
# Split data into X and Y
X = data[inputs]
Y = data[outputs]

# Saving model parameters to check model saving functionality later
if save_models:
    X.to_json(os.path.join(models_folder, 'input_parameters.json')) 

# Loop through outputs and fit models
figs_dict = {}
for output in outputs:
    y = Y[output]
    fig = fit(X, y, save_models, to_print)
    figs_dict[output] = fig

## Create dashboard for visualization ##    
css_loc = ['/static/stylesheet_cosmo.css']
app = dash.Dash(__name__, external_stylesheets=css_loc)    
app.layout = html.Div(
    id='main',
    children=create_dashboard_service(figs_dict)
    )

if __name__ == '__main__':
    app.run_server(debug=False)
    


