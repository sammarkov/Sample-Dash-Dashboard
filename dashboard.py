# -*- coding: utf-8 -*-
"""
Author: Samantha J Corrado

Description: Sample DOE results exploration/decision making dashboard sample
"""

import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_html_components as html
import numpy as np
import dashboard_functions as dbf


# Set up app
css_loc = ['/static/stylesheet_cosmo.css', '/static/stylesheet_loading.css']
app = dash.Dash(__name__, external_stylesheets=css_loc)
app.config['suppress_callback_exceptions']=True
app.title = 'Sample Dashboard'

# App layout
app.layout = html.Div([dbc.Tabs([dbc.Tab(dbf.welcome(), label='Welcome', tab_id='welcome'),
                                 dbc.Tab(dbf.dse(), label='Design Space Exploration', tab_id='dse'),
                                 dbc.Tab(dbf.pp(), label='Prediction Profiler', tab_id='pp'),
                                 dbc.Tab(dbf.fa(), label='Feasibility Analysis', tab_id='fa'),
                                 dbc.Tab(dbf.roa(), label='Ranking of Alternatives', tab_id='roa')], 
                                id='tabs', active_tab='welcome')])


### Callback Functions ###
## Design Space Exploration ##
@app.callback([Output('dse-dv_1-slider-label', 'children')],
              [Input('dse-dv_1-slider', 'value')])
def update_dse_dv_1_slider_label(slider_value):
    label = '{} to {} selected'.format(slider_value[0], slider_value[1])
    return [label]

@app.callback([Output('dse-dv_2-slider-label', 'children')],
              [Input('dse-dv_2-slider', 'value')])
def update_dse_dv_2_slider_label(slider_value):
    label = '{} to {} selected'.format(slider_value[0], slider_value[1])
    return [label]

@app.callback([Output('dse-dv_3-slider-label', 'children')],
              [Input('dse-dv_3-slider', 'value')])
def update_dse_dv_3_slider_label(slider_value):
    label = '{} to {} selected'.format(slider_value[0], slider_value[1])
    return [label]

@app.callback([Output('dse-dv_4-slider-label', 'children')],
              [Input('dse-dv_4-slider', 'value')])
def update_dse_dv_4_slider_label(slider_value):
    label = '{} to {} selected'.format(slider_value[0], slider_value[1])
    return [label]

@app.callback([Output('dse-dv_5-slider-label', 'children')],
              [Input('dse-dv_5-slider', 'value')])
def update_dse_dv_5_slider_label(slider_value):
    label = '{} to {} selected'.format(slider_value[0], slider_value[1])
    return [label]

@app.callback([Output('dse-dv_6-slider-label', 'children')],
              [Input('dse-dv_6-slider', 'value')])
def update_dse_dv_6_slider_label(slider_value):
    label = '{} to {} selected'.format(slider_value[0], slider_value[1])
    return [label]

@app.callback([Output('dse-dv_7-slider-label', 'children')],
              [Input('dse-dv_7-slider', 'value')])
def update_dse_dv_7_slider_label(slider_value):
    label = '{} to {} selected'.format(slider_value[0], slider_value[1])
    return [label]

@app.callback([Output('dse-dv_8-slider-label', 'children')],
              [Input('dse-dv_8-slider', 'value')])
def update_dse_dv_8_slider_label(slider_value):
    label = '{} to {} selected'.format(slider_value[0], slider_value[1])
    return [label]

@app.callback([Output('dse-out_1-slider-label', 'children')],
              [Input('dse-out_1-slider', 'value')])
def update_dse_out_1_slider_label(slider_value):
    label = '{} to {} selected'.format(slider_value[0], slider_value[1])
    return [label]

@app.callback([Output('dse-out_2-slider-label', 'children')],
              [Input('dse-out_2-slider', 'value')])
def update_dse_out_2_slider_label(slider_value):
    label = '{} to {} selected'.format(slider_value[0], slider_value[1])
    return [label]

@app.callback([Output('dse-scatter', 'figure')],
              [Input('dse-dv_1-slider', 'value'),
               Input('dse-dv_2-slider', 'value'),
               Input('dse-dv_3-slider', 'value'),
               Input('dse-dv_4-slider', 'value'),
               Input('dse-dv_5-slider', 'value'),
               Input('dse-dv_6-slider', 'value'),
               Input('dse-dv_7-slider', 'value'),
               Input('dse-dv_8-slider', 'value'),
               Input('dse-out_1-slider', 'value'),
               Input('dse-out_2-slider', 'value'),
               Input('dse-dv_1-check', 'value'),
               Input('dse-dv_2-check', 'value'),
               Input('dse-dv_3-check', 'value'),
               Input('dse-dv_4-check', 'value'),
               Input('dse-dv_5-check', 'value'),
               Input('dse-dv_6-check', 'value'),
               Input('dse-dv_7-check', 'value'),
               Input('dse-dv_8-check', 'value'),
               Input('dse-out_1-check', 'value'),
               Input('dse-out_2-check', 'value')])
def update_dse_scatter(dv_1_slider, dv_2_slider, dv_3_slider, dv_4_slider, dv_5_slider,
                       dv_6_slider, dv_7_slider, dv_8_slider,
                       out_1_slider, out_2_slider,
                       dv_1_check, dv_2_check, dv_3_check, dv_4_check, dv_5_check,
                       dv_6_check, dv_7_check, dv_8_check,
                       out_1_check, out_2_check):
    check_dict = {'dv_1': dv_1_check,
                  'dv_2': dv_2_check,
                  'dv_3': dv_3_check,
                  'dv_4': dv_4_check, 
                  'dv_5': dv_5_check,
                  'dv_6': dv_6_check,
                  'dv_7': dv_7_check,
                  'dv_8': dv_8_check,
                  'out_1': out_1_check,
                  'out_2': out_2_check}
    filter_dict = {'dv_1': dv_1_slider,
                   'dv_2': dv_2_slider,
                   'dv_3': dv_3_slider,
                   'dv_4': dv_4_slider, 
                   'dv_5': dv_5_slider,
                   'dv_6': dv_6_slider,
                   'dv_7': dv_7_slider,
                   'dv_8': dv_8_slider,
                   'out_1': out_1_slider,
                   'out_2': out_2_slider}
    dimensions = []
    for key in dbf.metric_dict:
        if check_dict[key] == [1]:
            dimensions.append(key)
    return [dbf.dse_scatterplot(filter_dict, dimensions)]

## Feasibility Analysis ##
@app.callback([Output('fa-out_1-cdf', 'figure')],
              [Input('fa-out_1-constraint', 'value')])
def update_fa_out_1_cdf(constraint):
    return [dbf.fa_cdf('out_1', constraint)]

@app.callback([Output('fa-out_2-cdf', 'figure')],
              [Input('fa-out_2-constraint', 'value')])
def update_fa_out_2_cdf(constraint):
    return [dbf.fa_cdf('out_2', constraint)]

## Ranking of Alternatives ##
@app.callback([Output('roa-bar-chart', 'figure')],
              [Input('roa-rank', 'n_clicks'),
               Input('roa-n_top', 'value')],
              [State('roa-out_1-slider', 'value'),
               State('roa-out_2-slider', 'value')])
def update_roa_bar_chart(n_clicks,
                         n_top,
                         out_1, out_2):
    weight_dict = {'out_1': out_1,
                   'out_2': out_2}
    top_designs = dbf.roa_TOPSIS(weight_dict, n_top)
    return [dbf.roa_bar_chart(top_designs)]

## Prediction Profiler ##
@app.callback([Output('pp-dv_1-slider-label', 'children')],
              [Input('pp-dv_1-slider', 'value')])
def update_pp_dv_1_slider_label(slider_value):
    label = '{} selected'.format(slider_value)
    return [label]

@app.callback([Output('pp-dv_2-slider-label', 'children')],
              [Input('pp-dv_2-slider', 'value')])
def update_pp_dv_2_slider_label(slider_value):
    label = '{} selected'.format(slider_value)
    return [label]

@app.callback([Output('pp-dv_3-slider-label', 'children')],
              [Input('pp-dv_3-slider', 'value')])
def update_pp_dv_3_slider_label(slider_value):
    label = '{} selected'.format(slider_value)
    return [label]

@app.callback([Output('pp-dv_4-slider-label', 'children')],
              [Input('pp-dv_4-slider', 'value')])
def update_pp_dv_4_slider_label(slider_value):
    label = '{} selected'.format(slider_value)
    return [label]

@app.callback([Output('pp-dv_5-slider-label', 'children')],
              [Input('pp-dv_5-slider', 'value')])
def update_pp_dv_5_slider_label(slider_value):
    label = '{} selected'.format(slider_value)
    return [label]

@app.callback([Output('pp-dv_6-slider-label', 'children')],
              [Input('pp-dv_6-slider', 'value')])
def update_pp_dv_6_slider_label(slider_value):
    label = '{} selected'.format(slider_value)
    return [label]

@app.callback([Output('pp-dv_7-slider-label', 'children')],
              [Input('pp-dv_7-slider', 'value')])
def update_pp_dv_7_slider_label(slider_value):
    label = '{} selected'.format(slider_value)
    return [label]

@app.callback([Output('pp-dv_8-slider-label', 'children')],
              [Input('pp-dv_8-slider', 'value')])
def update_pp_dv_8_slider_label(slider_value):
    label = '{} selected'.format(slider_value)
    return [label]

@app.callback([Output('pp-profiler', 'figure')],
              [Input('pp-dv_1-slider', 'value'),
               Input('pp-dv_2-slider', 'value'),
               Input('pp-dv_3-slider', 'value'),
               Input('pp-dv_4-slider', 'value'),
               Input('pp-dv_5-slider', 'value'),
               Input('pp-dv_6-slider', 'value'),
               Input('pp-dv_7-slider', 'value'),
               Input('pp-dv_8-slider', 'value')])
def update_pp_profilter(dv_1_slider, dv_2_slider, dv_3_slider, dv_4_slider, dv_5_slider,
                        dv_6_slider, dv_7_slider, dv_8_slider):
    inputs = {'dv_1': dv_1_slider,
              'dv_2': dv_2_slider,
              'dv_3': dv_3_slider,
              'dv_4': dv_4_slider, 
              'dv_5': dv_5_slider,
              'dv_6': dv_6_slider,
              'dv_7': dv_7_slider,
              'dv_8': dv_8_slider}
    return [dbf.pp_profiler(inputs)]


# Run app 
if __name__ == '__main__':
    app.run_server(debug=False)