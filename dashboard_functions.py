# -*- coding: utf-8 -*-
"""
Author: Samantha J Corrado

Description: Dashboard functions.
"""

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import colorlover as cl
import numpy as np
import pandas as pd
from skcriteria import Data, MIN, MAX
from skcriteria.madm import closeness
import base64
from SurrogateModels import SurrogateModels

# Load data
data = pd.read_csv('dashboard_data.csv', header=0)

# Load images
logo_asdl_filename = 'static/images/ASDL.png'
logo_gt_filename = 'static/images/GT.png'
logo_asdl = base64.b64encode(open(logo_asdl_filename, 'rb').read())
logo_gt = base64.b64encode(open(logo_gt_filename, 'rb').read())

# Loading surrogate models
SM = SurrogateModels()

# Dictionaries mapping column names to variable names
accept_reject_dict = {'accept': 'Accepted Design',
                      'reject': 'Rejected Design'}
input_dict = {'dv_1': 'Design Variable 1',
              'dv_2': 'Design Variable 2',
              'dv_3': 'Design Variable 3',
              'dv_4': 'Design Variable 4',
              'dv_5': 'Design Variable 5',
              'dv_6': 'Design Variable 6',
              'dv_7': 'Design Variable 7',
              'dv_8': 'Design Variable 8'}
input_plot_dict = {'dv_1': 'DV1',
                   'dv_2': 'DV2',
                   'dv_3': 'DV3',
                   'dv_4': 'DV4',
                   'dv_5': 'DV5',
                   'dv_6': 'DV6',
                   'dv_7': 'DV7',
                   'dv_8': 'DV8'}
output_dict = {'out_1': 'Output 1',
               'out_2': 'Output 2'}
output_plot_dict = {'out_1': 'O1',
                    'out_2': 'O2'}
output_priority_dict = {'out_1': 'min',
                        'out_2': 'max'}
metric_dict = {**input_dict, **output_dict}

metric_plot_dict = {**input_plot_dict, **output_plot_dict}

# Colors
colors = cl.scales['9']['qual']['Set1']  
greys = cl.scales['7']['seq']['Greys']
bg_colors = [greys[0], greys[1]]

# Adding description and color to data for scatterplot matrix
data['description'] = ['Design Point {}'.format(i+1) for i in range(len(data))]

# Styling
style_font_bold = {'font-weight': 'bold'}
style_font_bold_center = {'font-weight': 'bold', 'text-align': 'center'}
style_section = {'marginTop': 4, 'marginLeft': 4,
                 'border': '1px solid'}

# Defining slider bar properties and saving in dictionaries (including mean value as well)
slider_ranges_dict = {}
slider_tick_marks_dict = {}
for metric in metric_dict:
    slider_ranges_dict[metric] = [0, 1, 0.01, 0.5]
    tick_marks_array = np.arange(0, 1, 0.2)
    slider_tick_marks_dict[metric] = [np.round(i, 1) for i in tick_marks_array]


## Welcome ##
def welcome():
    height = 200
    return [dbc.Row(className='m-1',
                    children=[html.H1('Welcome to the Sample Dashboard', style=style_font_bold)],
                    justify='center'),
    dbc.Row(className='m-1',
            children=[dbc.Col(children=[dbc.Row(children=[html.Img(src='data:image/png;base64,{}'.format(logo_gt.decode()),
                                                                   height=height)],
                                                justify='end')],
                              width=6,
                              align='end'),
                      dbc.Col(children=[dbc.Row(children=[html.Img(src='data:image/png;base64,{}'.format(logo_asdl.decode()),
                                                                   height=height)],
                                                justify='start')],
                              width=6,
                              align='center')])]

## Design Space Exploration ##
def dse_filter_section(name):
    title_width = 4
    slider_width = 5
    label_width = 3
    if name == 'input':
        loop_dict = input_dict.copy()
        label = 'Inputs:'
    else:
        loop_dict = output_dict.copy()
        label= 'Outputs:'
    display = []
    i = 1
    for key in loop_dict:
        if (i % 2) == 0:
            style = {'backgroundColor': bg_colors[0]}
        else:
            style = {}
        marks = slider_tick_marks_dict[key]
        range_and_step = slider_ranges_dict[key]
        display.append(dbc.Row(className='m-1',
                               style=style,
                               form=True,
                               children=[dbc.Col(children=[dbc.Checklist(id='dse-'+key+'-check',
                                                                         options=[{'label': loop_dict[key] + ':', 'value': 1}])],
                                                 width=title_width),
                                         dbc.Col(children=dcc.RangeSlider(id='dse-'+key+'-slider',
                                                                          min=range_and_step[0],
                                                                          max=range_and_step[1],
                                                                          step=range_and_step[2],
                                                                          allowCross=False,
                                                                          marks={int(i) if i % 1 == 0 else i: '{}'.format(i) for i in marks},
                                                                          value=[range_and_step[0], range_and_step[1]]), width=slider_width),
                                         dbc.Label(id='dse-'+key+'-slider-label', width=label_width)]))
        i = i + 1
    title = [dbc.Label(label, style=style_font_bold)]
    return title + display

def dse_filters():
    return [html.Div(style=style_section,
             children=dse_filter_section('input')),
    html.Div(style=style_section,
             children=dse_filter_section('output'))]
    
def dse_filter_dataset(filter_dict, dimensions): 
    if dimensions is not None and dimensions != []:
        df_accept = data.copy()
        df_reject = pd.DataFrame()
        for dim in dimensions:
            df_accept_tmp = df_accept.copy()
            df_accept = df_accept.loc[df_accept[dim].between(filter_dict[dim][0], filter_dict[dim][1])]
            df_reject_tmp = df_accept_tmp.loc[~df_accept_tmp[dim].between(filter_dict[dim][0], filter_dict[dim][1])]
            df_reject = pd.concat([df_reject, df_reject_tmp], axis=0)
        df_accept['color'] = ['accept' for i in df_accept['dv_1']]
        df_reject['color'] = ['reject' for i in df_reject['dv_1']]
        df = pd.concat([df_reject, df_accept], axis=0)
    else:
        df = data.copy()
        df['color'] = ['reject' for i in df['dv_1']]
    return df
                              
def dse_scatterplot_fig(df, dimensions):
    color_discrete_map = {'accept': colors[1], 
                          'reject': colors[0]}
    fig = px.scatter_matrix(df,
                            dimensions=dimensions,
                            color='color',
                            hover_name='description',
                            color_discrete_map=color_discrete_map,
                            labels={key:metric_plot_dict[key] for key in metric_plot_dict})
    fig.update_traces(showupperhalf=False, hovertemplate=None, hoverinfo='text')
    fig.update_layout(height=900, width=1200)
    fig.for_each_annotation(lambda a: a.update(text=accept_reject_dict[a.text]))
    fig.for_each_trace(lambda t: t.update(name=accept_reject_dict[t.name]))
    return fig

def dse_scatterplot(filter_dict, dimensions):
    if filter_dict is not None:
        df = dse_filter_dataset(filter_dict, dimensions)
    else:
        df = data.copy()
        df['color'] = ['reject' for i in df['dv_1']]
    if dimensions == [] or len(df[df['color'] == 'accept']) == 0:
        trace = go.Scatter(x=[0,1], y=[0,1],
                           mode='markers',
                           marker=dict(color='rgba(0,0,0,0)'),
                           name='',
                           showlegend=False)
        layout = go.Layout(plot_bgcolor=('rgba(0,0,0,0)'),
                           paper_bgcolor=('rgba(0,0,0,0)'),
                           xaxis=dict(showticklabels=False),
                           yaxis=dict(showticklabels=False),
                           margin=dict(t=0),
                           annotations=[go.layout.Annotation(text='no data matching selections available',
                                                             x=0.25, y=0.5,
                                                             showarrow=False)])
        fig = go.Figure(trace, layout)
    else:
        fig = dse_scatterplot_fig(df, dimensions)
    return fig

def dse():
    return [dbc.Row(className='m-1',
                    children=[dbc.Col(children=dse_filters(), width=4),
                              dbc.Col(children=[dcc.Graph(id='dse-scatter',
                                                          config={'displayModeBar': True,
                                                                  'modeBarButtonsToRemove': ['toImage',
                                                                                             'plotly-logomark',
                                                                                             'toggleSpikelines',
                                                                                             'sendDataToCloud',
                                                                                             'zoom2d',
                                                                                             'pan2d',
                                                                                             'lasso2d',
                                                                                             'zoomIn2d',
                                                                                             'zoomOut2d',
                                                                                             'autoScale2d',
                                                                                             'resetScale2d',
                                                                                             'hoverClosestCartesian',
                                                                                             'hoverCompareCartesian',
                                                                                             'toggleHover'],
                                                                 'displaylogo': False})], width=8)])]   

## Feasibility Analysis ## 
def fa_row_input(output):
    if output_priority_dict[output] == 'max':
        text = 'Minimum ' + output_dict[output] + ' Value:'
    else:
        text = 'Maximum ' + output_dict[output] + ' Value:' 
    return dbc.FormGroup(children=[dbc.Label(text, width=6),
                                   dbc.Col(dbc.Input(id='fa-'+output+'-constraint',
                                                     className='m-1'),
                                           width=6)], row=True)

def fa_row_stats(output):
    df = pd.DataFrame({'Statistic': ['Minimum', 'Mean', 'Maximum'],
                       'Value': [slider_ranges_dict[output][0],
                                 slider_ranges_dict[output][3],
                                 slider_ranges_dict[output][1]]})
    return [dbc.Row(className='m-1',
                    children=[dbc.Label(output_dict[output] + ' Design Space Statistics', style=style_font_bold_center),
                              dbc.Table.from_dataframe(df,
                                                       size='sm',
                                                       style={'text-align': 'center'},
                                                       bordered=True)])]
def fa_cdf_shade_trace(x, y):
    trace = go.Scatter(x=x,
                       y=y,
                       mode='lines',
                       marker=dict(color=colors[0]),
                       name='CDF',
                       showlegend=False,
                       fill='tozeroy')
    return trace
    
    
def fa_cdf(output, constraint):
    output_data = data[output]
    y, x = np.histogram(output_data, bins=50, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    dx = x[1] - x[0]
    y = (np.cumsum(y)*dx)
    df = pd.DataFrame({'x': x, 'y': y})
    title = output_dict[output] + ' CDF'
    layout = go.Layout(title=dict(text=title,
                                  x=0.5),
                       plot_bgcolor=('rgba(0,0,0,0)'),
                       paper_bgcolor=('rgba(0,0,0,0)'))
    traces = []
    traces.append(go.Scatter(x=df['x'],
                             y=df['y'],
                             marker=dict(color=colors[1]),
                             name='CDF',
                             showlegend=False))
    if constraint is not None and constraint != '':
        traces.append(go.Scatter(x=[constraint, constraint],
                                 y=[0, 1],
                                 mode='lines',
                                 line=dict(color=colors[0],
                                           width=5),
                                 name='constraint',
                                 showlegend=False))        
        constraint = float(constraint)
        if output_priority_dict[output] == 'min':
            if constraint <= slider_ranges_dict[output][1] and constraint >= slider_ranges_dict[output][0]:
                traces.append(fa_cdf_shade_trace(df[df['x'] <= constraint]['x'], df[df['x'] <= constraint]['y']))
            elif constraint >= slider_ranges_dict[output][1]:
                traces.append(fa_cdf_shade_trace(x, y))
        else:
            if constraint <= slider_ranges_dict[output][1] and constraint >= slider_ranges_dict[output][0]:
                traces.append(fa_cdf_shade_trace(df[df['x'] >= constraint]['x'], df[df['x'] >= constraint]['y']))
            elif constraint <= slider_ranges_dict[output][0]:
                traces.append(fa_cdf_shade_trace(x, y))
    fig = go.Figure(traces, layout)
    return fig

def fa_row(output):
    return [html.H4(children=output_dict[output], style=style_font_bold),
            dbc.Row(className='m-1',
                    children=[dbc.Col(children=fa_row_stats(output), width=3, align='center'),
                              dbc.Col(children=fa_row_input(output), width=3, align='center'),
                              dbc.Col(children=[dcc.Graph(id='fa-'+output+'-cdf',
                                                          style={'height': '40vh'},
                                                          config={'displayModeBar': False})],
                                      width=6, align='center')])]   
    
def fa():
    return [html.Div(style=style_section,
                     children=fa_row(key)) for key in output_dict]
    
## Ranking of Alternatives ##
def roa_weighting_input():
    title_width = 5
    slider_width = 12 - title_width
    n_top_title_width = 7
    n_top_dropdown_width = 12 - n_top_title_width
    slider_range = [0, 1]
    slider_marks = {0: '0', 0.2: '2', 0.4: '4', 0.6: '6', 0.8: '8', 1: '10'}
    n_top_max = 15
    display = []
    i = 1
    for key in output_dict:
        if (i % 2) == 0:
            style = {'backgroundColor': bg_colors[0]}
        else:
            style = {}
        display.append(dbc.Row(className='m-1',
                               style=style,
                               form=True,
                               children=[dbc.Col(children=dbc.Label(output_dict[key] + ':'), align='center', width=title_width),
                                         dbc.Col(children=dcc.Slider(id='roa-'+key+'-slider',
                                                                     min=slider_range[0],
                                                                     max=slider_range[1],
                                                                     step=0.05,
                                                                     marks=slider_marks,
                                                                     value=0.5), width=slider_width)]))
        i = i + 1
    title = [dbc.Label('Outcomes:', style=style_font_bold)]
    weighting_sliders = [html.Div(style=style_section,
                                  children=title+display)]
    n_top = [html.Div(style=style_section,
                      children=[dbc.Row(className='m-1',
                                        children=[dbc.Col(children=dbc.Label('Number of Ranked Alternatives to Display:'), width=n_top_title_width),
                                                  dbc.Col(children=dcc.Dropdown(id='roa-n_top',
                                                                                  options=[{'label': i+3, 'value': i+3} for i in range(n_top_max+3)],
                                                                                  value=3,
                                                                                  clearable=False),
                                                          width=n_top_dropdown_width)])])]
    button = [dbc.Row(className='m-1',
                      children=dbc.Button(id='roa-rank',
                                          children='Rank',
                                          n_clicks=0))]
    return weighting_sliders + n_top + button

def roa_TOPSIS(weights, n_top=3):
    df = data.copy()
    # Check that the weights are not all zero
    if all(value == 0 for value in weights.values()):
        weights = {key: 0.5 for key in output_dict}
    # Get data for TOPSIS based on which metrics do not have zero weight assigned
    df_TOPSIS = pd.DataFrame()
    weights_TOPSIS = []
    criteria = []
    for key in weights:
        if weights[key] > 0:
            df_TOPSIS = pd.concat([df_TOPSIS, df[key]], axis=1)
            weights_TOPSIS.append(weights[key])
            # Set whether the criteria should be maximized or minimized in TOPSIS
            if output_priority_dict[key] == 'max':
                criteria.append(MAX)
            else:
                criteria.append(MIN)
    # Perform TOPSIS
    TOPSIS_data = Data(df_TOPSIS.values, criteria, weights=weights_TOPSIS)
    dm = closeness.TOPSIS(mnorm='sum', wnorm='sum')
    dec = dm.decide(TOPSIS_data)
    rank = list(dec.rank_)
    # Get top results to output
    if len(df) < n_top:
        n_top = len(df)
    des = []
    des_num = []
    score = []
    color = []
    for i in range(n_top):
        row = df.iloc[rank.index(i+1)]
        des.append(row['description']),
        score.append(dec.e_.closeness[rank.index(i+1)])
    top_designs = pd.DataFrame({'Design': des,
                                'Score': score})
    return top_designs

def roa_bar_chart(top_designs):
    top_designs = top_designs[::-1]
    trace = go.Bar(x=top_designs['Score'], y=top_designs['Design'],
                   orientation='h',
                   marker=dict(color=colors[1]))
    layout = go.Layout(title=dict(text='Ranked Alternatives by Closeness to Ideal Solution',
                                  x=0.5),
                       xaxis=dict(title='Closeness to Ideal Solution Score'),
                       plot_bgcolor=('rgba(0,0,0,0)'),
                       paper_bgcolor=('rgba(0,0,0,0)'),
                       margin=dict(t=25))
    fig = go.Figure(trace, layout)
    return fig

def roa():
    return [dbc.Row(className='m-1',
                    children=[dbc.Col(children=roa_weighting_input(), width=4),
                              dbc.Col(children=[dcc.Graph(id='roa-bar-chart',
                                                          style={'height': '45vh'},
                                                          config={'displayModeBar': False})],
                                      width=8)])]
    
## Prediction Profiler ##   
def pp_filters():
    title_width = 4
    slider_width = 5
    label_width = 3
    loop_dict = input_dict.copy()
    label = 'Inputs:'
    display = []
    i = 1
    for key in loop_dict:
        if (i % 2) == 0:
            style = {'backgroundColor': bg_colors[0]}
        else:
            style = {}
        marks = slider_tick_marks_dict[key]
        range_and_step = slider_ranges_dict[key]
        display.append(dbc.Row(className='m-1',
                               style=style,
                               form=True,
                               children=[dbc.Label(loop_dict[key]+':', width=title_width),
                                         dbc.Col(children=dcc.Slider(id='pp-'+key+'-slider',
                                                                     min=range_and_step[0],
                                                                     max=range_and_step[1],
                                                                     step=range_and_step[2],
                                                                     marks={int(i) if i % 1 == 0 else i: '{}'.format(i) for i in marks},
                                                                     value=range_and_step[3]), width=slider_width),
                                         dbc.Label(id='pp-'+key+'-slider-label', width=label_width)]))
        i = i + 1
    title = [dbc.Label(label, style=style_font_bold)]
    return title + display

def pp_profiler(inputs):
    samples = 25
    X_point = pd.DataFrame({key: [inputs[key]] for key in inputs})
    fig = make_subplots(rows=len(output_dict),
                        cols=len(input_dict),
                        shared_xaxes=True,
                        shared_yaxes=True,
                        vertical_spacing=0.01,
                        horizontal_spacing=0.01)
    c = 1
    for col in input_dict:
        r = 1
        X = pd.DataFrame()
        for key in input_dict:
            if col == key:
                X[key] = list(np.linspace(slider_ranges_dict[key][0], slider_ranges_dict[key][1], samples))
            else:
                X[key] = [inputs[key] for x in range(samples)]
        for row in output_dict:
            model = getattr(SM, row)
            Y = model.predict(X)
            y = model.predict(X_point)
            trace = go.Scatter(x=X[col],
                               y=Y,
                               mode='lines',
                               line=dict(color=colors[1],
                                         width=4),
                               name='',
                               showlegend=False)
            ver_trace = go.Scatter(x=[inputs[col], inputs[col]], 
                                   y=[slider_ranges_dict[row][0], slider_ranges_dict[row][1]],
                                   name='',
                                   mode='lines',
                                   line=dict(color=greys[2],
                                             dash='dash'),
                                    showlegend=False)
            hor_trace = go.Scatter(x=[slider_ranges_dict[col][0], slider_ranges_dict[col][1]], 
                                   y=[y[0], y[0]],
                                   name='',
                                   mode='lines',
                                   line=dict(color=greys[2],
                                             dash='dash'),
                                    showlegend=False)
            point_trace = go.Scatter(x=[inputs[col]], 
                                     y=y,
                                     name='',
                                     mode='markers',
                                     marker_symbol='circle-open',
                                     marker=dict(color='black',
                                                 size=9),
                                      showlegend=False)
            fig.add_trace(trace, row=r, col=c)
            fig.add_trace(ver_trace, row=r, col=c)
            fig.add_trace(hor_trace, row=r, col=c)
            fig.add_trace(point_trace, row=r, col=c)
            if c == 1:
                fig.update_yaxes(title_text=output_dict[row],
                                 range=[slider_ranges_dict[row][0], slider_ranges_dict[row][1]],
                                 row=r, col=1)   
            if r == len(output_dict):
                fig.update_xaxes(title_text=input_plot_dict[col],
                                 range=[slider_ranges_dict[col][0], slider_ranges_dict[col][1]],
                                 row=len(output_dict), col=c)                   
            r = r + 1
        c = c + 1      
    fig.update_layout(plot_bgcolor=greys[0],
                      paper_bgcolor=('rgba(0,0,0,0)'),
                      width=1400,
                      margin=dict(l=10))
    return fig
    

def pp():
    filter_width = 3
    profiler_width = 9
    return [dbc.Row(className='m-1',
                    children=[dbc.Col(children=[html.Div(children=pp_filters(), style=style_section)],
                                      width=filter_width),
                              dbc.Col(children=[dcc.Graph(id='pp-profiler',
                                                          style={'height': '60vh',
                                                                 'width': '100%'},
                                                          config={'displayModeBar': False})],
                                      width=profiler_width)])]