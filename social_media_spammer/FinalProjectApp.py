import numpy as np
import pandas as pd
import time
import plotly.express as px
from dash import Dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import base64
from flask import send_file
import dash
import seaborn as sns
import plotly.graph_objs as go
import plotly.subplots as sp

follower_f = pd.read_csv('follower_followee_Adjust.csv',encoding="GBK")
font1 = {'family':'serif', 'color':'blue','size':18}
font2 = {'family':'serif', 'color':'darkred', 'size':15}
follower_f = follower_f.dropna()
gender = follower_f.groupby(['gender']).agg({'followings':'sum','fans':'sum','other_post_num':'sum',
                                             'Spam_num':'sum','repost_num':'sum','comment_num':'sum'})
gender.reset_index(inplace = True)
first_or_last = follower_f.groupby(['first_or_last']).agg({'followings':'sum','fans':'sum','other_post_num':'sum',
                                             'Spam_num':'sum','repost_num':'sum','comment_num':'sum'})
isVIP = follower_f.groupby(['isVIP']).agg({'followings':'sum','fans':'sum','other_post_num':'sum',
                                             'Spam_num':'sum','repost_num':'sum','comment_num':'sum'})
isVIP.reset_index(inplace = True)
level = follower_f.groupby(['level']).agg({'followings':'sum','fans':'sum','other_post_num':'sum',
                                             'Spam_num':'sum','repost_num':'sum','comment_num':'sum'})
gender.reset_index(inplace=True)
first_or_last.reset_index(inplace=True)
isVIP.reset_index(inplace=True)
level.reset_index(inplace=True)
# 绿 #98FB98 蓝#ADD8E6 粉#F5DEB3 灰#D3D3D3
pd.set_option('display.float_format', lambda x: '%.2f' % x)
load_figure_template('CYBORG')
style_all = {'textAlign': 'center','font-size': 60}
# 居中
style1 = {'textAlign': 'center','font-size': 40} #一级标题
style2 = {"textAlign":"center", 'font-size': 30} #二级标题
style_text = {"textAlign":"center",'font-size': 25} #html.p用的普通文本
style3 = {'margin': '10px 0', 'color': 'white'} #三级标题
style_img = {'width': '70%',
             'display': 'block',
             'margin-left': 'auto',
             'margin-right': 'auto',
             'padding': '10px'}  #图片/视频
style2_bottom_line={"textAlign":"center", 'font-size': 30,'textDecoration': 'underline', 'textDecorationColor': 'pink'}
style_head = {
    "textAlign": "center",
    "fontSize": 15,
    "fontStyle": "italic",
    "textDecoration": "underline",
    "textDecorationColor": "yellow",
}
style_label = {"font-size": "15px",'color': "#98FB98"}
style_textarea = {'width': '100%', 'height': 100}
# external_stylesheets = ['https://cdn.jsdelivr.net/npm/bootswatch@4.5.2/dist/united/bootstrap.min.css']
my_app = Dash("Sina Weibo Data Visualization", external_stylesheets=[dbc.themes.QUARTZ])
my_app.layout = html.Div(style={'padding': '10px 10px 10px 10px'},
    children=[
    html.H1("Sina Weibo Data Visualization", style=style_all),
    html.P("Analyze the spammers of sina weibo and the other related features",
             style= style2),
    html.Br(),
    html.Img(src='jetbrains://pycharm/navigate/reference?project=LAB5.py&path=FinalProject/sinaweiboLogo.png',
             style={'width': '5%',
                     'display': 'block',
                     'margin-left': 'auto',
                     'margin-right': 'auto',
                     'padding': '10px'}),
    html.Br(),
    dcc.Tabs(id='tabs',
          children=[
              dcc.Tab(label='Final Project Introduction', value='introduction'),
              dcc.Tab(label='Line Plot', value='line'),
              dcc.Tab(label='Bar Plot & Count Plot', value='bar'),
              dcc.Tab(label='Histogram Plot', value='hist'),
              dcc.Tab(label='Pie Plot', value='pie'),
              dcc.Tab(label='KDE Plot ', value='kde'),
              dcc.Tab(label='3D Plot', value='3d')
          ], style={"color":"black"}, value='introduction'),
    html.Br(),
    html.Br(),
    html.Div(id='layout')
    ]
)

youtube_link = "https://www.youtube.com/watch?v=jNiUZtnOxss&ab_channel=B2Bwhiteboard"
introducion_layout = html.Div([
# "#FF6347", "#6A5ACD", "#20B2AA"
    html.H3("Final Project Introduction", style=style1),
    # html.Img(src='file:///D:/E/cs/python/informationvisualisation/FinalProject/sinaweiboRed.jpg', style={'width': '40%',
    #                                                      'display': 'block',
    #                                                      'margin-left': 'auto',
    #                                                      'margin-right': 'auto',
    #                                                      'padding': '10px'}),
    # html.Br(),
    # html.Video(src="https://www.youtube.com/watch?v=jNiUZtnOxss&ab_channel=B2Bwhiteboard",
    #            controls=True, autoPlay=True, style=style_img),
    html.Iframe(src=youtube_link, width="560", height="315"),
    html.Br(),
    html.Br(),
    html.H4(" What is this mock app about?",style= style2_bottom_line),
    html.Br(),
    html.P("This is a visual analysis dashboard that analyzes the spam situation on Sina Weibo.",
           style = style_text),
    html.H4(" What does this app shows?",style= style2_bottom_line),
    html.Br(),
    html.P("The visual analysis dashboard for Sina Weibo offers an insightful examination of the prevalent spam scenario within the platform.",
           style=style_text),
    html.P([
        'You can download the original dataset from the website: ',
        html.A('https://archive.ics.uci.edu/dataset/323/microblogpcu',
               href='https://archive.ics.uci.edu/dataset/323/microblogpcu',
               target='_blank',
               style={'color': 'lightblue'}),
        'or click the button below.'
    ]),
    html.Button("Download CSV", id="btn_csv"),
    dcc.Download(id="download-dataframe-csv"),
    html.Br()
])

@my_app.callback(
    Output(component_id='introduction', component_property='children'),
    [Input(component_id='introduction', component_property='figure')])

def render_content(introduction):

       return introduction

@my_app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    return dcc.send_data_frame(follower_f.to_csv, "follower_followee.csv")

# =============================================================
# line plot
# =============================================================
features_nospam = ['followings', 'fans', 'other_post_num', 'repost_num', 'comment_num']
features_bar =['followings', 'fans', 'other_post_num', 'Spam_num',
       'repost_num', 'comment_num']
line_layout = html.Div([
    html.Header("(Line plots can immediately assist us in understanding the diverse characteristic attributes of users.)", style = style_head),
    html.Br(),
    html.H3('Sina Weibo Spamming Line Plots', style= style1),
    html.Label("Select a Line Plot to explore...", style=style_label),
    html.Br(),
    dcc.Dropdown(id="line-dropdown",
                 options=[{"label": i, "value": i} for i in features_bar],
                 multi=True,
                 placeholder="Select Feature(s)...",
                 style={"color": "#ADD8E6",
                        "width": "50%"},
                 value='followings'
                 ),
    dcc.Loading(
            id="loading-1",
            type="default",
            children=html.Div(id="loading-output-1")
        ),
    html.Br(),
    html.Div(id='output1'),
    html.Br(),
    html.Br(),
    dcc.Graph(id='line-plot', style={'text-align': "center"}),
    html.Br(),
    dcc.Textarea(
        id='textarea-state-example-line',
        value='You can leave your comments and thoughts here...',
        style=style_textarea,
    ),
    html.Button('Submit', id='textarea-state-example-button-line', n_clicks=0),
    html.Div(id='textarea-state-example-output-line', style={'whiteSpace': 'pre-line'})
    ]
)
@my_app.callback(
    Output('textarea-state-example-output-line', 'children'),
    Input('textarea-state-example-button-line', 'n_clicks'),
    State('textarea-state-example-line', 'value')
)
def update_output(n_clicks, value):
    if n_clicks > 0:
        return 'You have entered: \n{}'.format(value)

@my_app.callback(
    Output("loading-output-1", "children"),
    Input("line-dropdown", "value"))
def input_triggers_spinner(value):
    time.sleep(1)
    return value

@my_app.callback(
    [Output(component_id='output1', component_property='children'),
    Output(component_id='line-plot', component_property='figure')],
    Input(component_id='line-dropdown', component_property='value')
)

def update_line(selected_feature):
    fig = px.line(level, y=selected_feature, x="level", title=f'Line Plot of {selected_feature} and Level', height=500).update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    fig.update_layout(
        title={'y': .95, 'x': 0.5},
        title_font=dict(color = "white", size=30, family="Times New Roman"),
        legend_title_font=dict(color="green", family='Courier New'),
        font=dict(color='yellow', size=18, family="Courier New"),
        xaxis=dict(
            title_font=dict(color="white", family='Courier New', size=30),
            tickfont=dict(color="white", size=30),
            # 这个表示的是坐标轴线的宽度
            linewidth=4
        ),
        yaxis=dict(
            title_font=dict(color="white", family='Courier New', size=30),
            tickfont=dict(color="white", size=30),
            linewidth=4
        )
    )
    # 设置线条宽度为4
    fig.update_traces(
        # 轨迹是图表中的数据系列，例如散点、线、柱状图等
        line=dict(width=4)
    )

    return f'The selected feature(s) inside the dropdown menu is {selected_feature}', fig
# =============================================================
# bar plot & count plot
# =============================================================
categorical_bar = [gender, first_or_last, isVIP, level]
categorical = ["gender", "first_or_last", "isVIP"]
Top_level = (follower_f.groupby(['level']).agg({'followings':'sum','fans':'sum','other_post_num':'sum',
                                             'Spam_num':'sum','repost_num':'sum','comment_num':'sum'}).
             sort_values("fans", ascending=False).head(10))
Top_level.reset_index(inplace=True)
graph1 = dcc.Graph(id='bar-plot', style={'width': '50%', 'display': 'inline-block', 'text-align': 'center'})
graph2 = dcc.Graph(id='bar-plot-level', style={'width': '50%', 'display': 'inline-block', 'text-align': 'center'})
bar_layout = html.Div([
    html.Header("The bar chart can display the frequency distribution or count of different categories or groups, especially useful for illustrating the distribution of discrete data.",
                style=style_head),
    html.Br(),
    html.H3('Sina Weibo Spamming Bar Plots', style= style1),
    html.Label("Select a Bar Plot to explore user gender's  and level's correlation with other features.",
               style=style_label),
    html.Br(),
    dcc.Dropdown(id="bar-dropdown",
                 options=[{"label": i, "value": i} for i in features_bar],
                 multi=True,
                 placeholder="Select Feature(s)...",
                 style={"color": "#ADD8E6",
                        "width": "50%"},
                 value='followings'
                 ),
    dcc.Loading(
        id="loading-2",
        type="default",
        children=html.Div(id="loading-output-2")
    ),
    html.Br(),
    html.Div(id='output2'),
    html.Br(),
    html.Br(),
    html.Div([
        graph1,
        graph2
    ], style={'text-align': 'center'}),
    html.Br(),
    html.Label("Select a Count Plot to explore different features' user count",
               style=style_label),
    dcc.Dropdown(id="bar-dropdown-count",
                 options=[{"label": i, "value": i} for i in categorical],
                 multi=False,
                 placeholder="Select Feature(s)...",
                 style={"color": "#ADD8E6",
                        "width": "50%"},
                 value='gender'
                 ),
    dcc.Loading(
        id="loading-2-2",
        type="default",
        children=html.Div(id="loading-output-2-2")
    ),
    html.Br(),
    html.Div(id='output2-count'),
    dcc.Graph(id='bar-plot-count', style={'width': '80%', 'display': 'inline-block', 'text-align': 'center'}),
    html.Br(),
    dcc.Textarea(
        id='textarea-state-example',
        value='You can leave your comments and thoughts here...',
        style=style_textarea,
    ),
    html.Button('Submit', id='textarea-state-example-button', n_clicks=0),
    html.Div(id='textarea-state-example-output', style={'whiteSpace': 'pre-line'})
]
)
@my_app.callback(
    [Output("loading-output-2", "children"),
    Output("loading-output-2-2", "children")],
    [Input("bar-dropdown", "value"),
    Input("bar-dropdown-count", "value")]
     )
def input_triggers_spinner(value,value2):
    time.sleep(1)
    return value,value2

@my_app.callback(
    [Output(component_id='output2', component_property='children'),
    Output(component_id='bar-plot', component_property='figure'),
    Output(component_id='bar-plot-level', component_property='figure'),
    Output(component_id='bar-plot-count', component_property='figure'),
     ],
    [Input(component_id='bar-dropdown', component_property='value'),
     Input(component_id='bar-dropdown-count', component_property='value')
     ]
)

def update_line(selected_feature, count_feature):
    fig1 = px.bar(gender, x=selected_feature, y="gender", title=f'bar plot of {selected_feature} of different gender',
                  height=500).update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    fig1.update_layout(
        title={'y': .95, 'x': 0.5},
        title_font=dict(color="white", size=30, family="Times New Roman"),
        legend_title_font=dict(color="green", family='Courier New'),
        font=dict(color='yellow', size=18, family="Courier New"),
        xaxis=dict(
            title_font=dict(color="white", family='Courier New', size=30),
            tickfont=dict(color="white", size=30),
            # 这个表示的是坐标轴线的宽度
            linewidth=4
        ),
        yaxis=dict(
            title_font=dict(color="white", family='Courier New', size=30),
            tickfont=dict(color="white", size=30),
            linewidth=4
        )
    )
    fig2 = px.bar(Top_level, x="level", y=selected_feature,
        height=500,title=f"Bar plot of top level of {selected_feature}").update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    fig2.update_layout(
        title={'y': .95, 'x': 0.5},
        title_font=dict(color="white", size=30, family="Times New Roman"),
        legend_title_font=dict(color="green", family='Courier New'),
        font=dict(color='yellow', size=18, family="Courier New"),
        xaxis=dict(
            title_font=dict(color="white", family='Courier New', size=30),
            tickfont=dict(color="white", size=30),
            # 这个表示的是坐标轴线的宽度
            linewidth=4
        ),
        yaxis=dict(
            title_font=dict(color="white", family='Courier New', size=30),
            tickfont=dict(color="white", size=30),
            linewidth=4
        )
    )
    # s = follower_f[count_feature].value_counts(ascending=False)
    s = follower_f[count_feature].value_counts().reset_index()
    s.columns = [count_feature, 'Count']
    current_palette = sns.color_palette()
    fig3 = px.bar( s, y=count_feature, x='Count',
        height=500,title=f"Count plot of {selected_feature}").update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    fig3.update_layout(
        title={'y': .95, 'x': 0.5},
        title_font=dict(color="white", size=30, family="Times New Roman"),
        legend_title_font=dict(color="green", family='Courier New'),
        font=dict(color='yellow', size=18, family="Courier New"),
        xaxis=dict(
            title_font=dict(color="white", family='Courier New', size=30),
            tickfont=dict(color="white", size=30),
            # 这个表示的是坐标轴线的宽度
            linewidth=4
        ),
        yaxis=dict(
            title_font=dict(color="white", family='Courier New', size=30),
            tickfont=dict(color="white", size=30),
            linewidth=4
        )
    )
    return f'The selected feature(s) inside the dropdown menu is {selected_feature}', fig1,fig2,fig3

@my_app.callback(
    Output('textarea-state-example-output', 'children'),
    Input('textarea-state-example-button', 'n_clicks'),
    State('textarea-state-example', 'value')
)
def update_output(n_clicks, value):
    if n_clicks > 0:
        return 'You have entered: \n{}'.format(value)
# =============================================================
# histogram  layout的顺序一定要和tab的顺序一致
# ============================================================
features_hist = ["level","followings","Spam_num"]
histogram_layout = html.Div([
    html.Header("Histograms portray data distribution via frequency within defined intervals (bins).", style=style_head),
    html.Br(),
    html.H3('Sina Weibo Spamming Histogram Plots', style= style1),
    html.Label("Select a Histogram Plot to explore...", style=style_label),
    html.Br(),
    dcc.Dropdown(id="hist-dropdown",
                 options=[{"label": i, "value": i} for i in features_hist],
                 multi=False,
                 placeholder="Select Feature(s)...",
                 style={"color": "#ADD8E6",
                        "width": "50%"},
                 value='level'
                 ),
    dcc.Loading(
            id="loading-hist",
            type="default",
            children=html.Div(id="loading-output-hist")
        ),
    html.Br(),
    html.Div(id='output-hist'),
    html.Br(),
    html.Br(),
    dcc.Graph(id='hist-plot', style={'text-align': "center"}),
    html.Br(),
    dcc.Textarea(
        id='textarea-state-example-hist',
        value='You can leave your comments and thoughts here...',
        style=style_textarea,
    ),
    html.Button('Submit', id='textarea-state-example-button-hist', n_clicks=0),
    html.Div(id='textarea-state-example-output-hist', style={'whiteSpace': 'pre-line'})
    ]
)
@my_app.callback(
    Output("loading-output-hist", "children"),
    Input("hist-dropdown", "value"))
def input_triggers_spinner(value):
    time.sleep(1)
    return value

@my_app.callback(
    Output('textarea-state-example-output-hist', 'children'),
    Input('textarea-state-example-button-hist', 'n_clicks'),
    State('textarea-state-example-hist', 'value')
)
def update_output(n_clicks, value):
    if n_clicks > 0:
        return 'You have entered: \n{}'.format(value)

@my_app.callback(
    [Output(component_id='output-hist', component_property='children'),
    Output(component_id='hist-plot', component_property='figure')],
    Input(component_id='hist-dropdown', component_property='value')
)

def update_hist(selected_feature):
    fig = px.histogram(data_frame=follower_f, x=selected_feature, color="gender", marginal='box',
                       title=f'Histogram Plot of {selected_feature}', height=500).update_layout(
                    {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    fig.update_layout(
        title={'y': .95, 'x': 0.5},
        title_font=dict(color = "white", size=30, family="Times New Roman"),
        legend_title_font=dict(color="green", family='Courier New'),
        font=dict(color='yellow', size=18, family="Courier New"),
        xaxis=dict(
            title_font=dict(color="white", family='Courier New', size=30),
            tickfont=dict(color="white", size=30),
            # 这个表示的是坐标轴线的宽度
            linewidth=4
        ),
        yaxis=dict(
            title_font=dict(color="white", family='Courier New', size=30),
            tickfont=dict(color="white", size=30),
            linewidth=4
        )
    )

    return f'The selected feature(s) inside the dropdown menu is {selected_feature}', fig
# ================================================================
# pie chart
# ================================================================
male_count = (follower_f['gender'] == 'male').sum()
female_count = (follower_f['gender'] == 'female').sum()
total_val = male_count + female_count
val = [male_count, female_count]
labels_gender = ['Male', 'Female']
gender_pie = {'label':labels_gender,
              'values': val}
# df = pd.DataFrame(gender_pie)
Normal_count = (follower_f['isVIP'] == 'Normal').sum()
VIP_count = (follower_f['isVIP'] == 'VIP').sum()
SVIP_count = (follower_f['isVIP'] == 'SVIP').sum()
VVIP_count = (follower_f['isVIP'] == 'VVIP').sum()
val_vip = [Normal_count,VIP_count,SVIP_count,VVIP_count]
labels_vip = ['Normal', 'VIP', 'SVIP', 'VVIP']
vip_pie = {'label':labels_vip,
              'values': val_vip}
# df = pd.DataFrame(gender_pie)
pie_layout = html.Div([
    html.Header("A pie plot visually represents data categories as proportional segments in a circle.",style=style_head),
    html.Br(),
    html.H3('Sina Weibo Spamming Pie Plots', style= style1),
    html.Label("Select a Pie Plot to explore...", style=style_label),
    html.Br(),
    dcc.RadioItems(id="pie-checklist",
                 options=[{"label":"Pie chart of total user divide by gender    ", "value":'gender'},
                          {'label':'Pie chart of users with different VIP levels    ','value':'vip'},
                          {'label':'Pie chart of Spam_num divide by gender','value    ':'spam_num_gender'},
                          {'label':'Pie chart of Spam_num divide by VIP class','value':'spam_num_vip'}
                          ],
                 inline=True,
                 # multi=False,
                 style={"color": "#ADD8E6", "font-size":20},
                 value='gender'
                 ),
    html.Br(),
    html.Br(),
    dcc.Graph(id='pie-plot', style={'text-align': "center"}),
    dcc.Loading(
            id="loading-pie",
            type="default",
            children=html.Div(id="loading-output-pie")
        ),
    html.Br(),
    dcc.Textarea(
        id='textarea-state-example-pie',
        value='You can leave your comments and thoughts here...',
        style=style_textarea,
    ),
    html.Button('Submit', id='textarea-state-example-button-pie', n_clicks=0),
    html.Div(id='textarea-state-example-output-pie', style={'whiteSpace': 'pre-line'})
])

@my_app.callback(
    Output('textarea-state-example-output-pie', 'children'),
    Input('textarea-state-example-button-pie', 'n_clicks'),
    State('textarea-state-example-pie', 'value')
)
def update_output(n_clicks, value):
    if n_clicks > 0:
        return 'You have entered: \n{}'.format(value)

@my_app.callback(
    Output("loading-output-pie", "children"),
    Input("pie-checklist", "value"))
def input_triggers_spinner(value):
    time.sleep(1)
    return value

@my_app.callback(
    [Output(component_id='pie-plot', component_property='figure')],
    Input(component_id='pie-checklist', component_property='value')
)

def update_line(pie_plot):
    # fig = sp.make_subplots(rows=2, cols=2, subplot_titles=('Gender Distribution', 'VIP Levels Distribution','1','3'),
    #                        specs=[[{'type': 'pie'}, {'type': 'pie'}],[{'type': 'pie'}, {'type': 'pie'}]]
    #                        )
    if pie_plot == 'gender':
        # fig = fig.add_trace(
        #     go.Pie(labels=labels_gender, values=val, textinfo='percent'),
        #     row=1, col=1
        # )
        fig1 = px.pie(gender_pie,
                      values='values',
               names='label'
               ).update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        fig1.update_traces(textinfo='label+percent',
                          textposition='inside',  # 将文本放在扇区内部
                          insidetextfont=dict(family='Arial', size=14, color='white')
                          )
        fig1.update_layout(
            title={'y': .95, 'x': 0.5},
            title_font=dict(color="white", size=30, family="Times New Roman"),
            legend_title_font=dict(color="green", family='Courier New'),
            font=dict(color='yellow', size=18, family="Courier New"),
            xaxis=dict(
                title_font=dict(color="white", family='Courier New', size=30),
                tickfont=dict(color="white", size=30),
                # 这个表示的是坐标轴线的宽度
                linewidth=4
            ),
            yaxis=dict(
                title_font=dict(color="white", family='Courier New', size=30),
                tickfont=dict(color="white", size=30),
                linewidth=4
            )
        )
        return [go.Figure(data=fig1)]
    if pie_plot == 'vip':
        # fig = fig.add_trace(
        #     go.Pie(labels=labels_vip, values=val_vip, textinfo='percent'),
        #     row=1, col=2
        # )
        fig2 = px.pie(vip_pie, values='values', names='label').update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        fig2.update_traces(textinfo='label+percent',
                          textposition='inside',  # 将文本放在扇区内部
                          insidetextfont=dict(family='Arial', size=14, color='white')
                          )
        fig2.update_layout(
            title={'y': .95, 'x': 0.5},
            title_font=dict(color="white", size=30, family="Times New Roman"),
            legend_title_font=dict(color="green", family='Courier New'),
            font=dict(color='yellow', size=18, family="Courier New"),
            xaxis=dict(
                title_font=dict(color="white", family='Courier New', size=30),
                tickfont=dict(color="white", size=30),
                # 这个表示的是坐标轴线的宽度
                linewidth=4
            ),
            yaxis=dict(
                title_font=dict(color="white", family='Courier New', size=30),
                tickfont=dict(color="white", size=30),
                linewidth=4
            )
        )
        return [go.Figure(data=fig2)]
    if pie_plot == 'spam_num_gender':
        # fig = fig.add_trace(
        #     px.pie(gender, labels=labels_gender, values="Spam_num"),
        #     row=2, col=1
        # )
        fig3 = px.pie(gender, values=gender['Spam_num'], names=gender["gender"]).update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        fig3.update_traces(textinfo='label+percent',
                          textposition='inside',  # 将文本放在扇区内部
                          insidetextfont=dict(family='Arial', size=14, color='white')
                          )
        fig3.update_layout(
            title={'y': .95, 'x': 0.5},
            title_font=dict(color="white", size=30, family="Times New Roman"),
            legend_title_font=dict(color="green", family='Courier New'),
            font=dict(color='yellow', size=18, family="Courier New"),
            xaxis=dict(
                title_font=dict(color="white", family='Courier New', size=30),
                tickfont=dict(color="white", size=30),
                # 这个表示的是坐标轴线的宽度
                linewidth=4
            ),
            yaxis=dict(
                title_font=dict(color="white", family='Courier New', size=30),
                tickfont=dict(color="white", size=30),
                linewidth=4
            )
        )
        # return fig3
        return [go.Figure(data=fig3)]
    if pie_plot == 'spam_num_vip':
        # fig = fig.add_trace(
        #     go.Pie(isVIP, labels=labels_vip, values="Spam_num", textinfo='percent'),
        #     row=2, col=2
        # )
        fig4 = px.pie(isVIP,values='Spam_num',names=labels_vip).update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        fig4.update_traces(textinfo='label+percent',
                          textposition='inside',  # 将文本放在扇区内部
                          insidetextfont=dict(family='Arial', size=14, color='white')
                          )
        fig4.update_layout(
            title={'y': .95, 'x': 0.5},
            title_font=dict(color="white", size=30, family="Times New Roman"),
            legend_title_font=dict(color="green", family='Courier New'),
            font=dict(color='yellow', size=18, family="Courier New"),
            xaxis=dict(
                title_font=dict(color="white", family='Courier New', size=30),
                tickfont=dict(color="white", size=30),
                # 这个表示的是坐标轴线的宽度
                linewidth=4
            ),
            yaxis=dict(
                title_font=dict(color="white", family='Courier New', size=30),
                tickfont=dict(color="white", size=30),
                linewidth=4
            )
        )
        return [go.Figure(data=fig4)]
# ===============================================
# Other Plot
# ===============================================
# dropdown_kde = dcc.Dropdown(id="kde-dropdown",
#                  options=[{"label": i, "value": i} for i in features_bar],
#                  multi=True,
#                  placeholder="Select Feature(s)...",
#                  style={"color": "#ADD8E6",
#                         "width": "50%"},
#                  value=['followings','other_post_num']
#                  )
# dropdown_violin = dcc.Dropdown(id="violin-dropdown",
#                  options=[{"label": i, "value": i} for i in ['gender','isVIP']],
#                  multi=True,
#                  placeholder="Select Feature(s)...",
#                  style={"color": "#ADD8E6",
#                         "width": "50%"},
#                  value='gender'
#                  )
# graph_kde = dcc.Graph(id='kde-plot', style={'width': '50%', 'display': 'inline-block', 'text-align': 'center'})
# graph_violin = dcc.Graph(id='bar-plot-level', style={'width': '50%', 'display': 'inline-block', 'text-align': 'center'})
other_layout = html.Div([
    html.Header("There are plenty of other pictures that do a great job of showing the characteristics and relationships of the data", style=style_head),
    html.Br(),
    html.H3('Sina Weibo Spamming KDE Plots', style= style1),
    html.Br(),
    # html.Div([dropdown_kde,dropdown_violin]),
    # html.Div([graph_kde,graph_violin])
    html.Label("Select a KDE Plot to explore...", style=style_label),
    dcc.Dropdown(id="kde-dropdown",
                 options=[{"label": i, "value": i} for i in features_bar],
                 multi=True,
                 placeholder="Select Feature(s)...",
                 style={"color": "#ADD8E6",
                        "width": "50%"},
                 value=['followings','other_post_num']
                 ),
    dcc.Loading(
            id="loading-kde",
            type="default",
            children=html.Div(id="loading-output-kde")
        ),
    html.Br(),
    dcc.Graph(id='kde-plot', style={'width':'100%','display': 'inline-block', 'text-align': 'center'}),
    html.Br(),
    dcc.Textarea(
        id='textarea-state-example-kde',
        value='You can leave your comments and thoughts here...',
        style=style_textarea,
    ),
    html.Button('Submit', id='textarea-state-example-button-kde', n_clicks=0),
    html.Div(id='textarea-state-example-output-kde', style={'whiteSpace': 'pre-line'})
    # html.Label("Select a Violin Plot to explore...", style={"font-size": 25}),
    # dcc.Dropdown(id="violin-dropdown",
    #              options=[{"label": i, "value": i} for i in categorical_bar],
    #              multi=True,
    #              placeholder="Select Feature(s)...",
    #              style={"color": "#ADD8E6",
    #                     "width": "50%"},
    #              # value=['followings', 'other_post_num']
    #              ),
    # html.Br(),
    # dcc.Dropdown(id="violin-dropdown2",
    #              options=[{"label": i, "value": i} for i in categorical_bar],
    #              multi=True,
    #              placeholder="Select Feature(s)...",
    #              style={"color": "#ADD8E6",
    #                     "width": "50%"},
    #              # value=['followings', 'other_post_num']
    #              ),
    # html.Br(),
    # dcc.Graph(id='violin-plot', style={"width":"100%",'display': 'inline-block', 'text-align': 'center'}),


    ])
@my_app.callback(
    Output("loading-output-kde", "children"),
    Input("kde-dropdown", "value"))
def input_triggers_spinner(value):
    time.sleep(1)
    return value

@my_app.callback(
    Output('textarea-state-example-output-kde', 'children'),
    Input('textarea-state-example-button-kde', 'n_clicks'),
    State('textarea-state-example-kde', 'value')
)
def update_output(n_clicks, value):
    if n_clicks > 0:
        return 'You have entered: \n{}'.format(value)

@my_app.callback(
    [Output(component_id='kde-plot', component_property='figure'),
            # Output(component_id='violin-plot', component_property='figure'),
     ],
    [Input(component_id='kde-dropdown', component_property='value'),
     # Input(component_id='violin-dropdown', component_property='value'),
     # Input(component_id='violin-dropdown2', component_property='value')
     ])

def update_kde(kde_plot):
    fig_kde = px.density_contour(follower_f, x=kde_plot,title=f"KDE Plot of {kde_plot}").update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    fig_kde.update_xaxes(range=[0, 3500])
    fig_kde.update_layout(
        title={'y': .95, 'x': 0.5},
        title_font=dict(color="white", size=30, family="Times New Roman"),
        legend_title_font=dict(color="green", family='Courier New'),
        font=dict(color='yellow', size=18, family="Courier New"),
        xaxis=dict(
            title_font=dict(color="white", family='Courier New', size=30),
            tickfont=dict(color="white", size=30),
            # 这个表示的是坐标轴线的宽度
            linewidth=4
        ),
        yaxis=dict(
            title_font=dict(color="white", family='Courier New', size=30),
            tickfont=dict(color="white", size=30),
            linewidth=4
        )
    )
    # fig_area = px.violin(follower_f,x=violin_plot,y=violin_dropdown2,title=f"Violin Plot of Spam_num and {violin_plot}").update_layout({
    # 'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    # # fig_area.update_xaxes(range=[0, 3500])
    # # fig_area.update_yaxes(range=[0, 10000])
    # fig_area.update_layout(
    #     title={'y': .95, 'x': 0.5},
    #     title_font=dict(color="white", size=30, family="Times New Roman"),
    #     legend_title_font=dict(color="green", family='Courier New'),
    #     font=dict(color='yellow', size=18, family="Courier New"),
    #     xaxis=dict(
    #         title_font=dict(color="white", family='Courier New', size=30),
    #         tickfont=dict(color="white", size=30),
    #         # 这个表示的是坐标轴线的宽度
    #         linewidth=4
    #     ),
    #     yaxis=dict(
    #         title_font=dict(color="white", family='Courier New', size=30),
    #         tickfont=dict(color="white", size=30),
    #         linewidth=4
    #     )
    # )
    return [go.Figure(data=fig_kde)]
# ===========================================================
# 3d plot
# ===========================================================
three_layout = html.Div([
    html.H1("3D Scatter Plot",style=style1),
    dcc.Graph(id='scatter-plot' ,style={ 'height': '50vh'}),
    html.Br(),
    html.Br(),
    dcc.Graph(
        id='scatter-animation',
        figure={}
    ),
    html.Label('Select Spam_num value with Slider'),
    dcc.Slider(
        id='spam-slider',
        min=0,
        max=100,
        step=1,
        value=follower_f['Spam_num'].min(),
        marks={str(s): {'label': str(s), 'style': {'color': '#98FB98', 'font-size': '10px'}} for s in
               np.arange(0, 101, 10)},
    ),
    dcc.Loading(
            id="loading-3d",
            type="default",
            children=html.Div(id="loading-output-3d")
        ),
    dcc.Textarea(
        id='textarea-state-example-3d',
        value='You can leave your comments and thoughts here...',
        style=style_textarea,
    ),
    html.Button('Submit', id='textarea-state-example-button-3d', n_clicks=0),
    html.Div(id='textarea-state-example-output-3d', style={'whiteSpace': 'pre-line'})
])

@my_app.callback(
    Output('textarea-state-example-output-3d', 'children'),
    Input('textarea-state-example-button-3d', 'n_clicks'),
    State('textarea-state-example-3d', 'value')
)
def update_output(n_clicks, value):
    if n_clicks > 0:
        return 'You have entered: \n{}'.format(value)

@my_app.callback(
    Output("loading-output-3d", "children"),
    Input("spam-slider", "value"))
def input_triggers_spinner(value):
    time.sleep(1)
    return value

@my_app.callback(
    [Output('scatter-plot', 'figure'),
    Output('scatter-animation', 'figure')],
    [Input('spam-slider', 'value')]
)
def update_3d_plot(spam_value):
    filtered_df = follower_f[follower_f['Spam_num'] == spam_value]
    trace = go.Scatter3d(
        x=filtered_df['followings'],
        y=filtered_df['fans'],
        z=filtered_df['Spam_num'],
        mode='markers',
        marker=dict(
            size=8,
            color='rgb(0,0,255)',
            opacity=0.8
        )
    )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='Followings'),
            yaxis=dict(title='Fans'),
            zaxis=dict(title='Spam_num'),
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig = go.Figure(data=[trace], layout=layout).update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    fig.update_traces(marker=dict(color='rgba(0, 0, 255, 0.7)'))
    filtered_df = follower_f[follower_f['Spam_num'] == spam_value]
    fig2 = px.scatter(filtered_df, x='followings', y='fans',size_max=60, animation_frame='Spam_num', title='Scatter Animation').update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    fig2.update_layout(
        title={'y': .95, 'x': 0.5},
        title_font=dict(color="white", size=30, family="Times New Roman"),
        legend_title_font=dict(color="green", family='Courier New'),
        font=dict(color='yellow', size=18, family="Courier New"),
        xaxis=dict(
            title_font=dict(color="white", family='Courier New', size=30),
            tickfont=dict(color="white", size=30),
            # 这个表示的是坐标轴线的宽度
            linewidth=4
        ),
        yaxis=dict(
            title_font=dict(color="white", family='Courier New', size=30),
            tickfont=dict(color="white", size=30),
            linewidth=4
        )
    )
    return fig,fig2




@my_app.callback(
    Output(component_id='layout', component_property='children'),
    [Input(component_id='tabs', component_property='value')])

def update_layout(ques):
    if ques=='introduction':
        return introducion_layout
    elif ques=='line':
        return line_layout
    elif ques=='bar':
        return bar_layout
    elif ques=='pie':
        return pie_layout
    elif ques=='hist':
        return histogram_layout
    elif ques=='kde':
        return other_layout
    elif ques=='3d':
        return three_layout

my_app.run_server(
        port=8045,
        host='0.0.0.0'
    )