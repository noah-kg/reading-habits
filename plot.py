import pandas as pd
import numpy as np
import cufflinks as cf
import chart_studio.plotly as py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

from plotly.offline import download_plotlyjs, init_notebook_mode
init_notebook_mode(connected=True)
cf.go_offline()

# Remove unnecessary control items in figures (for Plotly)
config = {
    'modeBarButtonsToRemove': ['zoomIn', 'zoomOut', 'resetScale2d', 'select2d', 'lasso2d'],
    'responsive': True,
    'displaylogo': False,
    'toImageButtonOptions': {
        'format': 'png',  # one of png, svg, jpeg, webp
        'filename': 'reading-habits',
        'scale': 1
      }
}

def gen_layout(fig, title='', title_size=40, legendy_anchor='bottom', legendx_anchor='center', 
               height=600, showlegend=False, plot_bg='#f0f0f0', paper_bg='#f0f0f0', 
               y_title=None, x_title=None, l_mar=45, r_mar=45, t_mar=115, b_mar=45, 
               x_showline=False, y_showline=False, linecolor='black', y_labels=True, 
               gridcolor='#cbcbcb', barmode='group', x_showgrid=False, y_showgrid=False,
               fontcolor="#001c40", fontsize=14):
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=title_size, family="Baskerville, Bold", color=fontcolor)),
        height=height,
        showlegend=showlegend,
        barmode=barmode,
        plot_bgcolor=plot_bg,
        paper_bgcolor=paper_bg,
        yaxis_title=y_title,
        xaxis_title=x_title,
        margin=dict(l=l_mar, r=r_mar, t=t_mar, b=b_mar),        
        xaxis=dict(
            showgrid=x_showgrid,
            showline=x_showline,
            linecolor=linecolor,
            gridcolor=gridcolor,
            # autorange=True
        ),
        yaxis=dict(
            showgrid=y_showgrid,
            showline=y_showline,
            showticklabels=y_labels,
            linecolor=linecolor,
            gridcolor=gridcolor,
            # autorange=True
        ),
        font=dict(
            family="Baskerville",
            color=fontcolor,
            size=fontsize
        )
    )
    return fig

def gen_menu(active, buttons):
    """
    Generates menu configurations for dropdown.
    
    active: default button to have upon generation
    buttons: list of different menu options
    """
    updatemenus = [
        go.layout.Updatemenu(
            active=active,
            buttons=buttons,
            x=1.0,
            xanchor='right',
            y=1.11,
            yanchor='top'
        )
    ]
    return updatemenus

def gen_buttons(vals, multi=0):
    """
    Generates dropdown menu buttons.
    
    vals: list of values to turn into buttons
    """
    buttons_opts = []    
    i = 0
    for val in vals:
        if multi:
            multivals = [v for v in vals for i in range(3)]
            args = [False] * len(multivals)
            args[i:i+3] = [True] * 3
            i += 3
        else:
            args = [False] * len(vals)
            args[i] = True
            i += 1

        buttons_opts.append(
            dict(
                method='update',
                label=val,
                args=[{
                    'visible': args, #this is the key line!
                    'title': val,
                    'showlegend': False
                }]
            )
        )
    return buttons_opts

def gen_bar_graph(df, col, title, sub, num=5, avg=False, color="#d27575", w_avg='Rating'):
    """
    Produces a simple bar graph with the given dataframe and column.
    
    df: dataframe containing relevant data
    col: data to be displayed along x-axis
    """
    colors = ['#d27575', '#529b9c', '#eac392', '#9cba8f', '#675a55'] * len(df.index)
    fig = go.Figure()
    
    # do this if you want the average
    if avg:        
        fig.add_trace(
            go.Bar(
                x=df[col],
                y=df[df.columns[1]],
                name='',
                customdata = np.stack((df['Total'], [w_avg] * len(df.index)), axis=-1), #[total, '{w_avg string}']
                marker_color=colors,
                hovertemplate="<b>%{x} Books</b>: %{customdata[0]}<br><b>Avg. %{customdata[1]}</b>: %{y}",
            )
        )
        
        # below is the code for the horizontal line
        weighted_avg = np.average(df[w_avg], weights=df['Total'])
        fig.add_hline(y=weighted_avg, line_width=2, 
                      line_dash="dash", line_color="#8e7cc3",
                      annotation_text=f"Weighted Avg: {weighted_avg:.2f}",
                      annotation_position="top right",
                      annotation_bordercolor="#c7c7c7",
                      annotation_borderwidth=1,
                      annotation_borderpad=3,
                      annotation_bgcolor="#b4a7d6",
                      annotation_opacity=0.8)
    
    # do this if you just want a normal bar graph
    else:
        dfp = df.groupby(col).count().sort_values('Title', ascending=False).reset_index()[:num]
        fig.add_trace(
            go.Bar(
                x=dfp[col],
                y=dfp['Title'],
                name='',
                marker_color=color,
                hovertemplate="<b>%{x}</b>: %{y}",
            )
        )
    
    # Styling
    title = f"{title}<br><sup>{sub}"
    fig = gen_layout(fig, title, l_mar=85, r_mar=85, t_mar=120, b_mar=45, y_showgrid=True, x_showline=True)
        
    return fig.show(config=config)

def gen_hbar_graph(df, col, title, sub, num=5, avg=False, color="#d27575", w_avg='Rating'):
    """
    Produces a simple bar graph with the given dataframe and column.
    
    df: dataframe containing relevant data
    col: data to be displayed along x-axis
    """
    colors = ['#d27575', '#529b9c', '#eac392', '#9cba8f', '#675a55'] * len(df.index)
    fig = go.Figure()
    
    # do this if you want the average
    if avg:        
        fig.add_trace(
            go.Bar(
                x=df[df.columns[1]],
                y=df[col],
                name='',
                orientation='h',
                customdata = np.stack((df['Total'], [w_avg] * len(df.index)), axis=-1), #[total, '{w_avg string}']
                marker_color=colors,
                hovertemplate="<b>%{y} Books</b>: %{customdata[0]}<br><b>Avg. %{customdata[1]}</b>: %{x}",
            )
        )
        
        # below is the code for the horizontal line
        weighted_avg = np.average(df[w_avg], weights=df['Total'])
        fig.add_vline(x=weighted_avg, line_width=2, line_dash="dash", line_color="#8e7cc3",
                      annotation_text=f"Weighted Avg: {weighted_avg:.2f}",
                      annotation_position="top right",
                      annotation_bordercolor="#c7c7c7",
                      annotation_borderwidth=1,
                      annotation_borderpad=3,
                      annotation_bgcolor="#b4a7d6",
                      annotation_opacity=0.8)
    
    # do this if you just want a normal bar graph
    else:
        dfp = df.groupby(col).count().sort_values('Title', ascending=False).reset_index()[:num]
        fig.add_trace(
            go.Bar(
                x=dfp['Title'],
                y=dfp[col],
                name='',
                orientation='h',
                marker_color=color,
                hovertemplate="<b>%{x}</b>: %{y}",
            )
        )
    
    # Styling
    title = f"{title}<br><sup>{sub}"
    fig = gen_layout(fig, title, l_mar=85, r_mar=85, t_mar=120, b_mar=45, x_showgrid=True, y_showline=True)
        
    return fig.show(config=config)

def gen_stacked_bar_graph(dfp, title, sub):
    """
    Produces a stacked bar graph with the given dataframe and column.
    
    df: dataframe containing relevant data
    col: data to be displayed along x-axis
    """
    colors = ['#d27575', '#529b9c']
    
    fig = go.Figure()
    for val in dfp.columns.unique():
        fig.add_trace(
            go.Bar(
                x = dfp.index,
                y = dfp[val],
                customdata = [val] * len(dfp.index),
                marker_color = colors[1] if val=='Physical' else colors[0],
                name = str(val),
                hovertemplate="<b>%{customdata}</b>: %{y}<extra></extra>",
            )
        )
    
    # Styling
    title = f"{title}<br><sup>{sub}"
    fig = gen_layout(fig, title, l_mar=85, r_mar=85, t_mar=120, b_mar=45, y_showgrid=True, barmode="stack", x_showline=True)
    # fig.update_layout(legend = list(orientation = 'h', xanchor = "center", x = 0.5, y= 1)) )
    fig.update_layout(legend=dict(orientation='h', yanchor="top", y=0.99, xanchor="center", x=0.5))
        
    return fig.show(config=config)

def gen_heatmap(df, title, sub):
    """
    Produces a heat map with the given dataframe.
    
    df: dataframe containing relevant data
    """
    
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=df.columns,
            y=df.index,
            z=df.loc[df.index],
            # xgap=1,
            # ygap=1,
            hoverongaps=False,
            hovertemplate="<b>%{y}-%{x}</b>: %{z}<extra></extra>",
        ) 
    )
    
    title = f"{title}<br><sup>{sub}"
    fig = gen_layout(fig, title, l_mar=85, r_mar=85, t_mar=120, b_mar=45)
    fig.update_layout(margin_pad=10)
    
    return fig.show(config=config)

def top10_graph(df, col1, col2, title, sub, color="#d27575"):
    """
    Produces a simple bar graph with the given dataframe and column.
    
    df: dataframe containing relevant data
    col1: data to be displayed along x-axis
    col2: data to be displayed along y-axis
    """
    colors = ['#d27575', '#529b9c', '#eac392', '#9cba8f', '#675a55'] * len(df.index)
    ticktext = ['#' + f'{x+1}' for x in list(df.index)][::-1]
    tickvals = list(df.index)
    author = df['Author']
    genre = list(df['Genre Pair'])
    
    # ticktext = [t.replace("Why Fish Don't Exist: A Story of Loss, Love, and the Hidden Order of Life", "Why Fish Don't Exist") for t in ticktext]
    
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df[col1],
            y=df[col2],
            text=df[col2],
            name='',
            orientation='h',
            marker_color=colors,
            customdata = np.stack((genre, author), axis=-1),
            hovertemplate="<b>%{y}</b> - %{x:.1f}<br>%{customdata[0]}"
        )
    )
    
    fig.update_traces(texttemplate='<i>%{text}</i> by %{customdata[1]}  ')
    fig.update_layout(
        yaxis_ticktext=ticktext,
        yaxis_tickvals=tickvals
    )
    
    # Styling
    title = f"{title}<br><sup>{sub}"
    fig = gen_layout(fig, title, l_mar=85, r_mar=85, t_mar=120, b_mar=45, x_showgrid=True, y_showline=True)
        
    return fig.show(config=config)

def gen_time_graph(df, title, sub):
    """
    Produces a simple bar graph with the given dataframe and column.
    
    df: dataframe containing count of books finished
    """
    active = 0
    df.reset_index(inplace=True)
    rows = df.columns
    yearlabels = [x for x in df[rows[0]] if len(x)==4]
    monthlabels = [x for x in df[rows[0]] if len(x)>4]
    years = len(yearlabels) #gets number of 'whole' years (2020, 2021, 2022, etc.)
    colors = ['#d27575', '#529b9c', '#eac392', '#9cba8f', '#675a55'] * len(monthlabels)
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x = df[rows[0]][:years], 
            y = df[rows[2]][:years],
            name = '',
            marker_color = colors,
            showlegend = False,
            customdata = yearlabels,
            hovertemplate="<b>Books Finished in %{customdata}</b>: %{y}",
            visible = True if active == 0 else False
        )
    )
    
    fig.add_trace(
        go.Bar(
            x = df[rows[0]][years:], 
            y = df[rows[2]][years:],
            name = '',
            marker_color = colors,
            showlegend = False,
            customdata = monthlabels,
            hovertemplate="<b>Books Finished in %{customdata}</b>: %{y}",
            visible = True if active == 1 else False
        )
    )
    
    button_opts = []
    button_opts.append(dict(method = "update",
                            args = [{'x': [df[rows[0]][:years]],
                                     'y': [df[rows[2]][:years]],
                                     'visible':[True, False]}, 
                                     {'title.text' : f"{title}<br><sup>Total Number of Books Finished by Year"}],
                            label = 'Year'))

    button_opts.append(dict(method = "update",
                            args = [{'x': [df[rows[0]][years:]],
                                     'y': [df[rows[2]][years:]],
                                     'visible':[False, True]}, 
                                     {'title.text' : f"{title}<br><sup>Total Number of Books Finished by Year-Month"}],
                            label = 'Year-Month'))
    
    fig.update_layout(updatemenus = gen_menu(active, button_opts))
    
    # Styling
    title = button_opts[active]['args'][1]['title.text'] #searches list in dictionary in list of dictionaries?
    fig = gen_layout(fig, title, l_mar=85, r_mar=85, t_mar=120, b_mar=45, y_showgrid=True, x_showline=True)
        
    return fig.show(config=config)

def gen_scatter(df, title, sub, color="#d27575"):
    """
    Produces a simple bar graph with the given dataframe and column.
    
    df: dataframe containing relevant data
    col: data to be displayed along x-axis
    """
    color_map = {'Fiction':'#d27575',
             'Nonfiction': '#529b9c',
             'Science': '#eac392',
             'Philosophy': '#9cba8f',
             'Psychology': '#675a55'}
    
    df['Color'] = df['Genre'].map(color_map)
    names = list(df['Genre'].unique())
    
    fig = go.Figure()
    for name in names:
        dfs = df[df['Genre'] == name]
        book = dfs['Title']
        duration = dfs['Duration']
        rating = dfs['Rating']
        
        fig.add_trace(
            go.Scatter(
                x=dfs['Duration'],
                y=dfs['Rating'],
                mode='markers',
                name=name,
                marker_line_width=1,
                marker_size=12,
                marker_color=dfs['Color'],
                customdata = np.stack((book, duration, rating), axis=-1),
                hovertemplate="""<b>Title</b>: %{customdata[0]}<br><b>Duration</b>: %{customdata[1]}<br><b>Rating</b>: %{customdata[2]:.1f}<extra></extra>"""
            )
        )
    
    fig.update_layout(legend=dict(x=0.5, y=1.03, orientation='h', xanchor='center'),
                      xaxis=dict(zeroline=False))
    
    # Styling
    title = f"{title}<br><sup>{sub}"    
    fig = gen_layout(fig, title, l_mar=85, r_mar=85, t_mar=120, b_mar=65, y_showgrid=True, x_showline=True, y_showline=False, x_title="Duration (Days)", showlegend=True)
        
    return fig.show(config=config)