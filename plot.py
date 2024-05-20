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

def gen_layout(fig, title, title_size=40, legendy_anchor='bottom', legendx_anchor='center', 
               height =600, plot_bg='#f0f0f0', paper_bg='#f0f0f0', 
               y_title=None, x_title=None, l_mar=45, r_mar=45, t_mar=115, b_mar=45, 
               x_showline=False, y_showline=False, linecolor='black', y_labels=True, 
               gridcolor='#cbcbcb', barmode='group', x_showgrid=False, y_showgrid=False,
               fontcolor="#001c40", fontsize=14):
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=title_size, family="Baskerville, Bold", color=fontcolor)),
        height=height,
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
            gridcolor=gridcolor
        ),
        yaxis=dict(
            showgrid=y_showgrid,
            showline=y_showline,
            showticklabels=y_labels,
            linecolor=linecolor,
            gridcolor=gridcolor
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
            y=1.1,
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

def gen_bar_graph(df, col, title, sub, num=5, color="#d27575"):
    """
    Produces a simple bar graph with the given dataframe and column.
    
    df: dataframe containing relevant data
    col: data to be displayed along x-axis
    """
    dfp = df.groupby(col).count().sort_values('Title', ascending=False).reset_index()[:num]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dfp[col],
            y=dfp['Title'],
            name='',
            marker_color=color,
            # hovertemplate="<b>%{x}</b><br>Podiums: %{y}",
        )
    )
    
    # Styling
    title = f"{title}<br><sup>{sub}"
    fig = gen_layout(fig, title, l_mar=85, r_mar=85, t_mar=120, b_mar=45, y_showgrid=True, x_showline=True)
        
    return fig.show(config=config)

def gen_stacked_bar_graph(dfp, title, sub, color="#d27575"):
    """
    Produces a stacked bar graph with the given dataframe and column.
    
    df: dataframe containing relevant data
    col: data to be displayed along x-axis
    """
    
    fig = go.Figure()
    for val in dfp.columns.unique():
        fig.add_trace(
            go.Bar(
                x = dfp.index,
                y = dfp[val],
                customdata = [val] * len(dfp.index),
                name = str(val),
                hovertemplate="<b>%{customdata}</b>: %{y}<extra></extra>",
            )    
        )
    
    # Styling
    title = f"{title}<br><sup>{sub}"
    fig = gen_layout(fig, title, l_mar=85, r_mar=85, t_mar=120, b_mar=45, y_showgrid=True, barmode="stack", x_showline=True)
        
    return fig.show(config=config)