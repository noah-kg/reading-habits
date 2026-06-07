import pandas as pd
import requests
import os
import json
import time
import numpy as np
import cufflinks as cf
import chart_studio.plotly as py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from plotly.offline import download_plotlyjs, init_notebook_mode
init_notebook_mode(connected=True)
cf.go_offline()

def load_api_key(filepath="book_creds.json"):
    """
    Checks for a GitHub environment variable first.
    Falls back to the local JSON file if running locally.
    """
    # 1. Check if running on GitHub Actions
    github_key = os.environ.get("BOOKS_API_KEY")
    if github_key:
        return github_key

    # 2. If not on GitHub, fall back to your local JSON file
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f).get("api_key")
        except Exception:
            return None
            
    return None

# Global key initialization
API_KEY = load_api_key()
# print(API_KEY)

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

##############################################################
### FUNCTIONS REGARDING PLOTLY LAYOUTS, BUTTONS, MENUS, ETC.
##############################################################

def gen_layout(fig, title='', title_size=35, autosize=True, height=600, width=1000, showlegend=False, plot_bg='#f0f0f0', 
               paper_bg='#f0f0f0', y_title=None, x_title=None, l_mar=45, r_mar=45, t_mar=115, b_mar=45, 
               x_showline=False, y_showline=False, linecolor='black', y_labels=True, 
               gridcolor='#cbcbcb', barmode='group', x_showgrid=False, y_showgrid=False, y2_showgrid=False,
               fontcolor="#001c40", fontsize=14, hover_font_size=16, zerolinewidth=1):
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=title_size, family="Baskerville, Bold", color=fontcolor)),
        autosize=autosize,
        height=height,
        width=width,
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
            zerolinewidth = zerolinewidth,
            # autorange=True
        ),
        yaxis=dict(
            showgrid=y_showgrid,
            showline=y_showline,
            showticklabels=y_labels,
            linecolor=linecolor,
            gridcolor=gridcolor,
            zerolinewidth = zerolinewidth,
            # autorange=True
        ),
        yaxis2=dict(
            showgrid=y2_showgrid,
            showline=y_showline,
            showticklabels=y_labels,
            linecolor=linecolor,
            gridcolor=gridcolor,
            zerolinewidth = zerolinewidth,
            # autorange=True
        ),
        font=dict(
            family="Baskerville",
            color=fontcolor,
            size=fontsize
        ),
        hoverlabel=dict(
            font_size=hover_font_size
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
            y=1.14,
            yanchor='top'
        )
    ]
    return updatemenus

def gen_buttons(vals, num_traces=3, multi=0, no_title=0):
    """
    Generates dropdown menu buttons.
    
    vals: list of values to turn into buttons
    """
    buttons_opts = []    
    i = 0
    for val in vals:
        if multi:
            multivals = [v for v in vals for i in range(num_traces)] #i think 3 is the number of traces you have - it can vary
            args = [False] * len(multivals)
            args[i:i+num_traces] = [True] * num_traces
            i += num_traces
        else:
            args = [False] * len(vals)
            args[i] = True
            i += 1

        if no_title:
            buttons_opts.append(
                dict(
                    method='update',
                    label=val,
                    args=[{
                        'visible': args, #this is the key line!
                        'showlegend': False
                        }]
                    )
                )
        else:
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

##############################################################
### FUNCTIONS FOR GRAPHING DATA
##############################################################

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

def gen_month_graph(df, title, sub):
    """
    Produces a simple bar graph with the given dataframe and column.
    
    df: dataframe containing relevant data
    col: data to be displayed along x-axis
    """
    color = ['#9cba8f'] * len(df.index)
    fig = go.Figure()
    
    # dfp = df.groupby(col).count().sort_values('Title', ascending=False).reset_index()[:num]
    fig.add_trace(
        go.Bar(
            x=df['Month Name'],
            y=df['Title'],
            name='',
            marker_color=color,
            hovertemplate="<b>%{x}</b>: %{y}",
        )
    )
    
    # Styling
    title = f"{title}<br><sup>{sub}"
    fig = gen_layout(fig, title, l_mar=85, r_mar=85, t_mar=120, b_mar=45, y_showgrid=True, x_showline=True)
        
    return fig.show(config=config)

def gen_bar_pie_graph(df, col, title, sub, num=5, avg=False, color="#d27575", w_avg='Rating'):
    """
    Produces a simple bar graph with the given dataframe and column.
    
    df: dataframe containing relevant data
    col: data to be displayed along x-axis
    """
    colors = ['#d27575', '#529b9c', '#eac392', '#9cba8f', '#675a55'] * len(df.index)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], specs=[[{'type':'bar'}, {'type':'pie'}]])
        
    dfp = df.groupby(col).count().sort_values('Title', ascending=False).reset_index()[:num]
    
    fig.add_trace(
        go.Bar(
            x=dfp[col],
            y=dfp['Title'],
            name='',
            marker_color=color,
            hovertemplate="<b>%{x}</b>: %{y}",
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Pie(
            labels=dfp[col],
            values=dfp['Title'],
            name='',
            marker_colors=colors,
            hovertemplate="<b>%{label}</b>: %{value}",
            hole=0.4
        ),
        row=1, col=2
    )
    
    total = dfp['Title'].sum()
    anno = f'<sup>Total: {total}'
    
    fig.add_annotation(dict(x=0.866, y=0.48,   ax=0, ay=0,
                        xref = "paper", yref = "paper", 
                        text= anno,
                        font_size=30
                      ))
    
    # Styling
    title = f"{title}<br><sup>{sub}"
    fig = gen_layout(fig, title, l_mar=85, r_mar=85, t_mar=120, b_mar=45, y_showgrid=True, x_showline=True, showlegend=False)
        
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
    fig = gen_layout(fig, title, l_mar=85, r_mar=85, t_mar=120, b_mar=45, y_showgrid=True, barmode="stack", x_showline=True, showlegend=True)
    # fig.update_layout(legend = list(orientation = 'h', xanchor = "center", x = 0.5, y= 1)) )
    fig.update_layout(legend=dict(orientation='h', yanchor="top", y=1.01, xanchor="center", x=0.5, bgcolor='rgba(0,0,0,0)'))
        
    return fig.show(config=config)

def gen_grouped_stacked_bar_graph(df, title, sub, col='Genre'):
    colors = ['#d27575', '#529b9c']
    
    fig = go.Figure()

    for gender_value, group_df in df.groupby('Gender'):
        
        # KEY FIX: Create customdata only for THIS group's rows
        # This ensures row 0 of the Male trace matches row 0 of Male customdata
        group_customdata = np.stack((group_df['Total'], group_df['Total %']), axis=-1)

        fig.add_trace(go.Bar(
            name=gender_value,
            x=group_df[col],
            y=group_df['Total'],
            marker_color=colors[1] if gender_value == 'Male' else colors[0],
            customdata=group_customdata, # Use the localized data
            offsetgroup=gender_value,
            hovertemplate=(
                f"<b>%{{x}} - {gender_value}</b>:<br>" + 
                "Books Read: %{customdata[0]}<br>" +
                "<b>%{customdata[1]}%</b> of %{x}<br>" +
                "<extra></extra>"
            )
        ))

    # Styling
    title = f"{title}<br><sup>{sub}"
    fig = gen_layout(fig, title, l_mar=85, r_mar=85, t_mar=120, b_mar=45, y_showgrid=True, barmode="stack", x_showline=True)
    fig.update_layout(barmode='group')
    fig.update_yaxes(type="log")
        
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

def gen_heatmap2(df, title, sub):
    """
    Produces a heat map with the given dataframe.
    
    df: dataframe containing relevant data
    """
    names=['All', 'No WH40k']
    active=0
    dfp = df.drop("Warhammer 40k", axis=1)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=df.columns,
            y=df.index,
            z=df.loc[df.index],
            hoverongaps=False,
            hovertemplate="<b>%{y}-%{x}</b>: %{z}<extra></extra>",
            visible=True,
            name='All'
        ) 
    )

    fig.add_trace(
        go.Heatmap(
            x=dfp.columns,
            y=dfp.index,
            z=dfp.loc[dfp.index],
            hoverongaps=False,
            hovertemplate="<b>%{y}-%{x}</b>: %{z}<extra></extra>",
            visible=False,
            name='No WH40k'
        ) 
    )
    
    button_opts = gen_buttons(names, num_traces=2, multi=0, no_title=0)
    fig.update_layout(updatemenus = gen_menu(active, button_opts))

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                showactive=True,
                x=1,
                y=1.1,
                buttons=button_opts
            )
        ]
    )
    
    title = f"{title}<br><sup>{sub}"
    fig = gen_layout(fig, title, l_mar=85, r_mar=85, t_mar=120, b_mar=45)
    fig.update_layout(margin_pad=10)
    
    return fig.show(config=config)

def top_graph(df, col1, col2, title, sub, color="#d27575"):
    """
    Produces a simple bar graph with the given dataframe and column.
    
    df: dataframe containing relevant data
    col1: data to be displayed along x-axis
    col2: data to be displayed along y-axis
    """
    # colors = ['#d27575', '#529b9c', '#eac392', '#9cba8f', '#675a55'] * len(df.index)
    ticktext = ['#' + f'{x+1}' for x in list(df.index)][::-1]
    tickvals = list(df.index)
    names = list(df['Genre'].unique())
    
    color_map = {'Fiction':'#d27575',
             'Nonfiction': '#529b9c',
             'Science': '#eac392',
             'Philosophy': '#9cba8f',
             'Psychology': '#675a55'}    
    df['Color'] = df['Genre'].map(color_map)
    
    # ticktext = [t.replace("Why Fish Don't Exist: A Story of Loss, Love, and the Hidden Order of Life", "Why Fish Don't Exist") for t in ticktext]
    
    fig = go.Figure()
    for name in names:
        dfs = df[df['Genre']==name]
        author = dfs['Author']
        genre = list(dfs['Genre Pair'])
        fig.add_trace(
            go.Bar(
                x=dfs[col1],
                y=dfs['index'],
                text=dfs[col2],
                name=name,
                orientation='h',
                marker_color=dfs['Color'],
                customdata = np.stack((genre, author), axis=-1),
                hovertemplate="<b>%{y}</b> - %{x:.1f}<br>%{customdata[0]}<extra></extra>"
            )
        )
    
    fig.update_traces(texttemplate='<i>%{text}</i> by %{customdata[1]}  ')
    fig.update_layout(
        yaxis_ticktext=ticktext,
        yaxis_tickvals=tickvals,
        legend=dict(x=0.5, y=1.05, orientation='h', xanchor='center')
    )
    
    # Styling
    title = f"{title}<br><sup>{sub}"
    fig = gen_layout(fig, title, height=800, l_mar=85, r_mar=85, t_mar=140, b_mar=45, x_showgrid=True, y_showline=True, showlegend=True)
        
    return fig.show(config=config)

def gen_time_graph(monthly_counts, title, sub):
    """
    Produces a 2-row layout chart using a flat DataFrame containing 'Year', 'Month', and 'Finished' columns.
    Top row: Total books finished by Year.
    Bottom row: Continuous Jan-Dec calendar view swapped dynamically by a Year slider.
    """
    # 1. Aggregate data for the Top Row
    yearly_data = monthly_counts.groupby('Year')['Finished'].sum().reset_index()
    available_years = sorted(monthly_counts['Year'].unique())
    initial_year = available_years[-1] if available_years else 2024
    
    # 2. Setup Subplot Layout Grid
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.16,
        subplot_titles=("Books Finished per Year", f"Books Finished per Month ({initial_year})")
    )
    
    # 3. Add Top Row Trace (Trace Index 0)
    fig.add_trace(
        go.Bar(
            x = yearly_data['Year'].astype(str),
            y = yearly_data['Finished'].tolist(),
            name = 'Yearly Total',
            marker_color = '#529b9c',
            showlegend = False,
            hovertemplate = "<b>Year %{x}</b>: %{y} books finished<extra></extra>"
        ),
        row=1, col=1
    )
    
    # 4. Set up Fixed Calendar Labels and Initial Bottom Row Trace (Trace Index 1)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    init_df = monthly_counts[monthly_counts['Year'] == initial_year]
    init_y = [init_df[init_df['Month'] == m]['Finished'].sum() for m in range(1, 13)]
    
    fig.add_trace(
        go.Bar(
            x = month_names,
            y = init_y,
            name = 'Monthly Total',
            marker_color = '#529b9c',
            showlegend = False,
            hovertemplate = "<b>%{x}</b>: %{y} books finished<extra></extra>"
        ),
        row=2, col=1
    )
    
    # 5. Build Interactive Year Slider Steps
    slider_steps = []
    for yr in available_years:
        yr_df = monthly_counts[monthly_counts['Year'] == yr]
        yr_y = [yr_df[yr_df['Month'] == m]['Finished'].sum() for m in range(1, 13)]
        
        step = dict(
            method = "update",
            args = [
                {'y': [yr_y]},
                {'annotations[1].text': f"Books Finished per Month ({yr})"},
                [1] # Target trace 1 exclusively
            ],
            label = str(yr)
        )
        slider_steps.append(step)
        
    sliders = [dict(
        active = available_years.index(initial_year),
        currentvalue = {"prefix": "Reading Year: ", "font": {"size": 14, "family": "Baskerville"}},
        pad = {"t": 50, "b": 10},
        steps = slider_steps
    )]
    
    # 6. Apply original styling adjustments
    full_title = f"{title}<br><sup>{sub}</sup>"
    fig = gen_layout(fig, title=full_title, l_mar=85, r_mar=85, t_mar=120, b_mar=100, height=850, y_showgrid=True, x_showline=True)
    
    # 7. Final layout override to attach category types and sliders
    fig.update_layout(
        xaxis = dict(type='category'),
        xaxis2 = dict(type='category'),
        sliders = sliders
    )
    
    return fig.show(config=config)

def gen_scatter(df, title, sub, color="#d27575"):
    """
    Produces a simple bar graph with the given dataframe and column.
    
    df: dataframe containing relevant data
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
        author = dfs['Author']
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
                customdata = np.stack((book, author, duration, rating), axis=-1),
                hovertemplate="""<b>Title</b>: %{customdata[0]}<br><b>Author</b>: %{customdata[1]}<br><b>Duration</b>: %{customdata[2]}<br><b>Rating</b>: %{customdata[3]:.1f}<extra></extra>"""
            )
        )
    
    fig.update_layout(legend=dict(x=0.5, y=1.03, orientation='h', xanchor='center'),
                      xaxis=dict(zeroline=False))
    
    # Styling
    title = f"{title}<br><sup>{sub}"    
    fig = gen_layout(fig, title, l_mar=85, r_mar=85, t_mar=120, b_mar=65, y_showgrid=True, x_showline=True, y_showline=False, x_title="Duration (Days)", showlegend=True)
        
    return fig.show(config=config)

def gen_infographic(df, full_df):
    fig = go.Figure()
    fig = make_subplots(
        rows=4, cols=3,
        # column_widths=[0.3, 0.3, 0.3],
        row_heights=[0.2, 0.4, 0.4, 0.3],
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
        specs=[[{"rowspan": 1, "colspan":3, "type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
               [{"rowspan": 2, "colspan":2, "type": "table"}, None, {"rowspan": 1, "colspan":1, "type": "table"}],
               [None, None, {"rowspan": 1, "colspan":1, "type": "table"}],
               [{"rowspan": 1, "colspan":2, "type": "bar"}, None, {"colspan": 1, "type": "bar"}]],
        subplot_titles=("", "", "",  "", "", "", "Sub-Genres", "Genres")
    )
    
    years = list(df['Date'])
    active = len(years)-1

    for year in years:
        dfp = df[df['Date'] == year]
        dff = full_df[full_df['Finish Date'].dt.year == year]

        booksPerYear = dfp.iloc[0,1]
        fig.add_trace(
            go.Indicator(
                title = {'text': "Total Books Read", 'font':{'size':25}},
                mode = "number",
                value = booksPerYear,
                number = {'valueformat':'f', 'font':{'size':50}},
                domain = {'row': 0, 'column': 0},
                visible = True if year == years[-1] else False
            )
        )

        pagesPerYear = dfp.iloc[0,2]
        fig.add_trace(
            go.Indicator(
                title = {'text': "Total Pages Read", 'font':{'size':25}},
                mode = "number",
                value = pagesPerYear,
                number = {'valueformat':',', 'font':{'size':50}},
                domain = {'row': 0, 'column': 1},
                visible = True if year == years[-1] else False
            )
        )

        authorsPerYear = dfp.iloc[0,3]
        fig.add_trace(
            go.Indicator(
                title = {'text': "Unique Authors Read", 'font':{'size':25}},
                mode = "number",
                value = authorsPerYear,
                number = {'valueformat':'f', 'font':{'size':50}},
                domain = {'row': 0, 'column': 2},
                visible = True if year == years[-1] else False
            )
        )

        top10 = full_df.drop(['Format', 'Duration', 'Genre Pair', 'Year', 'Start Date'], axis=1).sort_values(['Rating'])
        top10p = top10[(top10['Finish Date'].dt.year == year) & (top10['Pages'] >= 100)].tail(10)
        top10p = top10p.iloc[::-1]
        fig.add_trace(
            go.Table(
                header=dict(values=['My Highest Rated Books'],
                            align='center',
                            font_size=25,
                            height=35),
                cells=dict(values=[top10p['Title'] + ' - ' + top10p['Author']],
                           align='center',
                           fill_color='#f0f0f0',
                           font_size=18,
                           height=26),
                visible = True if year == years[-1] else False
            ),
            row=2, col=1
        )

        auths = list(zip(*dfp['Most Read Authors'].iloc[0]))

        #logic to grab < 3 authors
        if len(auths[0]) >= 3:
            mra = 3
        else:
            mra = len(auths[0])

        if year == 2020:
            auths = [auths[0][0] + ' (' + str(auths[1][0]) + ')']
        else:            
            auths = [auths[0][i] + ' (' + str(auths[1][i]) + ')' for i in range(mra)]

        fig.add_trace(
            go.Table(
                header=dict(values=['Most Read Authors'],
                            align='center',
                            font_size=25,
                            height=35),
                cells=dict(values=[auths],
                           align='center',
                           fill_color='#f0f0f0',
                           font_size=18,
                           height=26),
                visible = True if year == years[-1] else False
            ),
            row=2, col=3
        )

        pubs = list(zip(*dfp['Most Read Publishers'].iloc[0]))
        if year == 2020:
            pubs = [pubs[0][0] + ' (' + str(pubs[1][0]) + ')']
        else: 
            pubs = [pubs[0][i] + ' (' + str(pubs[1][i]) + ')' for i in range(len(pubs[0]))]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Most Read Publishers'],
                            align='center',
                            font_size=25,
                            height=35),
                cells=dict(values=[pubs],
                           align='center',
                           fill_color='#f0f0f0',
                           font_size=18,
                           height=26),
                visible = True if year == years[-1] else False
            ),
            row=3, col=3
        )

        dff1 = dff.groupby('Sub-Genre').count().sort_values('Title', ascending=False).reset_index()
        colors = ['#d27575', '#529b9c', '#eac392', '#9cba8f', '#675a55'] * len(dff1.index)
        fig.add_trace(
            go.Bar(
                x=dff1['Sub-Genre'],
                y=dff1['Title'],
                name='',
                marker_color=colors,
                # hovertemplate="<b>%{label}</b>: %{value}",
                showlegend=False,
                visible = True if year == years[-1] else False
            ),
            row=4, col=1
        )

        dff2 = dff.groupby('Genre').count().sort_values('Title', ascending=True).reset_index()
        colors = ['#d27575', '#529b9c', '#eac392', '#9cba8f', '#675a55'] * len(dff2.index)
        fig.add_trace(
            go.Bar(
                y=dff2['Genre'],
                x=dff2['Title'],
                name='',
                orientation='h',
                marker_color=colors,
                # hovertemplate="<b>%{label}</b>: %{value}",
                showlegend=False,
                visible = True if year == years[-1] else False
            ),
            row=4, col=3
        )
    
    button_opts = gen_buttons(years, num_traces=8, multi=1, no_title=1) #need the 1 to flag multi values

    fig.update_layout(
        updatemenus = gen_menu(active, button_opts),
        grid = {'rows': 4, 'columns': 3, 'pattern': "independent"},
        template = {
            'data': {'indicator': [{
                'title': {'align': 'center', 'font':{'size':25}}
                }]
            }
        },
        title_x = 0.5
    )

    fig.update_annotations(font_size=25)

    # Styling
    title = f"My Reading Stats by Year"
    fig = gen_layout(fig, title, height=800, l_mar=85, r_mar=85, t_mar=120, b_mar=65, y_showgrid=True, x_showline=True, y_showline=False, showlegend=True)
     
    return fig.show(config=config)

def gen_linegraph(df, title, sub, standard_pages=300):
    """
    Plots Actual Books Read vs. Normalized Books Read (Standard Book Equivalents)
    on a single clean Y-axis to eliminate dual-axis clutter.
    """
    df = df.copy()
    
    # Calculate Normalized Books (e.g., Total Pages / 300)
    df['Normalized Books'] = (df['Pages'] / standard_pages).round(1)
    
    fig = go.Figure()
    
    # Line 1: Actual Books Read (Count of unique Titles)
    fig.add_trace(
        go.Scatter(
            x = df.index,
            y = df['Title'],
            line_shape = 'hvh',  # Keeps your preferred step-line style
            mode = 'lines+markers',
            marker_size = 4,
            name = 'Actual Books Read',
            line_color = '#529b9c',
            hovertemplate = "<b>Actual Books</b>: %{y}<extra></extra>"
        )
    )
    
    # Line 2: Normalized Books Read
    fig.add_trace(
        go.Scatter(
            x = df.index,
            y = df['Normalized Books'],
            line_shape = 'hvh',
            mode = 'lines+markers',
            marker_size = 4,
            name = 'Normalized Books',
            line_color = '#d27575',
            hovertemplate = "<b>Normalized Books</b>: %{y}<extra></extra>"
        )
    )
    
    # Apply your template's styling rules
    full_title = f"{title}<br><sup>{sub}</sup>"
    fig = gen_layout(fig, title=full_title, l_mar=85, r_mar=85, t_mar=120, b_mar=45, y_showgrid=True, x_showline=False, showlegend=True)
    
    # Clean up the single Y-axis settings and legend
    fig.update_layout(
        hovermode = 'x unified',
        yaxis = dict(
            title_text = "Number of Books",
            showgrid = True,
            gridcolor = "lightgray",
            zeroline = True,           # Draw the line at Y = 0
            zerolinecolor = "black",   # Give it a crisp solid color
            # zerolinewidth = 1,       # Make it clean and visible
            rangemode = "tozero"       # Forces the axis scale to start cleanly at 0
        ),
        legend = dict(
            bgcolor = 'rgba(0,0,0,0)',
            orientation = 'h', 
            yanchor = "top", 
            y = 1.05, 
            xanchor = "center", 
            x = 0.5
        )
    )
    
    return fig.show(config=config)

def gen_choropleth(df, title, sub, col='Count'):
    """
    Produces a simple map showing country count
    
    df: dataframe containing relevant data
    """
    # 2. Create the Choropleth trace
    customdata = np.stack([df['Count']], axis=-1)

    fig = go.Figure(data=go.Choropleth(
        locations = df['ISO'],
        locationmode='ISO-3',
        z = df[col],
        text = df['Country'],
        customdata=customdata,
        colorscale = 'Viridis',
        autocolorscale = False,
        reversescale = False,
        marker_line_color = 'black',
        marker_line_width = 0.8,
        colorbar_title = col,
        hovertemplate = (
            "<b>%{text}</b>: " + "%{z}%<br>" +
            "Books Read: %{customdata[0]}<br>" +
            "<extra></extra>"
            )
        )
    )

    # 3. Update the layout
    fig.update_layout(
        # title_text = '2026 Global Distribution Map',
        geo = dict(
            bgcolor="#f0f0f0",
            showframe = False,
            showcoastlines = False,
            showcountries = True,
            projection_type = 'equirectangular' # Options: 'orthographic', 'mercator', etc.
        )
    )

    # Styling
    title = f"{title}<br><sup>{sub}"    
    fig = gen_layout(fig, title, l_mar=85, r_mar=85, t_mar=120, b_mar=65, y_showgrid=True, x_showline=True, y_showline=False, x_title="Duration (Days)", showlegend=True)
        
    return fig.show(config=config)

##############################################################
### FUNCTIONS TO SEARCH FOR THE BOOK COVERS FOR THE ITABLE
##############################################################

def sync_booklist_to_covers(booklist, covers):
    """
    Checks booklist.csv for new titles and adds them to new_covers.csv.
    Does not perform any API calls.
    """
    booklist = booklist.copy()
    booklist['Title'] = booklist['Title'].astype(str).str.strip()
    booklist['Author'] = booklist['Author'].astype(str).str.strip()

    # Find titles in booklist that aren't in covers yet
    new_books = booklist[~booklist['Title'].isin(covers['Title'])][['Title', 'Author']].drop_duplicates(subset=['Title'])

    if not new_books.empty:
        # Add the blank cover_url column to our new rows
        new_books['cover_url'] = np.nan
        new_books['cover_url'] = new_books['cover_url'].astype(object)
        
        # Combine the old covers with the new books dataframe
        covers = pd.concat([covers, new_books], ignore_index=True)
        
        # Save to CSV
        covers.to_csv('new_covers.csv', index=False)
        print(f"Added {len(new_books)} new title(s) to new_covers.csv (URLs are pending).")

def fetch_pending_covers():
    """
    Iterates through new_covers.csv and fetches URLs for rows with missing covers.
    """
    covers = pd.read_csv('new_covers.csv', dtype={'cover_url': object})
    pending = covers[covers['cover_url'].isna() | (covers['cover_url'] == "")]

    if pending.empty:
        return

    for index, row in pending.iterrows():
        title = row['Title']
        author = row['Author']# if 'Author' in row and pd.notna(row['Author']) else ""

        print(f"Fetching cover for: {title} by {author}...")
        
        query = f'intitle:{title} inauthor:{author}'
        # print(query)
        new_url = get_book_cover_v2(title, query)

        if new_url:
            covers.at[index, 'cover_url'] = new_url
            print(f"✅ Success: {new_url}")
        else:
            covers.at[index, 'cover_url'] = "N/A"
            print(f"⚠️ No result found for {title}.")
        
        time.sleep(1)

    covers.to_csv('new_covers.csv', index=False)
    print("Fetch process complete.")

def get_book_cover_v2(title, query):
    url = "https://www.googleapis.com/books/v1/volumes"
    
    # Base parameters
    params = {
        'q': query,
        'maxResults': 1
    }
    
    # Only append the key parameter if the key variable is not None
    if API_KEY:
        params['key'] = API_KEY

    headers = {'User-Agent': 'Mozilla/5.0'}
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers)
            # print(response.url)
            # {url}?q={query}maxResults=1 # sample of what the query looks like
            
            if response.status_code == 200:
                data = response.json()
                if data.get("totalItems", 0) > 0:
                    volume_info = data["items"][0].get("volumeInfo", {}) # Fix: added [0] index to grab first search result item
                    image_links = volume_info.get("imageLinks", {})
                    return image_links.get("thumbnail")
                return None
                
            elif response.status_code == 429:
                wait_time = int(response.headers.get("Retry-After", 2 ** (attempt + 2)))
                print(f"Rate limited. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                response.raise_for_status()
                
        except requests.exceptions.RequestException as e:
            print(f"Error making API request for '{title}': {e}")
            if attempt == max_retries - 1:
                return None
                
    return None

def show_thumb(img_url):
    img = f'<img src="{img_url}" style="max-height:120px; max-width:100px;">'
    return img
    # return f'<a href="{book_url}">{img}</a>'

def move_col(df, col, idx):
    col_series = df.pop(col)
    df = df.insert(idx, col, col_series)

##############################################################
### REWORKED GRAPHING FUNCTIONS
##############################################################

def gen_categorical_dropdown(df, columns_list=['Genre', 'Sub-Genre', 'Author', 'Format', 'Publisher', 'Gender'], top_n=10):
    """
    Creates a side-by-side layout (Bar + Donut) with a unified Dropdown Menu.
    Includes a dynamic center text annotation displaying total books per group view.
    """
    subtitle_map = {
        'Genre': 'I definitely enjoy fiction more than anything else',
        'Sub-Genre': 'Aside from WH40k, I enjoy Classics and Historial Fiction',
        'Author': 'My most-read authors (8 of them are WH40K writers!)',
        'Format': 'eBooks were massively convenient when I lived abroad',
        'Publisher': 'My most-read publishers (Black Library is WH40K)',
        'Gender': 'I tend to read more male authors, but I\'m expanding my horizons!'        
    }

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],       
        specs=[[{"type": "xy"}, {"type": "domain"}]], 
        horizontal_spacing=0.05
    )
    
    total_columns = len(columns_list)
    totals_per_column = [] # To store calculated totals for the dropdown updates
    
    for idx, col in enumerate(columns_list):
        # 1. Extract clean value counts
        counts = df[col].dropna().value_counts().reset_index()
        counts.columns = ['Category', 'Count']
        
        # Track the absolute total before we slice or cap for visualization limits
        if str(col).lower() == 'author':
            # For authors, since we drop the long tail, total equals sum of the Top 10
            col_total = counts.head(10)['Count'].sum()
        else:
            # For structural groupings, total equals the complete library volume
            col_total = counts['Count'].sum()
            
        totals_per_column.append(col_total)
        
        # 2. Clutter Filter
        if len(counts) > top_n:
            if str(col).lower() == 'author':
                counts = counts.head(top_n)
            else:
                top_m = counts.head(top_n - 1)
                leftovers_count = counts.iloc[top_n - 1:]['Count'].sum()
                other_row = pd.DataFrame([{'Category': 'Other', 'Count': leftovers_count}])
                counts = pd.concat([top_m, other_row], ignore_index=True)
            
        is_visible = (idx == 0)
        
        # Add Bar Chart Trace (Left Side)
        fig.add_trace(
            go.Bar(
                x = counts['Category'],
                y = counts['Count'],
                name = col,
                marker_color = '#529b9c',
                visible = is_visible,
                hovertemplate = "<b>%{x}</b><br>Volume: %{y} books<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Add Donut Chart Trace (Right Side -> hole set to 0.5)
        fig.add_trace(
            go.Pie(
                labels = counts['Category'],
                values = counts['Count'],
                name = col,
                visible = is_visible,
                textinfo = 'percent',          
                textposition = 'inside',
                hole = 0.3,                    # Expanded hole width for comfortable text housing
                marker = dict(colors=['#529b9c', '#d27575', '#9cba8f', '#eac435', '#a6b1e1', '#7f7f7f']),
                hovertemplate = "<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>"
            ),
            row=1, col=2
        )
        
    # 3. Create the Initial Center Annotation
    # Column 2 maps roughly to x-domain coordinates [0.55 to 1.0]. Center is around x=0.80
    center_annotation = dict(
        text = f"<b>{totals_per_column[0]}</b><br>Books",
        font = dict(family="Baskerville", size=18, color="#001c40"),
        showarrow = False,
        x = 0.838, y = 0.5, # Locks text exactly in the middle of the right column domain
        xref = "paper", yref = "paper"
    )
    
    # 4. Generate Dropdown Buttons
    buttons = []
    for idx, col in enumerate(columns_list):
        visibility = [False] * (total_columns * 2)
        visibility[idx * 2] = True       
        visibility[(idx * 2) + 1] = True 
        
        # Fetch subtitle from our map, fallback to a default if column isn't found
        current_sub = subtitle_map.get(col, f"Library distribution broken down by {col}")
        
        button = dict(
            method = "update",
            label = f"Group By: {col}",
            args = [
                {"visible": visibility}, 
                {
                    # Dynamically inject the personalized subtitle string here
                    "title.text": f"Library Composition by {col}<br><sup>{current_sub}</sup>",
                    "annotations[2].text": f"<b>{totals_per_column[idx]}</b><br>Books"
                } 
            ]
        )
        buttons.append(button)
        
    # 5. Apply everything to Layout
    fig.update_layout(
        updatemenus=[dict(
            active = 0,
            buttons = buttons,
            x = 1.0, y = 1.15, 
            xanchor = 'right', yanchor = 'top',
            font = dict(family="Baskerville", size=13)
        )],
        legend = dict(
            orientation = "h",
            yanchor = "top", y = -0.05,
            xanchor = "center", x = 0.5
        )
    )
    
    # 4. Generate Initial Static Title
    first_col = columns_list[0]
    initial_sub = subtitle_map.get(first_col, f"Library distribution broken down by {first_col}")
    initial_title = f"Library Composition by {first_col}<br><sup>{initial_sub}</sup>"

    # Style via your core notebook parameters
    fig = gen_layout(fig, title=initial_title, height=600, t_mar=130, l_mar=85, r_mar=85, b_mar=65, y_showgrid=True, x_showline=True)
    
    # Append our custom donut annotation onto the existing list created by gen_layout
    fig.add_annotation(center_annotation)
    
    return fig.show(config=config)

def gen_genre_sunburst(df):
    """
    Creates an interactive nested Sunburst chart showing the hierarchy 
    of Category -> Genre -> Sub-Genre.
    """
    # 1. Clean data: Drop rows that are missing any part of the hierarchy path
    # If a book has a Sub-Genre but no main Genre, Plotly's path logic will break.
    hierarchy_cols = ['Format', 'Genre', 'Sub-Genre']
    df_clean = df.dropna(subset=hierarchy_cols)
    
    # 2. Generate the Sunburst plot using Plotly Express
    fig = px.sunburst(
        df_clean, 
        path = hierarchy_cols, # Dictates the ring order from inside out
        color_discrete_sequence = ['#529b9c', '#d27575', '#9cba8f', '#eac435', '#a6b1e1']
    )
    
    # 3. Optimize the hover text and slice labels
    fig.update_traces(
        textinfo = "label+percent parent", # Displays the slice name and its % share of the parent ring
        hovertemplate = "<b>%{label}</b><br>Books: %{value}<br>Share of Parent: %{percentParent:.1%}<extra></extra>"
    )
    
    # 4. Bind it to your global notebook layout engine
    title = "Library Composition Hierarchy<br><sup>Click individual inner segments to drill down into deeper sub-genres</sup>"
    fig = gen_layout(fig, title=title, height=700, t_mar=130, b_mar=50)
    
    return fig.show(config=config)

def gen_genre_sunburst2(df):
    """
    Creates an interactive nested Sunburst chart using pure plotly.graph_objects.
    Dynamically maps hierarchies across Format -> Genre -> Sub-Genre.
    """
    hierarchy_cols = ['Format', 'Genre', 'Sub-Genre']
    df_clean = df.dropna(subset=hierarchy_cols)
    
    # --- STEP 1: TRANSLATE DATA FOR GRAPH OBJECTS ---
    # go.Sunburst needs a single list of unique labels, and a matching list of parents.
    labels = []
    parents = []
    values = []
    
    # Level 1: Root nodes (Formats)
    formats = df_clean['Format'].value_counts()
    for fmt, count in formats.items():
        labels.append(fmt)
        parents.append("") # Root elements have no parent
        values.append(count)
        
    # Level 2: Middle nodes (Format -> Genre)
    fmt_genre = df_clean.groupby(['Format', 'Genre']).size()
    for (fmt, genre), count in fmt_genre.items():
        # We use a unique ID "Format - Genre" to prevent duplicate names from colliding
        labels.append(f"{fmt} - {genre}") 
        parents.append(fmt)
        values.append(count)
        
    # Level 3: Outer nodes (Format -> Genre -> Sub-Genre)
    # We use tracking IDs to make sure the outer sub-genres nest under the right parent path
    full_path = df_clean.groupby(['Format', 'Genre', 'Sub-Genre']).size()
    for (fmt, genre, sub_genre), count in full_path.items():
        labels.append(f"{fmt} - {genre} - {sub_genre}")
        parents.append(f"{fmt} - {genre}")
        values.append(count)

    # --- STEP 2: BUILD THE GRAPH OBJECT TRACE ---
    fig = go.Figure()
    
    fig.add_trace(go.Sunburst(
        ids = labels, # Unique identification string for each slice
        # Clean up the display labels so the user only sees "Sci-Fi" instead of "Audiobook - Sci-Fi"
        labels = [x.split(" - ")[-1] for x in labels], 
        parents = parents,
        values = values,
        branchvalues = "total", # Ensures parent slices cleanly equal the sum of their children
        textinfo = "label+percent parent",
        hovertemplate = "<b>%{label}</b><br>Books: %{value}<br>Share of Parent: %{percentParent:.1%}<extra></extra>",
        marker = dict(colors=['#529b9c', '#d27575', '#9cba8f', '#eac435', '#a6b1e1'])
    ))
    
    # --- STEP 3: APPLY STANDARD TEMPLATE LAYOUT ---
    title = "Library Composition Hierarchy - WIP<br><sup>Click an inner segment to explore deeper within it</sup>"
    fig = gen_layout(fig, title=title, height=700, t_mar=130, b_mar=50)
    
    return fig.show(config=config)

def gen_reading_heatmap(df, date_col='Finish Date'):
    """
    Creates a density heatmap grid (Day of Week vs. Month) 
    showing completion velocity using pure plotly.graph_objects.
    """
    # 1. Ensure the date column is parsed and filter out missing values
    df_dates = df[df[date_col].notnull()].copy()
    df_dates[date_col] = pd.to_datetime(df_dates[date_col])
    
    # Extract shorthand string names
    df_dates['Month'] = df_dates[date_col].dt.strftime('%b')
    df_dates['Day'] = df_dates[date_col].dt.strftime('%a')
    
    # 2. Establish strict calendar sorting rules
    months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    days_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    # 3. Group and pivot data into a literal 2D matrix layout
    grid_data = df_dates.groupby(['Month', 'Day']).size().reset_index(name='Counts')
    matrix = grid_data.pivot(index='Day', columns='Month', values='Counts').fillna(0)
    
    # Reindex forces the matrix rows/columns to align to our strict calendar order
    matrix = matrix.reindex(index=days_order, columns=months_order).fillna(0)
    
    # 4. Generate the Heatmap Trace
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z = matrix.values,     # The raw density integer array
        x = matrix.columns,    # Months on the horizontal axis
        y = matrix.index,      # Days of the week on the vertical axis
        
        # Color gradient: Shifts from a soft neutral light gray (0) to your main theme teal (1)
        colorscale = [[0, '#f8f9fa'], [1, '#529b9c']], 
        showscale = True,
        
        # Clear, interactive custom hover tooltips
        hovertemplate = "<b>%{y}s in %{x}</b><br>Books Finished: %{z}<extra></extra>"
    ))
    
    # 5. Connect to your core layout pipeline
    title = "My Reading DNA<br><sup>Total book completions cross-referenced by Day of Week vs. Month</sup>"
    fig = gen_layout(fig, title=title, height=500, l_mar=70, r_mar=40, t_mar=100, b_mar=60)
    
    # Reverse the Y-axis so Monday starts cleanly at the top and Sunday rests at the bottom
    fig.update_layout(yaxis=dict(autorange="reversed"))
    
    return fig.show(config=config)

def gen_reading_intensity_heatmap(df, start_col='Start Date', finish_col='Finish Date', pages_col='Pages'):
    """
    Creates a density heatmap grid (Day of Week vs. Month) showing daily page volume.
    Distributes book pages across reading durations and silences empty hover gaps.
    """
    # 1. Clean data and ensure proper datetime parsing
    df_clean = df.dropna(subset=[start_col, finish_col, pages_col]).copy()
    df_clean[start_col] = pd.to_datetime(df_clean[start_col])
    df_clean[finish_col] = pd.to_datetime(df_clean[finish_col])
    
    # 2. MATH PIPELINE: Explode books day-by-day across their reading duration
    day_records = []
    for _, row in df_clean.iterrows():
        date_range = pd.date_range(start=row[start_col], end=row[finish_col])
        days_count = len(date_range)
        if days_count == 0:
            continue
            
        pages_per_day = row[pages_col] / days_count
        for single_date in date_range:
            day_records.append({'Date': single_date, 'Pages': pages_per_day})
            
    df_daily = pd.DataFrame(day_records)
    if df_daily.empty:
        print("No valid reading duration data found.")
        return
        
    # Sum pages for overlapping books read on the same calendar day
    df_daily_totals = df_daily.groupby('Date')['Pages'].sum().reset_index()
    
    # 3. EXTRACTION: Pull chronological string tokens from the dates
    df_daily_totals['Month'] = df_daily_totals['Date'].dt.strftime('%b')
    df_daily_totals['Day'] = df_daily_totals['Date'].dt.strftime('%a')
    
    # Calculate the average pages read for each Day of Week + Month combination
    # (Using average instead of raw sum prevents older years from completely dominating newer ones)
    grid_data = df_daily_totals.groupby(['Month', 'Day'])['Pages'].mean().reset_index(name='AvgPages')
    
    # 4. MATRIX PIPELINE: Pivot and sort into calendar grid order
    months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    days_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    matrix = grid_data.pivot(index='Day', columns='Month', values='AvgPages')
    matrix = matrix.reindex(index=days_order, columns=months_order)
    
    # 5. GENERATE THE HEATMAP TRACE
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z = matrix.values,     # The average daily pages array
        x = matrix.columns,    # Months on the horizontal axis
        y = matrix.index,      # Days of the week on the vertical axis
        
        # Color gradient: Neutral off-white up to your primary theme teal
        colorscale = [[0, '#f8f9fa'], [1, '#529b9c']], 
        
        # Turn off tooltips for cells with NaN (gaps where you have no historical reading)
        hoverongaps = False,
        
        showscale = True,
        colorbar = dict(title="Avg Pages"),
        
        hovertemplate = (
            "<b>%{y}s in %{x}</b><br>"
            "Pace: ~%{z:.1f} Pages/Day<extra></extra>"
        )
    ))
    
    # 6. LAYOUT PIPELINE
    title = "My Weekly Reading DNA<br><sup>Average daily reading volume cross-referenced by Day of Week vs. Month</sup>"
    fig = gen_layout(fig, title=title, height=450, l_mar=70, r_mar=40, t_mar=100, b_mar=60)
    
    # Reverse the Y-axis so Monday starts at the top and Sunday rests at the bottom
    fig.update_layout(yaxis=dict(autorange="reversed", showgrid=False, zeroline=False))
    
    return fig.show(config=config)

def gen_reading_calendar_waffle(df, start_col='Start Date', finish_col='Finish Date', pages_col='Pages', title_col='Title', genre_col='Sub-Genre', manga_genres=['Graphic Novel', 'Manga']):
    """
    Creates an annual calendar waffle chart (12 Months x 31 Days) using go.Heatmap.
    Uses hoverongaps=False to natively silence hover text on all inactive days.
    """
    # 1. Clean data and ensure proper datetime parsing
    df_clean = df.dropna(subset=[start_col, finish_col, pages_col, title_col]).copy()
    df_clean[start_col] = pd.to_datetime(df_clean[start_col])
    df_clean[finish_col] = pd.to_datetime(df_clean[finish_col])
    
    # 2. MATH PIPELINE: Explode books day-by-day
    day_records = []
    manga_genres_lower = [g.lower() for g in manga_genres]
    
    for _, row in df_clean.iterrows():
        date_range = pd.date_range(start=row[start_col], end=row[finish_col])
        days_count = len(date_range)
        if days_count == 0:
            continue
            
        pages_per_day = row[pages_col] / days_count
        is_manga = str(row.get(genre_col, '')).lower() in manga_genres_lower
        
        for single_date in date_range:
            day_records.append({
                'Date': single_date, 
                'Pages': pages_per_day,
                'Title': str(row[title_col]),
                'IsManga': is_manga
            })
            
    df_daily = pd.DataFrame(day_records)
    if df_daily.empty:
        print("No valid reading data found.")
        return
        
    def combine_day(group):
        titles = " | ".join(group['Title'].unique())
        total_pages = group['Pages'].sum()
        manga_pages = group[group['IsManga']]['Pages'].sum()
        prose_pages = group[~group['IsManga']]['Pages'].sum()
        return pd.Series({'Pages': total_pages, 'Titles': titles, 'IsMangaDay': manga_pages > prose_pages})

    df_daily_totals = df_daily.groupby('Date').apply(combine_day).reset_index()
    df_daily_totals['Year'] = df_daily_totals['Date'].dt.year
    unique_years = sorted(df_daily_totals['Year'].unique())
    
    prose_max_ceiling = 60.0
    months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # 3. BUILD THE DISCRETE COLORSCALE ENGINE
    # We change the 0 index from off-white to transparent/background color since it represents gaps now
    custom_colorscale = [[0.0, '#f8f9fa']] 
    rgba_bg = np.array(mcolors.to_rgba('#f8f9fa'))
    rgba_tl = np.array(mcolors.to_rgba('#529b9c'))
    
    for i in range(1, 101):
        pct = i / 100.0
        mixed_color = mcolors.to_hex(rgba_bg * (1 - pct) + rgba_tl * pct)
        custom_colorscale.append([i / 102.0, mixed_color])
        
    custom_colorscale.append([101 / 102.0, '#d27575'])
    custom_colorscale.append([102 / 102.0, '#d27575'])

    fig = go.Figure()
    
    # 4. TRACE LOOP: Generate grids for each unique year
    for idx, year in enumerate(unique_years):
        df_year = df_daily_totals[df_daily_totals['Year'] == year]
        
        # Initialize the baseline matrix as NaNs (gaps) instead of zeros
        matrix_z = np.full((12, 31), np.nan)
        matrix_titles = np.full((12, 31), '', dtype=object) 
        matrix_hover_pages = np.zeros((12, 31)) 
        
        # Pin down calendar days that flat out don't exist
        invalid_days_mask = np.zeros((12, 31), dtype=bool)
        for m_idx, month in enumerate(months_order):
            for d_idx in range(1, 32):
                try:
                    pd.to_datetime(f"{year}-{m_idx+1}-{d_idx}")
                except:
                    invalid_days_mask[m_idx, d_idx-1] = True

        for _, row in df_year.iterrows():
            m_name = row['Date'].strftime('%b')
            day_num = row['Date'].day
            if m_name in months_order and 1 <= day_num <= 31:
                m_idx = months_order.index(m_name)
                d_idx = day_num - 1
                
                pages = row['Pages']
                matrix_hover_pages[m_idx, d_idx] = pages
                matrix_titles[m_idx, d_idx] = row['Titles']
                
                if row['IsMangaDay']:
                    matrix_z[m_idx, d_idx] = 101 
                else:
                    pct = min(pages / prose_max_ceiling, 1.0)
                    matrix_z[m_idx, d_idx] = max(int(pct * 100), 1)

        # Force invalid calendar days to be None so they hide completely, 
        # while keeping unread active days as NaN so they display as soft gray boxes
        matrix_z = matrix_z.astype(object)
        matrix_z[invalid_days_mask] = None
        
        custom_data_pack = np.stack((matrix_titles, matrix_hover_pages), axis=-1)
        is_visible = (idx == len(unique_years) - 1)
        
        fig.add_trace(go.Heatmap(
            z = matrix_z,
            x = [f"{d}" for d in range(1, 32)],
            y = months_order,
            xgap = 3, ygap = 3,
            colorscale = custom_colorscale,
            zmin = 0, zmax = 102,
            
            # --- CRITICAL HOVER FIXES ---
            hoverongaps = False, # This natively kills the hover tooltip window on all NaN boxes!
            
            showscale = False,
            
            customdata = custom_data_pack,
            visible = is_visible,
            # Back to a strict single-string template structure that Plotly won't choke on
            hovertemplate = (
                "<b>%{y} %{x}, " + str(year) + "</b><br>"
                "Books: <i>%{customdata[0]}</i><br>"
                "Estimated Daily Total: %{customdata[1]:.1f} Pages<extra></extra>"
            )
        ))

    # 5. TIMELINE SLIDER GENERATION
    steps = []
    for idx, year in enumerate(unique_years):
        visibility = [False] * len(unique_years)
        visibility[idx] = True
        
        step = dict(
            method = "update",
            label = str(year),
            args = [
                {"visible": visibility}, 
                {"title.text": f"My Reading Blueprint ({year})<br><sup>Teal indicates prose intensity. Coral highlights Graphic Novels (they skew the data!)</sup>"} 
            ]
        )
        steps.append(step)
        
    sliders = [dict(
        active = len(unique_years) - 1, 
        currentvalue = {"prefix": "Reading Year: ", "font": {"family": "Baskerville", "size": 15}},
        pad = {"t": 20, "b": 30}, 
        yanchor = "top",          
        y = -0.05,                
        steps = steps
    )]
    
    # 6. CORE LAYOUT
    latest_year = unique_years[-1]
    initial_title = f"My Reading Blueprint ({latest_year})<br><sup>Teal indicates prose intensity. Coral highlights Graphic Novels (they skew the data!)</sup>"
    
    fig = gen_layout(fig, title=initial_title, height=520, t_mar=100, b_mar=140, l_mar=60, r_mar=40)
    fig.update_layout(
        sliders = sliders,
        yaxis = dict(autorange="reversed", showgrid=False, zeroline=False),
        xaxis = dict(showgrid=False, zeroline=False, tickmode="linear", dtick=5)
    )
    
    return fig.show(config=config)

def gen_reading_burnup(df, start_col='Start Date', finish_col='Finish Date', pages_col='Pages', daily_target_goal=30):
    """
    Generates a yearly Page Accumulation Burn-Up Chart using go.Scatter.
    Plots cumulative actual pages read against a linear target trajectory line.
    """
    # 1. Clean data and ensure proper datetime parsing
    df_clean = df.dropna(subset=[start_col, finish_col, pages_col]).copy()
    df_clean[start_col] = pd.to_datetime(df_clean[start_col])
    df_clean[finish_col] = pd.to_datetime(df_clean[finish_col])
    
    # 2. MATH PIPELINE: Explode books day-by-day
    day_records = []
    for _, row in df_clean.iterrows():
        date_range = pd.date_range(start=row[start_col], end=row[finish_col])
        days_count = len(date_range)
        if days_count == 0:
            continue
            
        pages_per_day = row[pages_col] / days_count
        for single_date in date_range:
            day_records.append({'Date': single_date, 'Pages': pages_per_day})
            
    df_daily = pd.DataFrame(day_records)
    if df_daily.empty:
        print("No valid reading data found.")
        return
        
    # Group by date to handle your rare "finish morning / start evening" double-book days
    df_daily_totals = df_daily.groupby('Date')['Pages'].sum().reset_index()
    
    # 3. CHRONOLOGICAL TIMELINE ALIGNMENT
    # Filter dataset down exclusively to the current calendar year
    current_year = 2026
    df_year = df_daily_totals[df_daily_totals['Date'].dt.year == current_year].copy()
    
    # Create a continuous baseline index from Jan 1st to Dec 31st of the current year
    full_year_range = pd.date_range(start=f"{current_year}-01-01", end=f"{current_year}-12-31")
    df_timeline = pd.DataFrame({'Date': full_year_range})
    
    # Merge your reading tracking data into the master calendar timeline
    df_timeline = pd.merge(df_timeline, df_year, on='Date', how='left').fillna(0)
    
    # Run the cumulative sum to generate the climbing velocity metrics
    df_timeline['CumulativePages'] = df_timeline['Pages'].cumsum()
    
    # 4. TARGET TRAJECTORY LINE LOGIC
    # Calculates a perfect diagonal target line: Day 1 (0 pages) up to Day 365 (365 * daily_target_goal)
    day_indices = np.arange(len(full_year_range))
    df_timeline['TargetTrajectory'] = day_indices * daily_target_goal
    
    # 5. GENERATE PLOTLY GRAPH OBJECTS TRACES
    fig = go.Figure()
    
    # Trace A: The Target Baseline (Dashed neutral line)
    fig.add_trace(go.Scatter(
        x = df_timeline['Date'],
        y = df_timeline['TargetTrajectory'],
        mode = 'lines',
        name = f'Target Baseline ({daily_target_goal} pgs/day)',
        line = dict(color='#cbd5e1', width=2, dash='dash'),
        hovertemplate = "Target Cumulative Total: %{y:,.0f} Pages<extra></extra>"
    ))
    
    # Trace B: Your Actual Reading Progress (Solid core theme color)
    fig.add_trace(go.Scatter(
        x = df_timeline['Date'],
        y = df_timeline['CumulativePages'],
        mode = 'lines',
        name = 'My Actual Progress',
        line = dict(color='#529b9c', width=3.5),
        fill = 'tozeroy', # Shades the area underneath the line to give it a solid "mountain" feel
        fillcolor = 'rgba(82, 155, 156, 0.06)', 
        hovertemplate = (
            "<b>%{x|%b %d, %Y}</b><br>"
            "Total Pages Read: <b>%{y:,.0f}</b><extra></extra>"
        )
    ))
    
    # 6. LAYOUT CONFIGURATION
    title = f"2026 Reading Burn-Up Campaign<br><sup>Tracking real-time cumulative page momentum against a yearly baseline trajectory</sup>"
    fig = gen_layout(fig, title=title, width=950, height=500, t_mar=100, b_mar=60, l_mar=70, r_mar=40)
    
    fig.update_layout(
        hovermode = "x unified", # Triggers both the target and actual popups simultaneously on vertical crosshairs
        legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis = dict(showgrid=True, gridcolor='#f1f5f9', tickformat="%b"),
        yaxis = dict(showgrid=True, gridcolor='#f1f5f9', title="Cumulative Pages Consumed")
    )
    
    return fig.show(config=config)

def gen_rating_thickness_scatter(df, rating_col='Rating', pages_col='Pages', title_col='Title', format_col='Format'):
    """
    Creates a 'Book Thickness' vs. Rating Scatter Plot with vertical jittering.
    Color-codes by reading format and preserves absolute ratings in the hovercards.
    """
    # 1. Clean data and drop missing value rows
    df_clean = df.dropna(subset=[rating_col, pages_col, title_col]).copy()
    
    # 2. Inject a tight vertical noise offset to spread out clusters on the 1-5 axis
    # We set a random seed so the data configuration remains stable across refreshes
    np.random.seed(42)
    jitter_range = 0.14
    df_clean['JitteredRating'] = df_clean[rating_col] + np.random.uniform(-jitter_range, jitter_range, size=len(df_clean))
    
    fig = go.Figure()
    
    # 3. TRACE LOOP: Split data by Format to enable automatic color-coding and legend filtering
    if format_col in df_clean.columns:
        unique_formats = sorted(df_clean[format_col].dropna().unique())
    else:
        unique_formats = ['All Books']
        
    # Cohesive theme palette matching your brand teal, a complementary warm coral, and deep slate
    theme_palette = ['#529b9c', '#d27575', '#475569', '#94a3b8']
    
    for idx, fmt in enumerate(unique_formats):
        if format_col in df_clean.columns:
            df_fmt = df_clean[df_clean[format_col] == fmt]
        else:
            df_fmt = df_clean
            
        color = theme_palette[idx % len(theme_palette)]
        
        fig.add_trace(go.Scatter(
            x = df_fmt[pages_col],
            y = df_fmt['JitteredRating'],
            mode = 'markers',
            name = str(fmt),
            
            # Using semi-transparent markers (opacity=0.7) means heavily overlapping 
            # regions naturally darken, showing you exactly where your rating "sweet spots" sit.
            marker = dict(
                size = 11,
                color = color,
                opacity = 0.7,
                line = dict(width=0.5, color='#ffffff')
            ),
            
            text = df_fmt[title_col],
            # customdata passes the true, unjittered integer rating straight to the hover template
            customdata = df_fmt[rating_col], 
            
            hovertemplate = (
                "<b>%{text}</b><br>"
                "Format: " + str(fmt) + "<br>"
                "Thickness: %{x} Pages<br>"
                "True Rating: <b>%{customdata} Stars</b><extra></extra>"
            )
        ))
        
    # 4. CORE LAYOUT PIPELINE
    title = "Book Thickness vs. Score Evaluation<br><sup>Analyzing page count influence across media formats (vertical jitter applied)</sup>"
    fig = gen_layout(fig, title=title, width=950, height=520, t_mar=100, b_mar=60, l_mar=90, r_mar=40)
    
    # 5. FIXED COORD GRID: Overwrite the Y-axis to frame the integer steps perfectly
    fig.update_layout(
        legend = dict(title="Reading Format", orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        xaxis = dict(
            title = "Book Thickness (Page Count)",
            showgrid = True,
            gridcolor = '#f1f5f9',
            zeroline = False
        ),
        yaxis = dict(
            title = "Assigned Rating Score",
            tickmode = 'array',
            # We explicitly target the clean 1-5 boundaries
            # tickvals = [1, 2, 3, 4, 5],
            # ticktext = ['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars'],
            # Hard bounds set between 0.5 and 5.5 give the jitter clouds room to breathe
            range = [0.5, 10.5], 
            showgrid = True,
            gridcolor = '#e2e8f0',
            zeroline = False
        )
    )
    
    return fig.show(config=config) 
