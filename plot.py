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
               fontcolor="#001c40", fontsize=14, hover_font_size=16):
    
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
            y=1.16,
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
    fig.update_layout(legend=dict(orientation='h', yanchor="top", y=1.01, xanchor="center", x=0.5))
        
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
        rows=3, cols=3,
        # column_widths=[0.3, 0.3, 0.3],
        row_heights=[0.2, 0.4, 0.4],
        vertical_spacing=0.1,
        # horizontal_spacing=0.06,
        specs=[[{"rowspan": 1, "colspan":3, "type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
               [{"rowspan": 2, "colspan":2, "type": "table"}, {}, {"rowspan": 1, "colspan":1, "type": "table"}],
               [{}, {}, {"rowspan": 1, "colspan":1, "type": "table"}]]
        # subplot_titles=("Total Books Read", "Total Pages Read", "Unique Authors Read")
    )
    
    years = list(df['Date'])
    active = len(years)-1

    for year in years:
        dfp = df[df['Date'] == year]

        booksPerYear = dfp.iloc[0,1]
        fig.add_trace(
            go.Indicator(
                title = {'text': "Total Books Read"},
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
                title = {'text': "Total Pages Read"},
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
                title = {'text': "Unique Authors Read"},
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
        if year == 2020:
            auths = [auths[0][0] + ' (' + str(auths[1][0]) + ')']
        else: 
            auths = [auths[0][i] + ' (' + str(auths[1][i]) + ')' for i in range(3)]

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
    
    button_opts = gen_buttons(years, num_traces=6, multi=1, no_title=1) #need the 1 to flag multi values

    fig.update_layout(
        updatemenus = gen_menu(active, button_opts),
        grid = {'rows': 3, 'columns': 3, 'pattern': "independent"},
        template = {
            'data': {'indicator': [{
                'title': {'align': 'center', 'font':{'size':25}}
                }]
            }
        },
        title_x = 0.5
    )

    # Styling
    title = f"My Reading Stats by Year"
    fig = gen_layout(fig, title, l_mar=85, r_mar=85, t_mar=120, b_mar=65, y_showgrid=True, x_showline=True, y_showline=False, x_title="Duration (Days)", showlegend=True)
     
    return fig.show()