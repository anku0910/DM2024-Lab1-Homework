""" Utility function for doing analysis on emotion datasets """
from collections import Counter, OrderedDict
import plotly.graph_objs as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
import math

def get_tokens_and_frequency(token_list):
    """obtain word frequecy from pandas dataframe column of lists"""
    counter = Counter(token_list)
    counter = OrderedDict(counter.most_common()) # sort by value
    tokens = counter.keys()
    tokens_count = counter.values()

    return tokens, tokens_count

def compute_frequencies(train_data, emotion, feature, frequency=True):
    """ compute word frequency for pandas datafram column of lists"""
    tokens =  train_data.loc[(train_data["emotions"] == emotion)][feature].values.tolist()
    tokens = [item for l in tokens for item in l]
    if frequency:
        return get_tokens_and_frequency(tokens)
    else:
        return tokens

###################################
""" Visualizing Functions """
###################################
def plot_word_frequency(word_list, plot_title):
    trace1 = {
        "x": list(word_list[0]),
        "y": list(word_list[1]),
        "type": "bar"
    }

    data = go.Data([trace1])

    layout = {
        "title": plot_title,
        "yaxis": {"title": "Frequency"}
    }

    fig = go.Figure(data = data, layout=layout)
    return fig

def plot_heat_map(plot_x, plot_y, plot_z):
    """ Helper to plot heat map """
    trace = {
        "x": plot_x,
        "y": plot_y,
        "z": plot_z,
        "colorscale": [[0.0, "rgb(158,1,66)"], [0.1, "rgb(213,62,79)"], [0.2, "rgb(244,109,67)"], [0.3, "rgb(253,174,97)"], [0.4, "rgb(254,224,139)"], [0.5, "rgb(255,255,191)"], [0.6, "rgb(230,245,152)"], [0.7, "rgb(171,221,164)"], [0.8, "rgb(102,194,165)"], [0.9, "rgb(50,136,189)"], [1.0, "rgb(94,79,162)"]],
        "type": "heatmap"
    }

    data = go.Data([trace])
    layout = {
        "legend": {
            "bgcolor": "#F5F6F9",
            "font": {"color": "#4D5663"}
        },
        "paper_bgcolor": "#F5F6F9",
        "plot_bgcolor": "#F5F6F9",
        "xaxis1": {
            "gridcolor": "#E1E5ED",
            "tickfont": {"color": "#4D5663"},
            "title": "",
            "titlefont": {"color": "#4D5663"},
            "zerolinecolor": "#E1E5ED"
        },
        "yaxis1": {
            "gridcolor": "#E1E5ED",
            "tickfont": {"color": "#4D5663"},
            "title": "",
            "titlefont": {"color": "#4D5663"},
            "zeroline": False,
            "zerolinecolor": "#E1E5ED"
        }
    }

    fig = go.Figure(data = data, layout=layout)
    return fig

def get_trace(X_pca, data, category, color):
    """ Build trace for plotly chart based on category """
    trace = go.Scatter3d(
        x=X_pca[data.apply(lambda x: True if x==category else False), 0],
        y=X_pca[data.apply(lambda x: True if x==category else False),1],
        z=X_pca[data.apply(lambda x: True if x==category else False),2],
        mode='markers',
        marker=dict(
            size=4,
            line=dict(
                color=color,
                width=0.2
            ),
            opacity=0.8
        ),
        text=data[data.apply(lambda x: True if x==category else False).tolist()]
    )
    return trace

def plot_word_cloud(text):
    """ Generate word cloud given some input text doc """
    word_cloud = WordCloud().generate(text)
    plt.figure(figsize=(8,6), dpi=90)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    
########################################
""" My Visualizing Functions """
########################################
def plot_category_distribution(df_original, df_resampled):
    cat_count_origin = df_original['category_name'].value_counts().tolist()
    cat_count_sample = df_resampled['category_name'].value_counts().tolist()
    
    labels = df_original['category_name'].unique()
    index = np.arange(len(labels)*2)
    ratings = cat_count_origin
    plt.bar(index[0::2], cat_count_origin, label="original")
    plt.bar(index[1::2], cat_count_sample, label="sample", color="r")
    plt.legend()
    sticks_pos = (index[0::2] + index[1::2])/2
    plt.xticks(sticks_pos, labels, rotation=-45)
    plt.ylabel("number of texts")
    plt.title("Category distribution")
    # Show the plot
    plt.show()

def plot_paginated_heatmap(tdm_df, rows_per_page=50, cols_per_page=50):        
    # Calculate the total number of row and column pages
    total_row_pages = (tdm_df.shape[0] + rows_per_page - 1) // rows_per_page
    total_col_pages = (tdm_df.shape[1] + cols_per_page - 1) // cols_per_page
    
    # Function to create heatmap based on page number
    def create_paginated_heatmap(row_page=1, col_page=1):
        # Calculate start and end indices for rows and columns based on the page number
        row_start = (row_page - 1) * rows_per_page
        row_end = min(row_start + rows_per_page, tdm_df.shape[0])
        col_start = (col_page - 1) * cols_per_page
        col_end = min(col_start + cols_per_page, tdm_df.shape[1])
        
        # Select the subset of the DataFrame for the given page
        sub_df = tdm_df.iloc[row_start:row_end, col_start:col_end]
    
        # Use Plotly to create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=sub_df.values,
            x=sub_df.columns,
            y=sub_df.index,
            colorscale='PuRd',
            zmin=0, zmax=1,
            showscale=True
        ))
    
        # Adjust the layout to match your desired figure size
        fig.update_layout(
            width=900,  # Adjust the width
            height=700,  # Adjust the height
            # title=f'Heatmap - Document Page {row_page}, Term Page {col_page}',
            xaxis=dict(tickmode='array', tickvals=sub_df.columns),
            yaxis=dict(tickmode='array', tickvals=sub_df.index)
        )
    
        # Display the interactive heatmap
        fig.show(config={
            'displayModeBar': False,       # Hide the mode bar
            'scrollZoom': False,           # Disable zoom with scroll
            'displaylogo': False,          # Hide the Plotly logo in the mode bar
            'staticPlot': False,           # Set to True to make it non-interactive
        })
    
    # Create slider widgets for row and column page control
    row_slider = widgets.IntSlider(value=1, min=1, max=total_row_pages, step=1,
                                   description='Document Page:',    # Label for the slider
                                   style={'description_width': '120px'},  # Adjust the description width
                                   layout=widgets.Layout(
                                       width='300px',       # Adjust the width
                                       height='20px',       # Adjust the height
                                       margin='10px 20px',  # Adjust the margins (top-bottom left-right)
                                       align_self='center'  # Center alignment (can be 'flex-start', 'center', 'flex-end')
                                   )
                                  )
    col_slider = widgets.IntSlider(value=1, min=1, max=total_col_pages, step=1,
                                   description='Term Page:',
                                   style={'description_width': '120px'},  # Adjust the description width
                                   layout=widgets.Layout(
                                       width='300px',       # Adjust the width
                                       height='20px',       # Adjust the height
                                       margin='10px 20px',  # Adjust the margins (top-bottom left-right)
                                       align_self='center'  # Center alignment (can be 'flex-start', 'center', 'flex-end')
                                   )
                                  )
                                   
    
    # Link the sliders to the heatmap function
    widgets.interact(create_paginated_heatmap, row_page=row_slider, col_page=col_slider)

    hbox = widgets.HBox([row_slider, col_slider])

    # Display the sliders
    display(hbox)


def plot_term_frequencies(tdm_df, start_range=0, end_range=100, logScale=False, ascending=None):
    term_frequencies = tdm_df.sum(axis=0).to_numpy()
    terms = tdm_df.columns.to_list()
    
    x_values = [s for s in terms]
    y_values = term_frequencies
    
    if logScale:
        y_values = [math.log(i) for i in y_values]
        title = "Interactive Term Log-Frequencies"
        yaxis_title ="Log Frequencies"
    else:
        title = "Interactive Term Frequencies"
        yaxis_title ="Frequencies"

    if ascending == True:
        # Sort term_frequencies in ascending order
        x_values = np.asarray(x_values)
        sorted_indices = np.argsort(y_values, )
        x_values = x_values[sorted_indices]
        y_values = y_values[sorted_indices]
    elif ascending == False:
        # Sort term_frequencies in descending order
        x_values = np.asarray(x_values)
        sorted_indices = np.argsort(y_values, )[::-1]
        x_values = x_values[sorted_indices]
        y_values = y_values[sorted_indices]

    # Create a figure object
    fig = go.Figure()
    
    # Add a bar trace with the initial range
    fig.add_trace(go.Bar(x=x_values[start_range:end_range], y=y_values[start_range:end_range]))
    
    # Update layout for better visualization and slider settings
    fig.update_layout(
        xaxis=dict(tickangle=-90),
        width=1000,
        height=500,
        title=title,
        xaxis_title="Terms",
        yaxis_title=yaxis_title,
        
        sliders=[{
            'active': 0,
            'currentvalue': {"prefix": "Range: "},
            'pad': {"t": 100},
            'steps': [
                {
                    'label': f'{i}-{i+end_range-1}',  # Show label for ranges (adjust end_range to control step size)
                    'method': 'update',
                    'args': [{
                        'x': [x_values[i:i+end_range-1]],
                        'y': [y_values[i:i+end_range-1]]
                    }]
                } for i in range(0, len(x_values), end_range)  # Steps of end_range terms at a time
            ]
        }]
    )
    
    # Show the interactive plot with sliders
    fig.show()

def plot_frequent_patterns(fp_df, start_range=0, end_range=30, ascending=False):
    fp_df = fp_df.sort_values(by='Support', ascending=ascending)
    
    term_supports = fp_df['Support'].to_numpy()
    terms = fp_df['Patterns'].to_list()
    
    x_values = [s for s in terms]
    y_values = term_supports
    
    title = "Interactive Term Supports"
    yaxis_title ="Supports"

    # Create a figure object
    fig = go.Figure()
    
    # Add a bar trace with the initial range
    fig.add_trace(go.Bar(x=x_values[start_range:end_range], y=y_values[start_range:end_range]))
    
    # Update layout for better visualization and slider settings
    fig.update_layout(
        xaxis=dict(tickangle=-90),
        width=1000,
        height=500,
        title=title,
        xaxis_title="Terms",
        yaxis_title=yaxis_title,
        
        sliders=[{
            'active': 0,
            'currentvalue': {"prefix": "Range: "},
            'pad': {"t": 100},
            'steps': [
                {
                    'label': f'{i}-{i+end_range-1}',  # Show label for ranges (adjust end_range to control step size)
                    'method': 'update',
                    'args': [{
                        'x': [x_values[i:i+end_range-1]],
                        'y': [y_values[i:i+end_range-1]]
                    }]
                } for i in range(0, len(x_values), end_range)  # Steps of end_range terms at a time
            ]
        }]
    )
    
    # Show the interactive plot with sliders
    fig.show()