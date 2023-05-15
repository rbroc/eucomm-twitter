from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.dates as md
from matplotlib import pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go


FREQ_DICT = {'D': 'day', 'W': 'week', 'M': 'month'}
FIGSIZE = {'D': (50,5), 'W': (50,5), 'M': (30, 5)}
FORMAT_DICT = {'D': '%Y-%m-%d', 'W': '%Y-%m-%d', 'M': '%Y-%m'}


def _pca_transform(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA(n_components=X.shape[1])
    pca.fit(X)
    new_feats = pca.transform(X)
    return new_feats, pca


def plot_volume(data, 
                freq='D',
                exclude_entities=None,
                plot_smooth_only=True,
                roll_window=7, 
                save=True, savename=None,
                figsize=None, colors=None, 
                interactive=True,
                width=1600, height=500,
                ylim=True, multiple=False):
    ''' Plot tweet volume 
    Args:
        data: pd.DataFrame
        freq (str): D, W, or M
        exclude_entities (list): list of entities to exclude
        plot_smooth_only (bool): plot unsmoothed data 
        roll_window (int): smoothing window
        save (bool): whether to save
        savename (str): filename 
        figsize (tuple): figsize, if not default
    '''
    if interactive is False:
        figsize = figsize or FIGSIZE[freq]
        fig, ax = plt.subplots(figsize=figsize)
    entities = data.entity.unique()
    colors = colors or sns.color_palette()[:len(entities)]
    loop_over = set(entities) - set(exclude_entities)
    if len(loop_over) == 2:
        other = list(loop_over - set(['EU_Commission']))[0]
        loop_over = ['EU_Commission', other]
    for i, e in enumerate(loop_over):
        grouper = pd.Grouper(key='created_at', 
                             axis=0, freq=freq)
        if data['created_at'].dtype != 'datetime64':
            data['created_at'] = pd.to_datetime(data['created_at'], 
                                                infer_datetime_format=True)  
        grouped = data[data['entity']==e].groupby(grouper).count().reset_index()
        grouped['smoothed'] = grouped['text'].rolling(roll_window).mean()
        if i == 0:
            years = grouped.created_at.dt.year.unique()[1:]
        grouped['year'] =  grouped.created_at.dt.year
        if interactive is False:
            if multiple == False:
                plt.axhline(grouped['text'].mean(), 
                            color='darkred', 
                            linestyle='--')
            if plot_smooth_only is False:
                sns.lineplot(data=grouped,
                             x='created_at', y='text', 
                             color=colors[i], alpha=.2,
                             legend=False)
            if len(loop_over) == 2 and i == 0:
                alpha = 1. # 7
            else:
                alpha = 1.
            sns.lineplot(data=grouped,
                         x='created_at', y='smoothed', 
                         label=e if len(loop_over)>1 else None , 
                         color=colors[i], alpha=alpha)
        else:
            if i == 0:
                fig = px.line(title='Tweet volume', 
                              width=width, 
                              height=height,
                              template='plotly_dark')
            fig.add_trace(go.Scatter(
                x=grouped['created_at'],
                y=grouped.rename({'smoothed': f'Tweets per {FREQ_DICT[freq]}'}, 
                                  axis=1)[f'Tweets per {FREQ_DICT[freq]}'],
                mode="lines", name=e))

    if interactive is False:  
        plt.ylabel(f'Tweets per {FREQ_DICT[freq]}')
        plt.xlabel('')
        # plt.title(f'Tweet volume')
        plt.xticks(rotation=60)
        for d in years:
            plt.axvline(x=np.datetime64(f'{d}-01-01'), 
                        color='grey', 
                        linestyle='dotted')
            y = 43
            if d not in [2022, 2010]:
                plt.annotate(text=d, 
                             xy=(np.datetime64(f'{d}-05-01'), y), 
                             color='black', fontsize=10)
        ax.xaxis.set_major_locator(md.MonthLocator((1,7)))
        ax.xaxis.set_major_formatter(md.DateFormatter('%b \'%y'))
        plt.xlim(np.datetime64('2010-05-01'),np.datetime64('2022-08-01'))
        if ylim == True:
            plt.ylim(-5,50)
        if len(loop_over) != 1:
            plt.legend().remove()
            plt.title(loop_over[1])
        plt.tight_layout()
        plt.savefig(f'figs/topic_volume.png', dpi=300)
            
        plt.show()
    else:
        fig.update_xaxes(dtick="M6",
                         tickformat='%b \'%y', 
                         tickangle=-60, title='')
        fig.update_layout(legend_title_text='Agency')
        if save:
            fig.write_html(f"figs/{savename}.html")
        fig.show()
        


def plot_topic_volume(data, 
                      topics,
                      freq='D',
                      entity='EU_Commission',
                      exclude_topics=None,
                      plot_smooth_only=True,
                      roll_window=7, 
                      ylim=None,
                      save=True, savename=None,
                      figsize=None, colors=None, 
                      interactive=True,
                      width=1600, height=500,
                      title=None, xticksize=12):
    ''' Plot volume per topic
    Args:
        data: pd.DataFrame
        freq (str): D, W, or M
        exclude_topics(list): list of topics to exclude
        plot_smooth_only (bool): plot unsmoothed data 
        roll_window (int): smoothing window
        save (bool): whether to save
        savename (str): filename 
        figsize (tuple): figsize, if not default
    '''
    figsize = figsize or FIGSIZE[freq]
    fig, ax = plt.subplots(figsize=figsize)
    df = data[data['entity']==entity]
    colors = colors or sns.color_palette()[:len(topics)]
    loop_over = set(topics)
    if exclude_topics:
        loop_over = loop_over - set(exclude_topics)
    for i, t in enumerate(loop_over):
        grouper = pd.Grouper(key='created_at', 
                             axis=0, freq=freq)
        if df['created_at'].dtype != 'datetime64':
            df['created_at'] = pd.to_datetime(df['created_at'], 
                                              infer_datetime_format=True)
        grouped = df.groupby(grouper).mean().reset_index()
        grouped['smoothed'] = grouped[t].rolling(roll_window,
                                                 min_periods=1).mean()
        max_smooth = grouped['smoothed'].max()
        if plot_smooth_only is False:
            sns.lineplot(data=grouped,
                         x='created_at', y=t, 
                         color=colors[i], alpha=.1,
                         legend=False)
        sns.lineplot(data=grouped,
                     x='created_at', y='smoothed', 
                     label=t if len(loop_over)>1 else None , 
                     color=colors[i])

    plt.ylabel(f'Topic volume', fontsize=16)
    plt.xlabel('')
    plt.title(f'Topic volume' if title is None else title, fontsize=20)
    plt.xticks(rotation=60, fontsize=xticksize)
    plt.yticks(fontsize=12)
    for d in grouped.created_at.dt.year.unique()[1:]:
        plt.axvline(x=np.datetime64(f'{d}-01-01'), 
                    color='grey', 
                    linestyle='dotted')
    ax.xaxis.set_major_locator(md.MonthLocator((1,7)))
    ax.xaxis.set_major_formatter(md.DateFormatter('%b \'%y'))
    plt.xlim(np.datetime64('2010-05-01'),np.datetime64('2022-12-01'))
    if ylim:
        plt.ylim(*ylim)
    else:
        plt.ylim(0,max_smooth+.03)
    if save:
        plt.tight_layout()
        plt.savefig(f'figs/{savename}.png', dpi=300)
    plt.show()
    
    
    
def plot_style_timeseries(data, 
                          metric,
                          entities,
                          freq='D',
                          normalized=False,
                          exclude_topics=None,
                          plot_smooth_only=True,
                          roll_window=7, 
                          ylim=None,
                          save=True, savename=None,
                          figsize=None, colors=None, 
                          interactive=True,
                          width=1600, height=500,
                          title=None, no_horizontal=False,
                          legend=True):
    ''' Plot style over time '''
    figsize = figsize or FIGSIZE[freq]
    fig, ax = plt.subplots(figsize=figsize)
    colors = colors or sns.color_palette()[:len(entities)]
    if no_horizontal is False:  
        plt.axhline(0, linestyle='--', color='darkred')
    scaler = StandardScaler()
    data[metric] = scaler.fit_transform(data[[metric]])
    for i, e in enumerate(entities):
        df = data[data['entity']==e].copy()
        grouper = pd.Grouper(key='created_at', 
                             axis=0, freq=freq)
        if df['created_at'].dtype != 'datetime64':
            df['created_at'] = pd.to_datetime(df['created_at'], 
                                              infer_datetime_format=True)
        if 'sentiment' in metric:
            df[metric] = np.where(df[metric]>.5, 1, 0)
        grouped = df.groupby(grouper).mean().reset_index()
        if no_horizontal == 'mean':
            plt.axhline(0, linestyle='--', color='darkred')
        grouped['smoothed'] = grouped[metric].rolling(roll_window,
                                                      min_periods=1).mean()
        max_smooth = grouped['smoothed'].max()
        if plot_smooth_only is False:
            sns.lineplot(data=grouped,
                         x='created_at', y=metric, 
                         color=colors[i], alpha=.1,
                         legend=False)
        if e != 'EU_Commission':
            alpha = .15 #5
        else:
            alpha = 1
        sns.lineplot(data=grouped,
                     x='created_at', y='smoothed', 
                     label=e if legend else None, 
                     color=colors[i], alpha=alpha,
                     legend=legend
                    )
    if 'sentiment' not in metric:
        plt.ylabel(f'score', fontsize=12)
    else:
        plt.ylabel(f'% tweets', fontsize=12)
    if legend:
        plt.legend(fontsize=12, loc='upper right')
    plt.xlabel('')
    plt.title(f'{metric}', fontsize=13)
    plt.xticks(rotation=60, fontsize=11)
    plt.yticks(fontsize=11)
    for d in grouped.created_at.dt.year.unique()[1:]:
        plt.axvline(x=np.datetime64(f'{d}-01-01'), 
                    color='grey', 
                    linestyle='dotted')
    ax.xaxis.set_major_locator(md.MonthLocator((1,7)))
    ax.xaxis.set_major_formatter(md.DateFormatter('%b \'%y'))
    plt.xlim(np.datetime64('2010-05-01'),np.datetime64('2022-12-01'))
    if ylim:
        plt.ylim(*ylim)
    plt.tight_layout()
    if save:
        plt.savefig(f'figs/{savename}.png', dpi=300)
    plt.show()
        

