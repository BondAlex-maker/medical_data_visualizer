import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('medical_examination.csv')

overweight_mask = (df['weight'] / ((df['height'] / 100) ** 2)) > 25
df['overweight'] = overweight_mask.astype(int)

gluc_mask = df['gluc'] > 1
df['gluc'] = gluc_mask.astype(int)
cholesterol_mask = df['cholesterol'] > 1
df['cholesterol'] = cholesterol_mask.astype(int)


def draw_cat_plot():
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    g = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar'
    )

    fig = g.fig

    fig.savefig('catplot.png')
    return fig


def draw_heat_map():
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] < df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] < df['weight'].quantile(0.975))
        ]

    corr = df_heat.corr()
    corr = np.round(corr, 1)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        cmap='coolwarm',
        square=True,
        linewidths=0.5
    )

    fig.savefig('heatmap.png')
    return fig
