import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec


plt.style.use('dark_background')
sns.set_style("darkgrid")


vibrant_palette = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#B19CD9"]
sns.set_palette(vibrant_palette)


dark_background = "#1A1A2E"  
darker_panel = "#16213E"     
grid_color = "#3A3A4E"       
text_color = "white"        


np.random.seed(42)
years = range(2009, 2023)
pollutants = ['PM2.5', 'NO2', 'O3', 'SO2', 'CO']
boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
neighborhoods = ['Midtown', 'Downtown', 'Uptown', 'Williamsburg', 'Astoria', 'Flushing', 'Jamaica']


df = pd.DataFrame({
    'Year': np.random.choice(years, 2000),
    'Month': np.random.randint(1, 13, 2000),
    'Indicator': np.random.choice(pollutants, 2000),
    'Geo Place Name': np.random.choice(boroughs + neighborhoods, 2000),
    'Data Value': np.abs(np.random.normal(20, 10, 2000))
})


df['Season'] = df['Month'].apply(
    lambda x: 'Winter' if x in [12,1,2] else 
              'Spring' if x in [3,4,5] else 
              'Summer' if x in [6,7,8] else 'Fall')



pm25_data = df[df['Indicator'] == 'PM2.5']
no2_data = df[df['Indicator'] == 'NO2']
o3_data = df[df['Indicator'] == 'O3']



fig = plt.figure(figsize=(18, 12), facecolor=dark_background)
gs = GridSpec(2, 2, figure=fig)



ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor(darker_panel)
sns.lineplot(
    data=df[df['Indicator'].isin(['PM2.5', 'NO2', 'O3'])], 
    x='Year', y='Data Value', hue='Indicator',
    style='Indicator', markers=True, dashes=False,
    linewidth=3, markersize=12,
    estimator='median', errorbar=None,
    ax=ax1
)
ax1.set_title('Air Pollution Trends in NYC (2009-2022)\n', 
             fontsize=16, fontweight='bold', pad=20, color=text_color)
ax1.set_ylabel('Concentration', fontsize=12, color=text_color)
ax1.set_xlabel('Year', fontsize=12, color=text_color)
ax1.grid(color=grid_color, linestyle='--', alpha=0.6)
ax1.tick_params(colors='lightgray')
ax1.legend(title='Pollutant', title_fontsize=12, 
          facecolor=darker_panel, framealpha=0.8,
          labelcolor=text_color)



ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor(darker_panel)
sns.boxplot(
    data=o3_data, x='Season', y='Data Value', 
    order=['Winter', 'Spring', 'Summer', 'Fall'],
    width=0.6, linewidth=2, fliersize=6,
    palette="YlOrRd_r",
    ax=ax2
)
ax2.set_title('Seasonal Variation of Ozone (O3)\n', 
             fontsize=14, fontweight='bold', pad=15, color=text_color)
ax2.set_ylabel('O3 Concentration (ppb)', fontsize=11, color=text_color)
ax2.set_xlabel('Season', fontsize=11, color=text_color)
ax2.grid(axis='y', color=grid_color, linestyle='--', alpha=0.6)
ax2.tick_params(colors='lightgray')



ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor(darker_panel)
borough_avg = pm25_data.groupby('Geo Place Name')['Data Value'].mean().sort_values()
borough_avg = borough_avg[borough_avg.index.isin(boroughs)]
sns.barplot(
    x=borough_avg.values, y=borough_avg.index,
    palette="viridis_r",
    saturation=0.9,
    ax=ax3
)
ax3.set_title('Average PM2.5 by Borough\n', 
             fontsize=14, fontweight='bold', pad=15, color=text_color)
ax3.set_xlabel('PM2.5 (µg/m³)', fontsize=11, color=text_color)
ax3.set_ylabel('', fontsize=11, color=text_color)
ax3.grid(axis='x', color=grid_color, linestyle='--', alpha=0.6)
ax3.tick_params(colors='lightgray')

plt.tight_layout()
plt.show()



plt.figure(figsize=(16, 8), facecolor=dark_background)
gs = GridSpec(1, 2, width_ratios=[2, 1])



ax1 = plt.subplot(gs[0])
ax1.set_facecolor(darker_panel)
sns.violinplot(
    data=df, x='Indicator', y='Data Value',
    inner='quartile', palette='rocket_r',
    cut=0, linewidth=2,
    ax=ax1
)
ax1.set_title('Pollution Distribution by Type\n', 
             fontsize=16, fontweight='bold', pad=20, color=text_color)
ax1.set_ylabel('Concentration', fontsize=12, color=text_color)
ax1.set_xlabel('Pollutant', fontsize=12, color=text_color)
ax1.grid(axis='y', color=grid_color, linestyle='--', alpha=0.6)
ax1.tick_params(colors='lightgray')



ax2 = plt.subplot(gs[1])
ax2.set_facecolor(dark_background)
pollution_counts = df['Indicator'].value_counts()
explode = [0.1 if i == pollution_counts.idxmax() else 0 for i in pollution_counts.index]
ax2.pie(
    pollution_counts, labels=pollution_counts.index,
    autopct='%1.1f%%', startangle=90,
    colors=vibrant_palette, explode=explode,
    textprops={'color': text_color, 'fontsize': 10},
    wedgeprops={'edgecolor': dark_background, 'linewidth': 2}
)
ax2.set_title('Pollution Measurement Composition\n', 
             fontsize=16, fontweight='bold', pad=20, color=text_color)

plt.tight_layout()
plt.show()



plt.figure(figsize=(14, 8), facecolor=dark_background)
ax = plt.gca()
ax.set_facecolor(darker_panel)



heatmap_data = df.pivot_table(
    index='Month', columns='Indicator', 
    values='Data Value', aggfunc='median'
)

sns.heatmap(
    heatmap_data, cmap="magma_r",
    annot=True, fmt=".1f", linewidths=0.5,
    annot_kws={"color": "white", "fontsize": 9},
    cbar_kws={"label": "Concentration"}
)
plt.title('Monthly Pollution Patterns\n', 
         fontsize=16, fontweight='bold', pad=20, color=text_color)
plt.xlabel('Pollutant', fontsize=12, color=text_color)
plt.ylabel('Month', fontsize=12, color=text_color)
plt.xticks(color='lightgray')
plt.yticks(color='lightgray', rotation=0)
plt.tight_layout()
plt.show()



plt.figure(figsize=(12, 10), facecolor=dark_background)
ax = plt.gca()
ax.set_facecolor(darker_panel)



corr_data = df.pivot_table(
    index=['Year', 'Month'], 
    columns='Indicator', 
    values='Data Value'
).corr()

mask = np.triu(np.ones_like(corr_data, dtype=bool))
sns.heatmap(
    corr_data, mask=mask, cmap="coolwarm",
    annot=True, fmt=".2f", center=0,
    linewidths=0.5, square=True,
    annot_kws={"color": "white", "fontsize": 10},
    cbar_kws={"label": "Correlation Coefficient"}
)
plt.title('Pollution Type Correlations\n', 
         fontsize=16, fontweight='bold', pad=20, color=text_color)
plt.xticks(color='lightgray')
plt.yticks(color='lightgray', rotation=0)
plt.tight_layout()
plt.show()



g = sns.FacetGrid(
    data=df[df['Geo Place Name'].isin(boroughs)],
    col='Geo Place Name', col_wrap=3,
    height=4, aspect=1.2,
    sharey=True, sharex=True,
    facecolor=dark_background
)
g.map_dataframe(
    sns.lineplot, x='Year', y='Data Value',
    hue='Indicator', estimator='median',
    linewidth=2.5, palette=vibrant_palette
)
g.set_titles(col_template="{col_name}", size=12, color=text_color)
g.fig.suptitle('Pollution Trends by Borough\n', 
              fontsize=16, fontweight='bold', y=1.05, color=text_color)
g.set_axis_labels("Year", "Concentration", color=text_color)
g.set(facecolor=darker_panel)
g.add_legend(title='Pollutant', facecolor=darker_panel, 
             labelcolor=text_color, framealpha=0.8)



for ax in g.axes.flat:
    ax.grid(color=grid_color, linestyle='--', alpha=0.6)
    ax.tick_params(colors='lightgray')
    ax.set_facecolor(darker_panel)

plt.tight_layout()
plt.show()