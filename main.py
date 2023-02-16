import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.display.float_format = "{:,.2f}".format
df_data = pd.read_csv("nobel_prize_data.csv")
# print(df_data.shape)
# print(df_data.head())

"""Check for Duplicates"""
dup_values = df_data.duplicated().values.any()
# print(f'Any duplicates? {dup_values}')

nan_values = df_data.isna().values.any()
# print(f'Any NaN values among the data? {nan_values}')

"""Check for NaN Values"""
nan_any = df_data.isna().any()
# print(nan_any)

nan_sum = df_data.isna().sum()
# print(nan_sum)

nan_rows = df_data[df_data.isna().values.any(axis=1)]
# print(nan_rows)

"""Why are there so many NaN values for the birth date?"""
col_subset = ['year', 'category', 'laureate_type', 'birth_date', 'full_name', 'organization_name']
bd_nan = df_data.loc[df_data.birth_date.isna()][col_subset]
# print(bd_nan)

"""Why are there so many missing values among the organisation columns?"""
col_subset = ['year', 'category', 'laureate_type', 'full_name', 'organization_name']
org_nan = df_data.loc[df_data.organization_name.isna()][col_subset]
# print(org_nan)

"""Convert Year and Birth Date to Datetime"""
df_data["birth_date"] = pd.to_datetime(df_data["birth_date"])
# df_data.info()

"""Add a Column with the Prize Share as a Percentage"""
# separated_values = df_data["prize_share"].str.split('/', expand=True)
# numerator = pd.to_numeric(separated_values[0])
# denominator = pd.to_numeric(separated_values[1])
# df_data["share_pct"] = numerator / denominator

separated_values = df_data["prize_share"].astype(str).str.split("/", expand=True).astype(int)
df_data["share_pct"] = separated_values[0] / separated_values[1]

# print(df_data["share_pct"])

"""Create a donut chart using plotly which shows how many prizes went to men compared to how many prizes went to women."""
biology = df_data["sex"].value_counts()
# print(biology)

# fig = px.pie(labels=biology.index,
#              values=biology.values,
#              title="Percentage of Male vs. Female Winners",
#              names=biology.index,
#              hole=0.4)
#
# fig.update_traces(textposition='inside', textfont_size=15, textinfo='percent')
#
# fig.show()

"""Who were the first 3 Women to Win the Nobel Prize?"""
first3_women = df_data[df_data["sex"] == "Female"].sort_values("sex", ascending=False)[:3]
# print(first3_women)

"""Did some people get a Nobel Prize more than once? If so, who were they?"""
is_winner = df_data.duplicated(subset="full_name", keep=False)
multiple_winners = df_data[is_winner]
# print(f"There are {multiple_winners.full_name.nunique()} winners who were awarded the prize more than once.")

col_subset = ['year', 'category', 'laureate_type', 'full_name']
multi_winner_rows = multiple_winners[col_subset]
# print(multi_winner_rows)

"""Find multiple prize winner names"""
# Example
df = pd.DataFrame({
    'brand': ["p1, p2", "p1", "p3, p4", "p5", "p5, p6"],
    'style': ["dual", "single", "dual", "single", "dual"]
})
# print(df)

sub = df["brand"].str.split(", ", expand=True)
# print(sub)

sta = sub.stack()
# print(sta)

dup = sta.duplicated(keep=False)
# print(dup)

name = sta[dup]
# print(name)

uni = name.unique()
# print(uni)

# Find winner names
split_winners = df_data["full_name"].str.split(", ", expand=True)
# print(split_winners)

stack_winners = split_winners.stack()
# print(stack_winners)

dup_winners = stack_winners.duplicated(keep=False)
# print(dup_winners)

all_winner = stack_winners[dup_winners]
# print(all_winner)

unique_winners = all_winner.unique()
# print(unique_winners)

"""In how many categories are prizes awarded? Which category has the most/fewest number of prizes awarded?"""
all_categories = df_data["category"].nunique()
# print(all_categories)

prizes_per_category = df_data["category"].value_counts()
# print(prizes_per_category)

"""Create a plotly bar chart with the number of prizes awarded by category."""
# v_bar = px.bar(x=prizes_per_category.index,
#                y=prizes_per_category.values,
#                color = prizes_per_category.values,
#                color_continuous_scale='Aggrnyl',
#                title="Number of Prizes Awarded per Category")
#
# v_bar.update_layout(xaxis_title="Nobel Prize Category",
#                     yaxis_title="Number of Prizes",
#                     coloraxis_showscale=False)
#
# v_bar.show()

"""When was the first prize in the field of Economics awarded?"""
first_econ_winner = df_data[df_data["category"] == "Economics"].sort_values("year")[:3]
# print(first_econ_winner)

"""Create a plotly bar chart that shows the split between men and women by category."""
cat_men_women = df_data.groupby(["category", "sex"], as_index=False).agg({"prize": pd.Series.count})
cat_men_women.sort_values("prize", ascending=False, inplace=True)
# print(cat_men_women)

# v_bar_split = px.bar(x=cat_men_women["category"],
#                      y=cat_men_women["prize"],
#                      color=cat_men_women["sex"],
#                      title="Number of Prizes Awarded per Category split by Men and Women")
#
# v_bar_split.update_layout(xaxis_title="Nobel Prize Category",
#                           yaxis_title="Number of Prizes")
#
# v_bar_split.show()

"""Are more prizes awarded recently than when the prize was first created?"""
# prize_per_year = df_data.groupby("year").agg({"prize": pd.Series.count})
prize_per_year = df_data.groupby(by="year").count()["prize"]
# print(prize_per_year)

moving_average = prize_per_year.rolling(window=5).mean()
# print(moving_average)

# plt.scatter(x=prize_per_year.index,
#             y=prize_per_year.values,
#             c="dodgerblue",
#             alpha=0.7,
#             s=100)
#
# plt.plot(prize_per_year.index,
#          moving_average.values,
#          c="crimson",
#          linewidth=3)
#
# plt.show()

"""Show a tick mark on the x-axis for every 5 years from 1900 to 2020. (Hint: you'll need to use NumPy)."""
figure_ticks = np.arange(1900, 2021, step=5)
# print(figure_ticks)

# plt.figure(figsize=(16, 8), dpi=200)
# plt.title("Number of Nobel Prizes Awarded per Year", fontsize=18)
# plt.yticks(fontsize=14)
#
# plt.xticks(ticks=np.arange(1900, 2021, step=5),
#            fontsize=14,
#            rotation=45)
#
# ax = plt.gca()  # get current axis
# ax.set_xlim(1900, 2020)
#
# plt.scatter(x=prize_per_year.index,
#             y=prize_per_year.values,
#             c="dodgerblue",
#             alpha=0.7,
#             s=100)
#
# plt.plot(prize_per_year.index,
#          moving_average.values,
#          c="crimson",
#          linewidth=3)
#
# plt.show()

"""Investigate if more prizes are shared than before. Calculate the 5 year rolling average of the percentage share."""
yearly_avg_share = df_data.groupby("year").agg({"share_pct": pd.Series.mean})
share_moving_average = yearly_avg_share.rolling(window=5).mean()
# share_moving_average[share_moving_average.notna().any(axis=1)]
# print(share_moving_average)

"""Modify the code to add a secondary axis and plot the rolling average of the prize share on this chart."""
# plt.figure(figsize=(16, 8), dpi=200)
# plt.title("Number of Nobel Prizes Awarded per Year", fontsize=18)
# plt.yticks(fontsize=14)
#
# plt.xticks(ticks=np.arange(1900, 2021, step=5),
#            fontsize=14,
#            rotation=45)
#
# ax1 = plt.gca()
# ax2 = ax1.twinx()  # create second y-axis
# ax1.set_xlim(1900, 2020)
#
# ax1.scatter(x=prize_per_year.index,
#             y=prize_per_year.values,
#             c="dodgerblue",
#             alpha=0.7,
#             s=100)
#
# ax1.plot(prize_per_year.index,
#          moving_average.values,
#          c="crimson",
#          linewidth=3)
#
# # Adding prize share plot on second axis
# ax2.plot(prize_per_year.index,
#          share_moving_average.values,
#          c="grey",
#          linewidth=3)
#
# plt.show()

"""Invert the secondary y-axis to make the relationship even more clear."""
# plt.figure(figsize=(16, 8), dpi=200)
# plt.title("Number of Nobel Prizes Awarded per Year", fontsize=18)
# plt.yticks(fontsize=14)
#
# plt.xticks(ticks=np.arange(1900, 2021, step=5),
#            fontsize=14,
#            rotation=45)
#
# ax1 = plt.gca()
# ax2 = ax1.twinx()  # create second y-axis
# ax1.set_xlim(1900, 2020)
#
# ax2.invert_yaxis()  # invert axis
#
# ax1.scatter(x=prize_per_year.index,
#             y=prize_per_year.values,
#             c="dodgerblue",
#             alpha=0.7,
#             s=100)
#
# ax1.plot(prize_per_year.index,
#          moving_average.values,
#          c="crimson",
#          linewidth=3)
#
# # Adding prize share plot on second axis
# ax2.plot(prize_per_year.index,
#          share_moving_average.values,
#          c="grey",
#          linewidth=3)
#
# plt.show()

"""Create a Pandas DataFrame called top20_countries that has the two columns. The prize column should contain the total number of prizes won."""
# Example:
# slicing_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# print(slicing_test[:-3])

top_countries = df_data.groupby("birth_country_current", as_index=False).agg({"prize": pd.Series.count})
top_countries.sort_values(by="prize", inplace=True)
top20_countries = top_countries[-20:]
# print(top20_countries)

# h_bar = px.bar(x=top20_countries.prize,
#                y=top20_countries.birth_country_current,
#                orientation='h',
#                color=top20_countries.prize,
#                color_continuous_scale='Viridis',
#                title='Top 20 Countries by Number of Prizes')
#
# h_bar.update_layout(xaxis_title='Number of Prizes',
#                     yaxis_title='Country',
#                     coloraxis_showscale=False)
#
# h_bar.show()

"""Use a Choropleth Map to Show the Number of Prizes Won by Country"""
df_countries = df_data.groupby(["birth_country_current", "ISO"], as_index=False).agg({"prize": pd.Series.count})
df_countries.sort_values("prize", ascending=False)
# print(df_countries)

# world_map = px.choropleth(df_countries,
#                           locations="ISO",
#                           color="prize",
#                           hover_name="birth_country_current",
#                           color_continuous_scale=px.colors.sequential.matter)
#
# world_map.update_layout(coloraxis_showscale=True)
#
# world_map.show()

"""See if you can divide up the plotly bar chart you created above to show the which categories made up the total number of prizes. """
cat_country = df_data.groupby(["birth_country_current", "category"], as_index=False).agg({"prize": pd.Series.count})
cat_country.sort_values(by="prize", ascending=False, inplace=True)
# print(cat_country)

merged_df = pd.merge(cat_country, top20_countries, on="birth_country_current")
merged_df.columns = ["birth_country_current", "category", "cat_prize", "total_prize"]
merged_df.sort_values(by="total_prize", inplace=True)
# print(merged_df)

# cat_cntry_bar = px.bar(x=merged_df.cat_prize,
#                        y=merged_df.birth_country_current,
#                        color=merged_df.category,
#                        orientation="h",
#                        title="Top 20 Countries by Number of Prize and Category")
#
# cat_cntry_bar.update_layout(xaxis_title="Number of Prizes",
#                             yaxis_title="Country")
#
# cat_cntry_bar.show()

"""Cumsum by index example"""
df_new = pd.DataFrame({"name": ["Jack", "Jack", "Jack", "Jack", "Jill", "Jill"],
                       "day": ["Monday", "Tuesday", "Tuesday", "Wednesday", "Monday", "Wednesday"],
                       "no": [10, 20, 10, 50, 40, 110]})

# print(df_new)

num_per_name_per_day = df_new.groupby(["name", "day"]).sum()
# print(num_per_name_per_day)

cumsum_num_per_name_by_day = df_new.groupby(['name', 'day']).sum().groupby(level=0).cumsum()
# print(cumsum_num_per_name_by_day)

cumulative_with_index = df_new.groupby(['name', 'day']).sum().groupby(level=0).cumsum().reset_index()
# print(cumulative_with_index)

"""Number of Prizes Won by Each Country Over Time step by step"""
annual_winners = df_data.groupby(["birth_country_current", "year"], as_index=False).count()
yearly_winners = annual_winners.sort_values("year")
prize_by_year = yearly_winners[["year", "birth_country_current", "prize"]]
# print(prize_by_year)

prize_per_country_per_year = prize_by_year.groupby(by=["birth_country_current", "year"]).sum()
# print(prize_per_country_per_year)

cumsum_prize_per_country_by_year = prize_by_year.groupby(by=["birth_country_current", "year"]).sum().groupby(level=0).cumsum()
# print(cumsum_prize_per_country_by_year)

cumsum_with_index = cumsum_prize_per_country_by_year.reset_index()
# print(cumsum_with_index)

"""Number of Prizes Won by Each Country Over Time"""
# cumulative_prizes = prize_by_year.groupby(by=["birth_country_current", "year"]).sum().groupby(level=["birth_country_current"]).cumsum()
cumulative_prizes = prize_by_year.groupby(by=["birth_country_current", "year"]).sum().groupby(level=[0]).cumsum()
cumulative_prizes.reset_index(inplace=True)
# print(cumulative_prizes)

# l_chart = px.line(data_frame=cumulative_prizes,
#                   x="year",
#                   y="prize",
#                   color="birth_country_current",
#                   hover_name="birth_country_current")
#
# l_chart.update_layout(xaxis_title="Year",
#                       yaxis_title="Number of Prizes")
#
# l_chart.show()

"""What are the Top Research Organisations?"""
top20_orgs = df_data["organization_name"].value_counts()[:20]
top20_orgs.sort_values(ascending=True, inplace=True)
# print(top20_orgs)

# org_bar = px.bar(x=top20_orgs.values,
#                  y=top20_orgs.index,
#                  color=top20_orgs.values,
#                  color_continuous_scale=px.colors.sequential.haline,
#                  orientation='h',
#                  title='Top 20 Research Institutions by Number of Prizes')
#
# org_bar.update_layout(xaxis_title='Number of Prizes',
#                       yaxis_title='Institution',
#                       coloraxis_showscale=False)
#
# org_bar.show()

"""Which Cities Make the Most Discoveries?"""
top20_org_cities = df_data["organization_city"].value_counts()[:20]
top20_org_cities.sort_values(ascending=True, inplace=True)
# print(top20_org_cities)

# org_city_bar = px.bar(x=top20_org_cities.values,
#                       y=top20_org_cities.index,
#                       orientation='h',
#                       color=top20_org_cities.values,
#                       color_continuous_scale=px.colors.sequential.Plasma,
#                       title='Which Cities Do the Most Research?')
#
# org_city_bar.update_layout(xaxis_title='Number of Prizes',
#                            yaxis_title='City',
#                            coloraxis_showscale=False)
#
# org_city_bar.show()

"""Where are Nobel Laureates Born? Chart the Laureate Birth Cities"""
top20_cities = df_data["birth_city"].value_counts()[:20]
top20_cities.sort_values(ascending=True, inplace=True)
# print(top20_cities)

# city_bar = px.bar(x=top20_cities.values,
#                   y=top20_cities.index,
#                   orientation='h',
#                   color=top20_cities.values,
#                   color_continuous_scale=px.colors.sequential.Plasma,
#                   title='Where were the Nobel Laureates Born?')
#
# city_bar.update_layout(xaxis_title='Number of Prizes',
#                        yaxis_title='City of Birth',
#                        coloraxis_showscale=False)
#
# city_bar.show()

"""Plotly Sunburst Chart: Combine Country, City, and Organisation"""
country_city_org = df_data.groupby(["organization_country", "organization_city", "organization_name"], as_index=False).agg({"prize": pd.Series.count})
country_city_org.sort_values("prize", ascending=False)
# print(country_city_org)

# burst = px.sunburst(country_city_org,
#                     path=['organization_country', 'organization_city', 'organization_name'],
#                     values='prize',
#                     title='Where do Discoveries Take Place?')
#
# burst.update_layout(xaxis_title='Number of Prizes',
#                     yaxis_title='City',
#                     coloraxis_showscale=False)
#
# burst.show()

"""Calculate the age of the laureate in the year of the ceremony and add this as a column called winning_age to the df_data DataFrame."""
birth_years = df_data.birth_date.dt.year
df_data["winning_age"] = df_data.year - birth_years
# print(df_data.head())

"""What are the names of the youngest and oldest Nobel laureate?"""
oldest_winner = df_data.nlargest(1, "winning_age")
# print(oldest_winner[["full_name", "winning_age"]])
youngest_winner = df_data.nsmallest(1, "winning_age")
# print(youngest_winner[["full_name", "winning_age"]])

# print(df_data["winning_age"].describe())

"""Visualise the distribution in the form of a histogram using Seaborn's .histplot() function."""
# plt.figure(figsize=(8, 4), dpi=200)
#
# sns.histplot(data=df_data,
#              x=df_data["winning_age"],
#              bins=30)
#
# plt.xlabel('Age')
# plt.title('Distribution of Age on Receipt of Prize')
#
# plt.show()

"""Use Seaborn to create a .regplot with a trendline."""
# plt.figure(figsize=(8, 4), dpi=200)
#
# with sns.axes_style("whitegrid"):
#     sns.regplot(data=df_data,
#                 x='year',
#                 y='winning_age',
#                 lowess=True,
#                 scatter_kws={'alpha': 0.4},
#                 line_kws={'color': 'black'})
#
# plt.show()

"""Use Seaborn's .boxplot() to show how the mean, quartiles, max, and minimum values vary across categories. Which category has the longest "whiskers"?"""
# plt.figure(figsize=(8, 4), dpi=200)
#
# with sns.axes_style("whitegrid"):
#     sns.boxplot(data=df_data,
#                 x='category',
#                 y='winning_age')
#
# plt.show()

"""Now use Seaborn's .lmplot() and the row parameter to create 6 separate charts for each prize category."""
# with sns.axes_style('whitegrid'):
#     sns.lmplot(data=df_data,
#                x='year',
#                y='winning_age',
#                row='category',
#                lowess=True,
#                aspect=2,
#                scatter_kws={'alpha': 0.6},
#                line_kws={'color': 'black'})
#
# plt.show()

"""Create another chart with Seaborn. This time use .lmplot() to put all 6 categories on the same chart using the hue parameter."""
# with sns.axes_style("whitegrid"):
#     sns.lmplot(data=df_data,
#                x='year',
#                y='winning_age',
#                hue='category',
#                lowess=True,
#                aspect=2,
#                scatter_kws={'alpha': 0.5},
#                line_kws={'linewidth': 5})
#
# plt.show()
