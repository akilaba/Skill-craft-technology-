
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap

# Load a sample of the dataset (replace path with full dataset if needed)
df = pd.read_csv("US_Accidents_Dec21_updated.csv")

# Convert date column to datetime
df['Start_Time'] = pd.to_datetime(df['Start_Time'])

# Create additional features
df['Hour'] = df['Start_Time'].dt.hour
df['Weekday'] = df['Start_Time'].dt.day_name()
df['Month'] = df['Start_Time'].dt.month_name()

# Sample EDA: accidents by hour
plt.figure(figsize=(10,6))
sns.countplot(x='Hour', data=df, palette='mako')
plt.title("Accidents by Hour of the Day")
plt.xlabel("Hour")
plt.ylabel("Number of Accidents")
plt.show()

# Accidents by weather condition
plt.figure(figsize=(12,6))
top_weather = df['Weather_Condition'].value_counts().nlargest(10)
sns.barplot(x=top_weather.index, y=top_weather.values, palette='Set2')
plt.xticks(rotation=45)
plt.title("Top 10 Weather Conditions in Accidents")
plt.ylabel("Accident Count")
plt.show()

# Accidents by road condition (Side as proxy here)
plt.figure(figsize=(6,4))
sns.countplot(x='Side', data=df, palette='coolwarm')
plt.title("Accidents by Side of Road")
plt.xlabel("Road Side")
plt.ylabel("Count")
plt.show()

# Heatmap: Accidents by Hour and Weekday
heatmap_data = df.groupby(['Weekday', 'Hour']).size().unstack()
plt.figure(figsize=(12,6))
sns.heatmap(heatmap_data, cmap="YlGnBu")
plt.title("Accidents Heatmap by Hour and Weekday")
plt.show()

# Map Visualization (requires plotly or folium)
# Example with folium for hotspots
# Reduce dataset for performance
map_data = df[['Start_Lat', 'Start_Lng']].dropna().sample(10000)

# Create base map
m = folium.Map(location=[39.8283, -98.5795], zoom_start=5)
HeatMap(data=map_data.values, radius=8).add_to(m)

# Save map
m.save("accident_hotspots_map.html")
print("Interactive accident heatmap saved as 'accident_hotspots_map.html'")
