import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -------------------------------
# Individual Criminal Behaviour Clustering
# -------------------------------

individual_data = {
    'age': [25, 30, 22, 35, 40, 21, 19, 33],
    'crime_count': [5, 15, 7, 10, 12, 3, 2, 18],
    'crime_severity': [2, 5, 3, 4, 4, 1, 1, 5]
}

df_individual = pd.DataFrame(individual_data)

# Standardize data
scaler = StandardScaler()
scaled_features_individual = scaler.fit_transform(df_individual)

# Apply KMeans
kmeans_individual = KMeans(n_clusters=2, random_state=0)

# Create clusters
df_individual['cluster'] = kmeans_individual.fit_predict(
    scaled_features_individual
)

print("Individual Criminal Behavior Clustering:")
print(df_individual)

# -------------------------------
# District Crime Rate Segmentation
# -------------------------------

district_data = {
    'district': ['Central', 'East', 'West', 'North',
                 'South', 'East-2', 'Central-2', 'West-2'],

    'crime_count': [120, 85, 90, 60, 130, 78, 150, 88],

    'violent_crime': [40, 18, 27, 12, 56, 11, 60, 21],

    'property_crime': [60, 40, 37, 35, 55, 30, 70, 39],

    'fraud_cases': [20, 27, 26, 13, 19, 37, 20, 28],

    'population': [10000, 8000, 9000, 4000,
                   12000, 8500, 10500, 9200]
}

df_district = pd.DataFrame(district_data)

# Calculate crime rate per 1000 people
df_district['crime_rate_1000'] = (
    df_district['crime_count'] /
    df_district['population']
) * 1000

# Features for clustering
features = [
    'crime_count',
    'violent_crime',
    'property_crime',
    'fraud_cases',
    'crime_rate_1000'
]

# Standardize features
scaled_features_district = scaler.fit_transform(
    df_district[features]
)

# Apply KMeans clustering
kmeans_district = KMeans(
    n_clusters=3,
    random_state=42
)

df_district['segment'] = kmeans_district.fit_predict(
    scaled_features_district
)

print("\nDistrict Crime Rate Segmentation:")
print(df_district)

# -------------------------------
# GRAPH VISUALIZATION
# -------------------------------

plt.figure(figsize=(12, 7))

# Scatter Plot
scatter = plt.scatter(
    df_district['crime_rate_1000'],
    df_district['violent_crime'],
    c=df_district['segment'],
    cmap='viridis',
    s=200
)

# Add district labels
for i, district in enumerate(df_district['district']):

    plt.text(
        df_district['crime_rate_1000'][i] + 0.05,
        df_district['violent_crime'][i] + 0.5,
        district,
        fontsize=9
    )

# Labels and Title
plt.xlabel('Crime Rate per 1000 People')
plt.ylabel('Violent Crime Count')

plt.title('Crime Behaviour Segmentation by District')

# Color bar
plt.colorbar(scatter, label='Cluster Segment')

# Grid
plt.grid(True)

# Description Text
plt.figtext(
    0.1,
    -0.05,

    "This graph visualizes district-wise crime behaviour segmentation using K-Means clustering. "
    "Districts are grouped based on crime rate, violent crime, property crime, and fraud cases. "
    "Different colors represent different crime clusters, helping identify high-risk areas.",

    wrap=True,
    horizontalalignment='left',
    fontsize=10
)

# Save graph as image
plt.savefig("crime_segmentation_graph.png", bbox_inches='tight')

print("\nGraph saved successfully as 'crime_segmentation_graph.png'")

# Show graph
plt.show()