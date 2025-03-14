import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv("C:/Users/marya/Documents/PythonProjects/DataSets/data.csv")



# Select columns
selected_columns = [
   'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
   'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',
   'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
   'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# selected_columns = ['radius_mean', 'texture_mean']: if we want the clusters to be fully separated

selected_data = data[selected_columns]

# Scale the selected data
scaled_data = preprocessing.scale(selected_data)
# scaled_data = preprocessing.scale(selected_columns)
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(selected_data)

# Hard way of creating the data frame:
#dataFrame = pd.DataFrame(scaled_data, columns=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
#    'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',
#    'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
#    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'])


dataFrame = pd.DataFrame(scaled_data, columns=selected_columns)
print(dataFrame)

#Choosing the optimal k:
# Perform KMeans and determine the number of clusters using WCSS
# wcss = []
# for i in range(1, 15):
#     kmeans = KMeans(n_clusters=i)
#     kmeans.fit(dataFrame)
#     wcss.append(kmeans.inertia_)

# plt.figure(1)
# plt.plot(range(1, 15), wcss)
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show(block=False)



# Fit KMeans with the chosen number of clusters
kmeans = KMeans(n_clusters=2, init='k-means++')
cluster_labels = kmeans.fit_predict(scaled_data)

# # Plotting using all the features by reducing the dimensions to 2:
# # Apply PCA to reduce dimensions to 2D
# pca = PCA(n_components=2)
# principal_components = pca.fit_transform(scaled_data)

# # Add cluster labels to the PCA components
# pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
# pca_df['Cluster'] = cluster_labels

# # Plotting the clusters in 2D
# plt.figure(figsize=(8, 6))

# for cluster in pca_df['Cluster'].unique():
#     plt.scatter(
#         pca_df[pca_df['Cluster'] == cluster]['PC1'],
#         pca_df[pca_df['Cluster'] == cluster]['PC2'],
#         label=f'Cluster {cluster}'
#     )

# plt.scatter(
#     kmeans.cluster_centers_[:, 0],
#     kmeans.cluster_centers_[:, 1],
#     s=300,
#     c='red',
#     label='Centroids',
#     marker='x'
# )

# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('KMeans Clustering with PCA (2D)')
# plt.legend()
# plt.show()

 
# # Visualize clusters using 2 dimensions only due to the difficulty of reducing the dimensions and using all features: 
# plt.figure(2)

# # Assuming 'radius_mean' and 'texture_mean' are used for visualization
# plt.scatter(dataFrame['radius_mean'], dataFrame['texture_mean'], c=cluster_labels, cmap='cool')
# plt.xlabel('Radius Mean')
# plt.ylabel('Texture Mean')
# plt.grid()
# plt.show(block=False)



# # Plot centroids
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroids')
# plt.legend()
# plt.show()


# Calculate silhouette score
score = silhouette_score(dataFrame, cluster_labels, metric='euclidean')
print('Silhouette Score: %.3f' % score)


# Measure SSE (Sum of Squared Errors)
sse = kmeans.inertia_
print(f"Sum of Squared Errors (SSE): ", sse)

# Measure Silhouette Coefficient
silhouette_coefficient = silhouette_score(dataFrame, cluster_labels)
print(f"Silhouette Coefficient: ", silhouette_coefficient)