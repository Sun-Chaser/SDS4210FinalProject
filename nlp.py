import polars as pl 
import pandas as pd 
import seaborn as sns 
from sklearn.mixture import GaussianMixture 
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 

df = pl.read_csv("Data/full_data_kalman.csv") 

model = SentenceTransformer("all-MiniLM-L6-v2") 
embeddings = model.encode(df["sicDescription"])  

gm = GaussianMixture(n_components=10, random_state=76).fit(embeddings) 
cluster = gm.fit_predict(embeddings) 

scaler = StandardScaler() 
scaled_embeddings = scaler.fit_transform(embeddings) 

pca = PCA(n_components=2)
pca_results = pca.fit_transform(scaled_embeddings)

pca_df = pd.DataFrame(pca_results, columns=["PCA1","PCA2"])
pca_df["cluster"] = cluster 

# plot the two PCA with the cluster as color
sns.scatterplot(pca_df, x="PCA1", y="PCA2", hue="cluster", palette="deep")

df = df.with_columns(
    pred_cluster = cluster 
)



df_reduced = (
    df
    .unpivot(
        index = [col for col in df.columns if not col.startswith("20")], 
        on = [col for col in df.columns if col.startswith("20")], 
        variable_name = "year_quarter", 
        value_name = "gross_income"
    )
    .drop_nulls("gross_income")
    .with_columns(
        year = pl.col("year_quarter").str.slice(0,4).cast(pl.Int32), 
        gross_income = (pl.col("gross_income") / 1e8) 
    )
    .filter(pl.col("year").is_between(2021, 2024))
    .group_by(["year","cik","entityName","loc","sicDescription","pred_cluster"])
    .agg(pl.col("gross_income").sum().alias("yearly_gross")) 
)

df_reduced.write_csv("yearly_gross_with_cluster.csv") 


