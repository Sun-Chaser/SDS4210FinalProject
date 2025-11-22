import polars as pl 
from sklearn.mixture import GaussianMixture 
from sentence_transformers import SentenceTransformer

df = pl.read_csv("full_data_kalman.csv") 

model = SentenceTransformer("all-MiniLM-L6-v2") 
embeddings = model.encode(df["sicDescription"])  

gm = GaussianMixture(n_components=10, random_state=76).fit(embeddings) 
cluster = gm.fit_predict(embeddings) 

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


