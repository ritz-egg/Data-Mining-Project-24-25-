import pandas as pd
import numpy as np
import math

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import warnings

from itertools import combinations
from collections import Counter


from matplotlib.gridspec import GridSpec

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer

from sklearn.preprocessing import RobustScaler


warnings.filterwarnings('ignore')
%matplotlib inline

df = pd.read_csv("DM2425_ABCDEats_DATASET.csv")


# ----------- DUPLICATES 
df[df.duplicated(keep=False)]

df_original = df.copy()
df = df.drop_duplicates()
df_original.shape[0] - df.shape[0] # 13 dups 

df.set_index("customer_id", inplace=True)

df[df.duplicated(keep=False)] 
n_dups = len(df[df.duplicated(keep=False)]) / 2
n_dups # 47 
df_len_prev = df.shape[0]
df = df.drop_duplicates()
n_dups_droped = df_len_prev - df.shape[0]
n_dups_droped

hr_cols = [col for col in df.columns if "HR" in col]
dow_cols = [col for col in df.columns if "DOW" in col]
CUI_cols = [col for col in df.columns if "CUI" in col]


# ------------------------------------------------- Numerical Features ----------------------------------------

# ----------- AUXILIARY FUNCTIONS 

def plot_numerical(df, col):
    sns.set(style="white")
    fig = plt.figure(figsize=(8, 6), tight_layout=True)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2, 1.7], hspace=0.03)
    
    col_median = np.median(df[col])
    
    ax1 = fig.add_subplot(gs[0, 0])
    bin_edges = np.histogram_bin_edges(df[col], bins='auto')
    if df[col].dtype == int:
        bin_edges = np.arange(int(bin_edges.min()), int(bin_edges.max()) + 1)

    ax1.hist(df[col], bins=bin_edges, alpha=0.9, color="lightblue", edgecolor="gray")
    
    if df[col].dtype == int:
        ax1.axvline(df[col].mode()[0], color='orange', linestyle='--', label=f"mode: {round(df[col].mode()[0])}", alpha=0.8)    
    ax1.axvline(col_median, color='lightgray', linestyle='--', label=f"median: {round(col_median)}", alpha=0.8)
    
    ax1.set_xticks([])
    ax1.set_ylabel("frequency")
    ax1.legend()
    ax1.grid(True, linestyle='-', alpha=0.6)
    ax2 = fig.add_subplot(gs[1, 0])
    sns.boxplot(x=df[col], ax=ax2, color="lightblue", width=0.25,
                boxprops=dict(alpha=0.5), flierprops=dict(marker='o', alpha=0.35))
    ax2.set_xlabel(col)
    ax2.set_yticks([]) 
    ax2.grid(True, linestyle='-', alpha=0.6)

    plt.suptitle(f"'{col}' distribution", fontsize=14, fontweight='bold')
    plt.show()
    
    
    
    

def plot_distribution_grid(df, subset_num, cols=2, title="Feature Distributions"):
    sns.set(style="white")
    rows = math.ceil(len(subset_num) / cols)
    fig = plt.figure(figsize=(cols * 9, rows * 5))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.94) 
    outer = gridspec.GridSpec(rows, cols, wspace=0.3, hspace=0.5)

    for i, feature in enumerate(subset_num):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i], wspace=0.1, hspace=0.1)

        for j in range(2):
            ax = plt.Subplot(fig, inner[j])
            if j == 0:
                img = df.copy()
                img.dropna(subset=[feature], inplace=True)
                bin_edges = np.histogram_bin_edges(img[feature], bins='auto')
                if img[feature].dtype == int:
                    bin_edges = np.arange(int(bin_edges.min()), int(bin_edges.max()) + 1)
                sns.histplot(img[feature], bins=bin_edges, kde=False, ax=ax, color="lightblue", edgecolor="gray", alpha=0.7)
                
                if img[feature].dtype == int:
                    ax.axvline(img[feature].mode()[0], color='orange', linestyle='--', label=f"mode: {round(img[feature].mode()[0])}", alpha=0.8)    
                ax.axvline(img[feature].median(), color='gray', linestyle='--', label=f'median: {round(img[feature].median())}', alpha=0.8)
                
                
                ax.legend()
                ax.set_xticks([])
                ax.set_xlabel('')
                ax.set_title(f'Distribution of {feature}')
                ax.grid(True, linestyle='-', alpha=0.6, axis='both')
            else:
                sns.boxplot(x=img[feature], ax=ax, color="lightblue", width=0.25,
                            boxprops=dict(alpha=0.5), flierprops=dict(marker='o', alpha=0.35))
                ax.grid(True, linestyle='-', alpha=0.6)
            fig.add_subplot(ax)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    plt.show()





def compare_figure_outliers(df_original, df, num_feats):
    sns.set_style('whitegrid')
    frows = math.ceil(len(num_feats) / 2)
    fcols = 2
    
    fig = plt.figure(figsize=(15, 5 * frows))
    
    subfigs = fig.subfigures(frows, fcols, wspace=0.03, hspace=0.03)
    
    for sfig, feat in zip(subfigs.flatten(), num_feats):
        axes = sfig.subplots(2, 1, sharex=True)
        
        sns.boxplot(x=df_original[feat], ax=axes[0])
        axes[0].set_ylabel("Original")
        axes[0].set_title(feat, fontsize="large")
        
        sns.boxplot(x=df[feat], ax=axes[1])
        axes[1].set_ylabel("Outliers\nRemoved")
        axes[1].set_xlabel("")
        
        sfig.set_facecolor("#F9F9F9")
        sfig.subplots_adjust(left=0.2, right=0.95, bottom=0.1)
        
    plt.show()
    sns.set()
    
    

# -------------- Numericals 

numericals = [col for col in df.columns if df[col].dtype != object and df[col].dtype != bool]
subset_num = [x for x in numericals if x not in hr_cols and x not in CUI_cols and x not in dow_cols]
plot_distribution_grid(df, subset_num, title="Distribution of Numerical Features")



plot_distribution_grid(df, dow_cols, title="Distribution of Numerical Features")
plot_distribution_grid(df, hr_cols, title="Distribution of Numerical Features")



# --------------- First order 

df["first_order"].isna().sum()


# ----------- Total orders 

df["total_orders"] = df[dow_cols].apply(lambda x: x.sum(), axis=1)
df["total_orders"].describe().T
# INCOSISTENCY BEFORE
df[df["total_orders"] == 0].shape # 138 cases
len_original = df.shape[0]
df = df[df["total_orders"] > 0]
len_original - df.shape[0] # 138 cases

# --------------- Continuation of (First Order)

df[df.first_order.isna()].total_orders.value_counts()
df[(df.total_orders == 1) & (df.first_order > 1)][["first_order", "last_order", "total_orders"]]

df["first_order"] = np.where(df.first_order.isna(), df.last_order, df.first_order)
df.first_order.isna().sum()


# ------------------- Product count 

# Inconsistency  
df["product_count"].describe()

df[df["product_count"] == 0].shape # 18 cases 

df[df["product_count"] == 0]["total_orders"].value_counts()

df["product_count"] = np.where((df["product_count"] == 0) & (df[CUI_cols].sum(axis=1) > 0), 1, df["product_count"])

df[df["product_count"] == 0].shape # 0 cases 

len_original = df.shape[0]

df = df[df["product_count"] > 0]



# ------------------- Vendor count
df["vendor_count"].describe()




# --------------- HR_0 
df["HR_0"].isna().sum()

df["total_hours"] = df[hr_cols].apply(lambda x: x.sum(), axis=1)
df["total_days"] = df[dow_cols].apply(lambda x: x.sum(), axis=1)

df["HR_0"] = np.where(df.HR_0.isna(), df.total_days - df.total_hours, df.HR_0)
df.drop(columns=["total_hours", "total_days"], inplace=True)
df.HR_0.isna().sum()

df["HR_0"].describe()





# ------------------------- New features --------------------------------------------#  

# ------------------ Product per vendor
df["products_per_vendor"] = round(df.product_count / df.vendor_count)

df["products_per_vendor"] = round(df["products_per_vendor"])
df["products_per_vendor"].describe()

plot_numerical(df, "products_per_vendor")

df["products_per_vendor"].isna().sum()




# ----------- Percent chain ----------------------> ESTA
df["percentage_chain"] = np.where(df.product_count == 0, 0, df.is_chain / df.total_orders * 100)

plot_numerical(df, "percentage_chain")

# ----------- Customer lifetime ----------------------> ESTA (MAS ESTA DA PARA VER COM CATEGORICA) ---------> ACTIVE / RECENT / CHURNING
df["customer_lifetime"] = df.last_order - df.first_order
df["customer_lifetime"].describe().T
df["customer_lifetime"].isna().sum()
df[df["customer_lifetime"].isna()]

plot_numerical(df, "customer_lifetime")

# ---------- Order freq --------------------> ESTA
df["weekly_order_freq"] = np.where(df.customer_lifetime== 0, df.total_orders / (1 / 7), df.total_orders / (df.customer_lifetime / 7))
plot_numerical(df, "weekly_order_freq")
df["weekly_order_freq"].describe().T

df["weekly_order_freq"] = round(df["weekly_order_freq"])

plot_numerical(df, "weekly_order_freq")


# ----------- Avg time between orders
df["avg_days_between_orders"] = np.where(df.total_orders == 0, 0, df.customer_lifetime / df.total_orders)

(df.total_orders == 0).sum()
plot_numerical(df, "avg_days_between_orders")


# ----------- Total spent
df["total_spent"] = df[CUI_cols].apply(lambda x: x.sum(), axis=1)
df["total_spent"].describe().T

# INCOSISTENCY BEFORE
df[df["total_spent"] == 0].shape # 138 cases ?? 
df[df["total_spent"] == 0]

plot_numerical(df, "total_spent")

# ----------- Avg  spent weekly -----------------> ESTA
df["avg_total_spent_weekly"] = np.where(df.customer_lifetime == 0, df.total_spent / (1 / 7), df.total_spent / (df.customer_lifetime / 7))
plot_numerical(df, "avg_total_spent_weekly")

plot_distribution_grid(df, ["avg_total_spent_weekly", "total_spent"])
df["avg_total_spent_weekly"].describe().T
df["avg_total_spent_weekly"].mode()
df["avg_total_spent_weekly"].dtype

# ----------- Avg order value ----------------> ESTA 
df["avg_order_value"] = np.where(df.total_orders == 0, 0, df.total_spent / df.total_orders)
df["avg_order_value"].describe().T

numericals = [col for col in df.columns if df[col].dtype != object and df[col].dtype != bool]
subset = numericals[-8:]
plot_distribution_grid(df, subset)


# ------------------ Weekend orders 


subset = dow_cols
plot_distribution_grid(df, subset)

df[dow_cols].describe().T
df[dow_cols].dtypes

dof = {}
for col in dow_cols:
    dof[col] = df[col].sum()
dof = pd.Series(dof)
x = np.arange(len(dof))
y = dof.values
m, b = np.polyfit(x, y, 1) # linear regression; m = slope; b = intercept
dof.plot(kind="bar")
plt.plot(x, m*x + b, color="red", linestyle="--", label="trend")
plt.legend()
plt.show()



df["weekend_orders"] = df[["DOW_0", "DOW_6"]].apply(lambda x: x.sum(), axis=1)

# df["avg_n_weekend_orders"] = df[["DOW_0", "DOW_6"]].apply(lambda x: x.sum(), axis=1) / 2
# df["avg_n_weekend_orders"].describe()
# df["avg_n_weekend_orders"].isna().sum()

# plot_numerical(df, "avg_n_weekend_orders")

# df["avg_n_weekend_orders"] = round(df["avg_n_weekend_orders"])

# df["avg_n_weekend_orders"] = df["avg_n_weekend_orders"].astype(int)

# plot_numerical(df, "avg_n_weekend_orders")


# ------------------ Week orders

df["week_orders"] = df[["DOW_1", "DOW_2", "DOW_3", "DOW_4", "DOW_5"]].apply(lambda x: x.sum(), axis=1)

# df["avg_n_week_orders"] = df[["DOW_0", "DOW_1", "DOW_2", "DOW_3", "DOW_4"]].apply(lambda x: x.sum(), axis=1) / 5

# df["avg_n_week_orders"].describe()

# df["avg_n_week_orders"].isna().sum()

# df["avg_n_week_orders"] = round(df["avg_n_week_orders"])

# df["avg_n_week_orders"] = df["avg_n_week_orders"].astype(int)

# plot_numerical(df, "avg_n_week_orders")

# plot_distribution_grid(df, ["avg_n_weekend_orders", "avg_n_week_orders"])

# plot_numerical(df, "avg_n_weekend_orders")


# ------------------ 
    
# ------------------ Weekend percent -------------> ESTA 

df["weekend_percent"] = df["weekend_orders"] / df["total_orders"] * 100

df["weekend_percent_normalized"] = (df[["DOW_0", "DOW_6"]].apply(lambda x: x.sum(), axis=1) / 2) / df[dow_cols].apply(lambda x: x.sum(), axis=1) * 100

plot_numerical(df, "weekend_percent")

df["weekend_percent"].describe()




# ------------------ Hours aggregation

hours = {}
hr_cols = [col for col in df.columns if "HR" in col]
df[hr_cols].describe().T
for col in hr_cols:
    hours[col] = df[col].sum()
hours = pd.Series(hours)
hours.plot(kind="bar")
hours_df = pd.DataFrame(hours)
plt.show()

plot_distribution_grid(df, hr_cols)



five_hours_bins = [0, 5, 10, 15, 20]

df["HR_0_5"] = df[hr_cols[:5]].apply(lambda x: x.sum(), axis=1)
df["HR_6_11"] = df[hr_cols[6:11]].apply(lambda x: x.sum(), axis=1)
df["HR_12_17"] = df[hr_cols[12:17]].apply(lambda x: x.sum(), axis=1)
df["HR_18_23"] = df[hr_cols[18:]].apply(lambda x: x.sum(), axis=1)

new_hrs = ["HR_0_5", "HR_6_11", "HR_12_17", "HR_18_23"]
df[new_hrs].describe().T

plot_distribution_grid(df, new_hrs)



# ------------------ Night preference ------------>  ESTA 

df["night_preference_ratio"] = df[["HR_0_5", "HR_18_23"]].apply(lambda x: x.sum(), axis=1) / df["total_orders"] * 100

df["night_preference_ratio"].describe()

plot_numerical(df, "night_preference_ratio")





# ------------------- Percent CUISINE 


totals = {}
df[CUI_cols].describe().T
for col in CUI_cols:
    totals[col] = df[col].sum()
totals = pd.Series(totals)
totals.sort_values(ascending=False).plot(kind="bar")
plt.show()



for cuisine in CUI_cols:
    df[cuisine + "_percent"] = df[cuisine] / df[CUI_cols].sum(axis=1) * 100

cuisine_percent = [col for col in df.columns if "percent" in col and "CUI" in col]
df[cuisine_percent].describe().T

plot_distribution_grid(df, cuisine_percent)



df[cuisine_percent].columns


# Aggregate cuisines  ???? 
 
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler

ex = df.copy()

cuisine_percent
normalized_data = StandardScaler().fit_transform(ex[cuisine_percent].T)
kmeans = KMeans(n_clusters=15, random_state=42, init='k-means++', n_init=15)

cuisines_clusters = kmeans.fit_predict(normalized_data)



cuisine_cluster_df = pd.DataFrame({
    'Cuisine': ex[cuisine_percent].columns,
    'Cluster': cuisines_clusters
})
cuisine_cluster_df
print(cuisine_cluster_df.groupby("Cluster").value_counts())

centroids = kmeans.cluster_centers_ 
centroids.shape

from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters=2, linkage='ward')
agg_clusters = agg.fit_predict(centroids)

from scipy.cluster.hierarchy import dendrogram, linkage

linked = linkage(centroids, 'ward')
linked.shape

plt.figure(figsize=(10, 6))
dendrogram(linked, labels=[f'Cluster {i}' for i in range(len(centroids))])
plt.title('Dendrogram of K-means Centroids')
plt.xlabel('K-means Clusters')
plt.ylabel('Distance')
plt.xticks(rotation=45)
plt.show()




df[CUI_cols].columns
df["CUI_grouped_Asian"] = df["CUI_Asian"] + df["CUI_Japanese"] + df["CUI_Thai"] + df["CUI_Chinese"] + df["CUI_Indian"]

df["CUI_Other_grouped"] = df["CUI_OTHER"] + df["CUI_Beverages"] + df["CUI_Healthy"] + df["CUI_Street Food / Snacks"] + df["CUI_Desserts"] + df["CUI_Cafe"] + df["CUI_Chicken Dishes"] + df["CUI_Noodle Dishes"] + df["CUI_American"] + df["CUI_Italian"]




plot_distribution_grid(df, new_CUI_cols)

for col in new_CUI_cols:
    df[col + "_percent"] = df[col] / df[new_CUI_cols].sum(axis=1) * 100

new_CUI_percent_cols = [col for col in df.columns if "_percent" in col and "CUI" in col]
new_CUI_percent_cols[-3:]

plot_distribution_grid(df, new_CUI_percent_cols[-3:])


CUI_cols = [col for col in df.columns if "CUI" in col]
df[CUI_cols].describe().T
for col in CUI_cols:
    totals[col] = df[col].sum()
totals = pd.Series(totals)
totals.sort_values(ascending=False).plot(kind="bar")
plt.show()

# -- value based: avg order valye / total freq / order freq 
# -- preferences: cozinha (15) -- 3 clusters 


# ------------------------------------------------- Categorical Features ----------------------------------------

cat_cols = [col for col in df.columns if df[col].dtype == object or df[col].dtype == bool]

cols = 2
rows = math.ceil(len(cat_cols) / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*5))
for ax, feat in zip(axes.flatten(), cat_cols):
    order = df[feat].value_counts().index
    sns.countplot(x=feat, data=df, ax=ax, order=order)
plt.suptitle("Categorical Variables Distribution")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

df["favorite_cuisine"] = df[CUI_cols].idxmax(axis=1).apply(lambda x: x.split("_")[1])

main_cat = "customer_region"
len(df.favorite_cuisine.unique())
predominant_cuisine = df.groupby(main_cat)["favorite_cuisine"].apply(lambda x: x.mode().iloc[0])
cuisine_colors = {"Italian": "pink", "OTHER": "#4f82d1", "Asian": "yellow"}
for cuisine in df["favorite_cuisine"].unique():
    if cuisine not in cuisine_colors.keys():
        cuisine_colors[cuisine] = "grey"
(df.groupby(["customer_region", "favorite_cuisine"]).size() / df.groupby(["customer_region"])["favorite_cuisine"].size()).unstack().plot(
    kind='bar',
    stacked=True,
    figsize=(15, 5),
    color=cuisine_colors)
plt.title(f"Favorite Cuisine by Customer Region")
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))

df.drop(columns=["favorite_cuisine"], inplace=True)

# TRANSFORMATION 
df["customer_region"] = np.where(df["customer_region"].str.startswith("8"), "8670", df["customer_region"])
df["customer_region"] = np.where(df["customer_region"].str.startswith("4"), "4660", df["customer_region"])
df["customer_region"] = np.where(df["customer_region"].str.startswith("2"), "2370", df["customer_region"])
df["customer_region"].value_counts(normalize=True).plot(kind="bar")
plt.title("Customer Regions")
plt.show()


# --------------------- Imputation of missing values ---------------------


# --------------- Customer_age and customer_region: 

indexes = df[df["customer_region"] == "-"].index

index_age = df[df["customer_age"] == np.nan].index
df["customer_region"] = df["customer_region"].replace({"-": np.nan})
df.customer_region.isna().sum()

df.customer_age.isna().sum()

label_encoder = LabelEncoder()
df.loc[df["customer_region"].notna(), "customer_region"] = label_encoder.fit_transform(df.loc[df["customer_region"].notna(), "customer_region"])
df["customer_region"].unique()

label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
label_mapping

num_cols = [col for col in df.columns if df[col].dtype != object and df[col].dtype != bool]
#num_cols = [col for col in num_cols if col not in CUI_log_cols]
#num_cols.remove("total_spent_log")

num_cols.append("customer_region")

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

imputer = KNNImputer(n_neighbors=5)
df[num_cols] = imputer.fit_transform(df[num_cols])
df.isna().sum()

df.customer_region.unique()

df[num_cols] = scaler.inverse_transform(df[num_cols])
df.customer_region.unique()

df["customer_region"] = df["customer_region"].round().astype(int)

label_map_invert = {value: key for key, value in label_mapping.items()}

df["customer_region"] = df["customer_region"].replace(label_map_invert)

df.loc[indexes, "customer_region"].value_counts(normalize=True)

df.loc[index_age, "customer_age"].value_counts(normalize=True)

df.drop(hr_cols, axis=1, inplace=True)
df.drop(dow_cols, axis=1, inplace=True)

df["customer_age"].value_counts()


# ------------------ Feat eng cats 

df["last_promo"] = np.where(df.last_promo == "-", "No promo", df.last_promo)
df["active_customer"] = np.where((df.last_order >= 60), True, False)
df["recent_customer"] = np.where((df.first_order >= 60), True, False)
df["churning_customer"] = np.where((df.last_order < 60), True, False)
df["customer_activity"] = np.where((df.churning_customer == True), "Churning",
                                   np.where((df.recent_customer == True), "Recent", "Active"))
df.drop(columns=["active_customer", "recent_customer", "churning_customer"], inplace=True)


sns.set(style="whitegrid")
cat_cols = [col for col in df.columns if df[col].dtype == object or df[col].dtype == bool]
subset = cat_cols[-5:]
cols = 2
rows = math.ceil(len(subset) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(15, 25))
for ax, feat in zip(axes.flatten(), subset):
    if feat == "most_freq_hour":
        order = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                 "11", "12", "13", "14", "15", "16", "17", "18",
                 "19", "20", "21", "22", "23"]
    elif feat == "most_freq_day":
        order = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    else:
        order = df[feat].value_counts().index
    sns.countplot(x=feat, data=df, ax=ax, order=order)
    if feat == "favorite_cuisine":
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
title = "Categorical Variables Distribution"
plt.suptitle(title)
plt.tight_layout(rect=[0, 0, 1, 0.96])




df.columns[:50]
df.columns[50:]

num_cols = [col for col in df.columns if df[col].dtype != object and df[col].dtype != bool]

cat_cols = [col for col in df.columns if df[col].dtype == object or df[col].dtype == bool]


# ------------------ Outliers removal ------------------

filter = (df["weekend_orders"] < 15)
df_out = df[(filter)]
df.shape[0] - df_out.shape[0] # 46 cases
df_out.shape[0] / df.shape[0] * 100  # 0,2% removed
compare_figure_outliers(df, df_out, ["weekend_orders"])


filter = (df["week_orders"] < 40)
df_out = df[(filter)]
df.shape[0] - df_out.shape[0] # 77 cases
df_out.shape[0] / df.shape[0] * 100  # 0,3% removed

num_cols = [col for col in df.columns if df[col].dtype != object and df[col].dtype != bool]
num_cols = [col for col in num_cols if col not in CUI_log_cols]
num_cols.remove("total_spent_log")
num_cols = [col for col in num_cols if col not in hr_cols and col not in dow_cols and col not in CUI_cols]


filters_man = (
    (df['HR_0_5']<=10)
    &
    (df['HR_6_11']<=15)
    &
    (df['HR_12_17']<=15)
    &
    (df['HR_18_23']<=15))
df_out = df[(filters_man)]
df.shape[0] - df_out.shape[0] # 222 cases 
df_out.shape[0] / df.shape[0] * 100  # 0,7% removed


filters_man = (df['products_per_vendor']<=15)
df_out = df.copy()
df_out = df[(filters_man)]
df.shape[0] - df_out.shape[0] # 24 cases
df_out.shape[0] / df.shape[0] * 100  # 0,1% removed 
compare_figure_outliers(df, df_out, ["products_per_vendor"])

df_out["products_per_vendor"].describe()

filters_man = (
    (df['customer_age']<=60)
    &
    (df['vendor_count']<=25)
    &
    (df['product_count']<=60)
    &
    (df['is_chain'] <= 40)
    &
    (df['total_orders']<=50)
    &
    (df["order_freq"] <= 10)
    &
    (df['avg_time_between_orders']<=40)
    &
    (df["avg_order_value"] <=70)
    &
    (df['total_spent']<=600)
    & 
    (df["total_spent_log"] >= 0)
    & 
    (df["weekend_orders"] <= 15)
    & 
    (df["week_orders"] <= 40)
    & 
    (df['HR_0_5']<=10)
    & 
    (df['HR_6_11']<=20)
    &
    (df['HR_12_17']<=20)
    &
    (df['HR_18_23']<=20)
    &
    (df['products_per_vendor']<=20))

df_original = df.copy()

df_man = df_original[filters_man]

df_original.shape[0] - df_man.shape[0] # 757 cases

df_man.shape[0] / df.shape[0] * 100

df = df_original[(filters_man)]

df.total_spent.describe()


numerical_feats = [col for col in df.columns if df[col].dtype != object and df[col].dtype != bool]

cuisine_percent = [col for col in df.columns if "percent" in col]
cuisine_percent.remove("percentage_chain")

numerical_feats = [col for col in numerical_feats if col not in CUI_cols and col not in hr_cols and col not in dow_cols and col not in cuisine_percent]

numerical_feats[:16]
numerical_feats[16:]
                                         

compare_figure_outliers(df_original, df, numerical_feats[:16])

compare_figure_outliers(df_original, df, numerical_feats[16:])




numericals = [col for col in df.columns if df[col].dtype != object and df[col].dtype != bool]

numericals = [col for col in numericals if col not in CUI_cols]

numericals = [col for col in numericals if col not in cuisine_percent] # not normaly distr


feats_to_drop = ["product_count", "vendor_count", "total_orders", "total_spent", "is_chain"]

df.columns



# ------------------ Multivariate analysis ------------------

# ------------------ Correlation matrix 

# ------------------ High corr  


# AUXILIARY FUNCTION FOR REDUDANT FEATURE REMOVAL

def cross_corr_mean(df_input, corr_coeff=0.95):
    # adjusted from: https://github.com/adityav95/variable_reduction_correlation/blob/master/variable_reduction_by_correlation.ipynb
    # in the original they used pearson correlation here we use spearman
	""" The function retuns a list of features to be dropped from the input features.
	
	INPUTS:
	1. df_input: n input features (pandas dataframe)
	2. corr_coeff: Coefficient threshold (absolute value, no negatives) for a pair of variables above which one of the two will be dropped
	
	NOTICE:
	- The dataframe df_input (should contain only the n input features i.e. no ID and targets) 
	
	SUMMARY OF LOGIC:
	1. The n input variables are taken and a n X n matrix of correlation is created (these are absolute values i.e. a correlation of -0.8 is treated as 0.8)
	2. Variable pairs with correlation higher than the corr_coeff threshold are picked and one of the two variables will be dropped
	3. Which of the two will be dropped is based on the one having lower mean absolute correlation with all other variables 

	"""


	# Generating correlation matrix of input features
	corr_matrix = df_input.corr(method = 'spearman')

	# Generating correlation with the target
	corr_mean = abs(corr_matrix).mean()

	# Preparing data
	features_drop_list = [] # This will contain the list of features to be dropped
	features_index_drop_list = [] # This will contain the index of features to be dropped as per df_input
	corr_matrix = abs(corr_matrix)

	# Selecting features to be dropped (Using two for loops that runs on one triangle of the corr_matrix to avoid checking the correlation of a variable with itself)
	for i in range(corr_matrix.shape[0]):
		for j in range(i+1,corr_matrix.shape[0]):

			# The following if statement checks if each correlation value is higher than threshold (or equal) and also ensures the two columns have NOT been dropped already.  
			if corr_matrix.iloc[i,j]>=corr_coeff and i not in features_index_drop_list and j not in features_index_drop_list:
			
				# The following if statement checks which of the 2 variables with high correlation has a lower correlation with target and then drops it. If equal we can drop any and it drops the first one (This is arbitrary)
				if corr_mean[corr_matrix.columns[i]] >= corr_mean[corr_matrix.columns[j]]:
					features_drop_list.append(corr_matrix.columns[i])	# Name of variable that needs to be dropped appended to list
					features_index_drop_list.append(i)	# Index of variable that needs to be dropped appended to list. This is used to not check for the same variables repeatedly
				else:
					features_drop_list.append(corr_matrix.columns[j])
					features_index_drop_list.append(j)

	return features_drop_list



numericals = [col for col in df.columns if df[col].dtype != object and df[col].dtype != bool]

numericals = [col for col in numericals if col not in CUI_cols]

numericals = [col for col in numericals if col not in cuisine_percent] # not normaly distr

len(numericals)
numericals.append("weekend_percent")

corr_mat = df[numericals].corr(method='spearman')
bot_mat = np.triu(np.ones(corr_mat.shape)).astype(bool)
corr_mat_lower = corr_mat.mask(bot_mat)

high_corr_pairs = []


for col in corr_mat_lower.columns:
    for row in corr_mat_lower.index:
        if abs(corr_mat_lower.loc[row, col]) > 0.2 and row != col:
            high_corr_pairs.append((row, col, corr_mat_lower.loc[row, col]))

high_corr_pairs = sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)
for pair in high_corr_pairs:
    print(f"Pair: {pair[0]} - {pair[1]}, Correlation: {pair[2]:.2f}")


feats_to_drop = ["product_count", "vendor_count", "total_orders", "total_spent", "is_chain"]
# total_orders - week orders 
# product count - week orders ; total_spent 
# vendor_count // product/vendor 

numericals = [col for col in numericals if col not in feats_to_drop]
len(numericals)


# ATE AQUI!!! 
# --- falta arrendondar a idade a integer (int)
# --- ideia agrupar as cozinhas (semantica) - e usar percentagem em vez de CUI log  
# --- falta fazer a normalizacao dados - robust scaler !! / standard scaler

# Cluster -- 3 tipos cluster 
# -- value-based - total_spent log; avg_order_value; / avg_time_between_orders; -- 3 ou 4 (2)
# -- preferences - CUI logs / cuisine percent (e agrupar)  -- 3 ou 4 (2) / dias / hora / percentage_chain / weekend_percent / night_preference_ratio 
# ---- 3 * 3 = 9 
# -- behavior - order frequency ; products per vendor ; weekend preference; hours / avg_time_between_orders / 
# ---  demographic - customer region (CATEGORICA); age ;



# -- outliers 
# -- IQR / z-score 
# -- DBSCAN ! (bom para univariates; ou multivariates)

#-- standard
# agrupar cozinhas 

# feature selection 
# -- Correlacao 




# cluster: kmeans; SOM; hierarchical

# agrupar 

# behavior / preferences /  
# value 

# ------ R2 --- para cluster -- quanto maior R2 ; melhor clusters ! 



# -- K prototypes (mistura de kmeans e k modes) 

    """
    Western Cuisines: CUI_American_log, CUI_Italian_log /
    
    Asian Cuisines: CUI_Asian_log, CUI_Chinese_log, CUI_Indian_log, CUI_Japanese_log, CUI_Thai_log , CUI_Noodle Dishes_log
    
    Street Food and Snacks (OTHER): CUI_Street Food / Snacks_log, CUI_Beverages_log, CUI_Desserts_log / CUI_Cafe_log / CUI_OTHER_log / CUI_Healthy_log, CUI_Chicken Dishes_log
    
    4_ nao tem noodles / deserts 
    8: nao tem cafe / chicken / italian / indian / noodles / other / thai 
    """

# profiling - demographic - media da idade; moda de regiao para cada cluster!!!  

# --- (1) fazer kmeans (alto numero de cluster) - inercia / elbow / silhoute
# --- (2) fazer SOMPs -- com os vectores de SOMPs  
# ---- fazer hierarchical dendogramas para os centroids kmeans 

# PROXIMA MEETING 

# -- Cluster cozinha / agrupar 


df

analyses_df = df[numericals] 










subset = [x for x in numericals if x not in CUI_log_cols]
subset.remove("product_count")
subset.remove("vendor_count")
corr = df[subset].corr(method='spearman')
lower_triangle_mask = np.triu(np.ones(corr.shape)).astype(bool)
cor_mat_lower = corr.mask(lower_triangle_mask)
plt.figure(figsize = (30,20))
sns.heatmap(cor_mat_lower[(abs(cor_mat_lower) >= 0.2)],
            annot=True,
            cmap='PiYG',
            center=0);

df.drop(columns=["is_chain"], inplace=True)
numericals.remove("is_chain")

corr = df[CUI_cols].corr(method='spearman')
lower_triangle_mask = np.triu(np.ones(corr.shape)).astype(bool)
cor_mat_lower = corr.mask(lower_triangle_mask)
plt.figure(figsize = (25,16))
sns.heatmap(cor_mat_lower[abs(cor_mat_lower) > 0.15],
            annot=True, cmap='PiYG',
            center=0)
            #cmap='RdBu_r');
# corr chine e noodle
# corr entre cafe e other


# corr entre dias
corr = df[dow_cols].corr(method='spearman')
lower_triangle_mask = np.triu(np.ones(corr.shape)).astype(bool)
cor_mat_lower = corr.mask(lower_triangle_mask)
plt.figure(figsize = (25,16))
sns.heatmap(cor_mat_lower[abs(cor_mat_lower) > 0.15],
            annot=True,
            cmap='PiYG',
            center=0);
# correlacao ligeiramente maior se comprar no domingo compra na segunda
# se comprar na segunda compra na terca


# ver correlacao entre horas
corr = df[hr_cols].corr(method='pearson')
lower_triangle_mask = np.triu(np.ones(corr.shape)).astype(bool)
cor_mat_lower = corr.mask(lower_triangle_mask)
plt.figure(figsize = (25,16))
sns.heatmap(cor_mat_lower[abs(cor_mat_lower) > 0.15],
            annot=True,
            cmap='PiYG',
            center=0);
# correlacao com horas previas
# correlacao horas de almoco e jantar


# ------------------ Pairplot

sns.pairplot(df[numericals],
             diag_kind='kde',
             markers='o',
             palette='husl',
             plot_kws={'alpha':0.5},
             diag_kws={'alpha':0.5, 'color':'green'},
             corner=True)

subset = [x for x in numericals if x not in hr_cols and x not in CUI_cols and x not in dow_cols and x not in CUI_log_cols]
subset.remove("product_count")
subset.remove("vendor_count")




# ------------------ Jointplot

def plot_jointplot(df, x, feats):
    for feat in feats:
        sns.jointplot(x=x, y=feat, data=df, kind='hex', color='skyblue')
        plt.tight_layout()
        plt.show()
    return

    
plot_jointplot(df, "customer_age", numerical_feats)
    
sns.jointplot(x="customer_age", y="night_preference_ratio", data=df, kind='hex', color='skyblue')

for feat in numerical_feats:
    sns.jointplot(x="night_preference_ratio", y=feat, data=df, kind='hex', color='skyblue')
    plt.tight_layout()
    plt.show()

# ------------------ Categorical vs. Categoricals 




main_cat = "customer_activity"
other_vars = ["last_promo"]
for cat in other_vars:
    (df.groupby([main_cat, cat]).size() / df.groupby([main_cat])[cat].size()).unstack().plot(kind='bar', stacked=True, figsize=(15, 5))
    plt.title(f"{cat} by {main_cat}")
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.show()
    
    
# ------------------ Categorical vs. Numericals

numerical_feats.append("night_preference_ratio")


sns.set(style="whitegrid")
cat = "customer_region"
n_figures = len(numerical_feats)
cols = 2
rows = math.ceil(n_figures / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*5))
for ax, num in zip(axes.flatten(), numerical_feats):
    if df[num].isna().sum() > 0:
        df_copy = df.dropna(subset=[num])
    else:
        df_copy = df.copy()
    num_groups = []
    categories = df_copy[cat].unique()
    for c in categories:
         num_group = df_copy[df_copy[cat] == c][num]
         num_groups.append((c, num_group))
    if cat == "most_freq_day":
        sorted_categories = ["Sun", "Sat", "Fri", "Thu", "Wed", "Tue", "Mon"]
        sorted_num_groups = [df_copy[df_copy[cat] == day][num] for day in sorted_categories]
    elif cat == "most_freq_hour":
        sorted_categories = [str(i) for i in range(23, -1, -1)]
        sorted_num_groups = [df_copy[df_copy[cat] == hour][num] for hour in sorted_categories]
    else:
        num_groups.sort(key=lambda x: x[1].median())
        sorted_categories = [x[0] for x in num_groups]
        sorted_num_groups = [x[1] for x in num_groups]
    ax.boxplot(sorted_num_groups, labels=sorted_categories, patch_artist=True, vert=False)
    ax.set_xlabel(num)
title = f"{cat.capitalize()} vs Continuos Variables Analysis"
plt.suptitle(title)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(wspace=0.4, hspace=0.6)



num_cols = [col for col in df.columns if df[col].dtype != object and df[col].dtype != bool]
num_cols = [col for col in num_cols if col not in CUI_cols and col not in hr_cols and col not in dow_cols and col not in CUI_log_cols]
num_cols.remove("product_count")
num_cols.remove("vendor_count")
num_cols.remove("total_spent")
num_cols.remove("last_order")
num_cols.remove("first_order")

cat_cols = [col for col in df.columns if df[col].dtype == object or df[col].dtype == bool]
cat_cols.remove("last_promo")



sns.set(style="whitegrid")
cat = "customer_region"
n_figures = len(num_cols)
cols = 2
rows = math.ceil(n_figures / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*5))
for ax, num in zip(axes.flatten(), num_cols):
    if df[num].isna().sum() > 0:
        df_copy = df.dropna(subset=[num])
    else:
        df_copy = df.copy()
    num_groups = []
    categories = df_copy[cat].unique()
    for c in categories:
         num_group = df_copy[df_copy[cat] == c][num]
         num_groups.append((c, num_group))
    if cat == "most_freq_day":
        sorted_categories = ["Sun", "Sat", "Fri", "Thu", "Wed", "Tue", "Mon"]
        sorted_num_groups = [df_copy[df_copy[cat] == day][num] for day in sorted_categories]
    elif cat == "most_freq_hour":
        sorted_categories = [str(i) for i in range(23, -1, -1)]
        sorted_num_groups = [df_copy[df_copy[cat] == hour][num] for hour in sorted_categories]
    else:
        num_groups.sort(key=lambda x: x[1].median())
        sorted_categories = [x[0] for x in num_groups]
        sorted_num_groups = [x[1] for x in num_groups]
    ax.boxplot(sorted_num_groups, labels=sorted_categories, patch_artist=True, vert=False)
    ax.set_xlabel(num)
title = f"{cat.capitalize()} vs Continuos Variables Analysis"
plt.suptitle(title)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(wspace=0.4, hspace=0.6)



sns.set(style="whitegrid")
cat = "customer_region"

region_colors = {
    "2370": "skyblue",
    "4660": "lightgreen",
    "8670": "salmon"
}

colors = []
labels = []
CUI_logs = []
for reg in ["2370", "4660", "8670"]:
    for cui in CUI_log_cols:
        data = df[df["customer_region"] == reg][cui].dropna()
        if not data.empty:
            labels.append(f"{reg} - {cui.split('_')[1]}")
            CUI_logs.append(data)
            colors.append(region_colors[reg])

medians = [np.median(log) for log in CUI_logs]

sorted_data = sorted(zip(medians, CUI_logs, labels, colors), key=lambda x: x[0])
medians, CUI_logs, labels, colors = zip(*sorted_data)

fig, ax = plt.subplots(figsize=(15, 11))
boxplots = ax.boxplot(CUI_logs, labels=labels, patch_artist=True, vert=False)
plt.title(f"{cat.capitalize()} vs Continuos Variables")

for patch, color in zip(boxplots['boxes'], colors):
    patch.set_facecolor(color)

plt.tight_layout()
plt.show()




sns.set(style="whitegrid")
cat = "last_promo"
n_figures = len(num_cols)
cols = 2
rows = math.ceil(n_figures / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*5))
for ax, num in zip(axes.flatten(), num_cols):
    if df[num].isna().sum() > 0:
        df_copy = df.dropna(subset=[num])
    else:
        df_copy = df.copy()
    num_groups = []
    categories = df_copy[cat].unique()
    for c in categories:
         num_group = df_copy[df_copy[cat] == c][num]
         num_groups.append((c, num_group))
    if cat == "most_freq_day":
        sorted_categories = ["Sun", "Sat", "Fri", "Thu", "Wed", "Tue", "Mon"]
        sorted_num_groups = [df_copy[df_copy[cat] == day][num] for day in sorted_categories]
    elif cat == "most_freq_hour":
        sorted_categories = [str(i) for i in range(23, -1, -1)]
        sorted_num_groups = [df_copy[df_copy[cat] == hour][num] for hour in sorted_categories]
    else:
        num_groups.sort(key=lambda x: x[1].median())
        sorted_categories = [x[0] for x in num_groups]
        sorted_num_groups = [x[1] for x in num_groups]
    ax.boxplot(sorted_num_groups, labels=sorted_categories, patch_artist=True, vert=False)
    ax.set_xlabel(num)
title = f"{cat.capitalize()} vs Continuos Variables Analysis"
plt.suptitle(title)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(wspace=0.4, hspace=0.6)



# ------------------ 3 vars associations 

# ------------------ Numerical vs. Numerical vs. Categorical

sns.set(style="whitegrid")
n_figures = len(CUI_log_cols)
cols = 2
rows = math.ceil(n_figures / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
for ax, num in zip(axes.flatten(), CUI_log_cols):
    kde = sns.kdeplot(data=df, x="customer_age", y=num, hue="customer_region", ax=ax)
    ax.set_xlabel("Customer Age")
    ax.set_ylabel(num)
    kde.legend_.prop.set_size(6)
    kde.legend_.get_title().set_fontsize(8)
plt.tight_layout(rect=[0, 0, 0.9, 0.96])
plt.subplots_adjust(wspace=0.4, hspace=0.3)




n_figures = len(numerical_feats)
cols = 2
rows = math.ceil(n_figures / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
for ax, num in zip(axes.flatten(), numerical_feats):
    kde = sns.kdeplot(data=df, x="customer_age", y=num, hue="customer_region", ax=ax)
    ax.set_xlabel("Customer Age")
    ax.set_ylabel(num)
    kde.legend_.prop.set_size(6)
    kde.legend_.get_title().set_fontsize(8)
plt.tight_layout(rect=[0, 0, 0.9, 0.96])
plt.subplots_adjust(wspace=0.4, hspace=0.3)


# ------------------ Categorical vs. Categorical vs. Numerical