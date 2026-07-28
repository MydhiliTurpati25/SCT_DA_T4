import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create images folder
os.makedirs("images", exist_ok=True)

# Load Dataset
df = pd.read_csv("data/marketing_campaign.csv", sep="\t")

# Data Cleaning
df.drop_duplicates(inplace=True)
df["Income"].fillna(df["Income"].median(), inplace=True)

# Create New Columns
df["Age"] = 2026 - df["Year_Birth"]
df["Total_Spending"] = (
    df["MntWines"] +
    df["MntFruits"] +
    df["MntMeatProducts"] +
    df["MntFishProducts"] +
    df["MntSweetProducts"] +
    df["MntGoldProds"]
)

print("Dataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nStatistical Summary:\n", df.describe())

# ---------------- Age Distribution ----------------
plt.figure(figsize=(6,4))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.tight_layout()
plt.savefig("images/age_distribution.png", dpi=300)
plt.show()

# ---------------- Income Distribution ----------------
plt.figure(figsize=(6,4))
sns.histplot(df["Income"], bins=20, kde=True)
plt.title("Income Distribution")
plt.tight_layout()
plt.savefig("images/income_distribution.png", dpi=300)
plt.show()

# ---------------- Education Distribution ----------------
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Education")
plt.xticks(rotation=30)
plt.title("Education Distribution")
plt.tight_layout()
plt.savefig("images/education_distribution.png", dpi=300)
plt.show()

# ---------------- Product Spending ----------------
products = [
    "MntWines","MntFruits","MntMeatProducts",
    "MntFishProducts","MntSweetProducts","MntGoldProds"
]

plt.figure(figsize=(7,4))
df[products].sum().plot(kind="bar")
plt.title("Product Spending")
plt.tight_layout()
plt.savefig("images/product_spending.png", dpi=300)
plt.show()

# ---------------- Purchase Channels ----------------
channels = [
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases"
]

plt.figure(figsize=(6,4))
df[channels].sum().plot(kind="bar")
plt.title("Purchase Channels")
plt.tight_layout()
plt.savefig("images/purchase_channels.png", dpi=300)
plt.show()

# ---------------- Campaign Response ----------------
plt.figure(figsize=(5,5))
df["Response"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    labels=["No","Yes"]
)
plt.ylabel("")
plt.title("Campaign Response")
plt.tight_layout()
plt.savefig("images/campaign_response.png", dpi=300)
plt.show()

# ---------------- Correlation Heatmap ----------------
plt.figure(figsize=(10,8))
sns.heatmap(df.select_dtypes(include="number").corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png", dpi=300)
plt.show()

# ---------------- Business Insights ----------------
print("\n========== BUSINESS INSIGHTS ==========")
print("Average Age :", round(df["Age"].mean(),2))
print("Average Income :", round(df["Income"].mean(),2))
print("Average Spending :", round(df["Total_Spending"].mean(),2))
print("Most Common Education :", df["Education"].mode()[0])
print("Most Preferred Purchase Channel :", df[channels].sum().idxmax())
print("Highest Spending Product :", df[products].sum().idxmax())

print("\nProject Completed Successfully!")
print("Charts saved inside the 'images' folder.")