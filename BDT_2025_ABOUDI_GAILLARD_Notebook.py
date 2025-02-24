# Databricks notebook source
# MAGIC %md
# MAGIC #  BIG DATA TOOLS  - Academic Year 2024/2025
# MAGIC ### Group 2 : ABOUDI Arajem , GAILLARD Paul

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Importations**

# COMMAND ----------

# DBTITLE 1,Importations
# LIBRARY IMPORTATIONS
from pyspark.sql.functions import *
from pyspark.sql.functions import isnan, when, count, col, variance, count
from pyspark.sql.functions import col, current_date, datediff, max
from pyspark.sql.window import Window
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql.functions import to_timestamp 
from pyspark.sql.functions import col, to_timestamp
from pyspark.sql.functions import isnan, when, count, col
from pyspark.ml import Pipeline
from pyspark.ml.feature import RFormula, StandardScaler, ChiSqSelector
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import col
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import pandas as pd


# FILE IMPORTATION

# Product Table
file_path = "/FileStore/products.csv"
products = spark.read.csv(file_path, header=True, inferSchema=True)
products.write.format("csv")\
  .mode("overwrite")\
  .save("/data_project/products.csv")

# Order Table
file_path = "/FileStore/orders.csv"
orders = spark.read.csv(file_path, header=True, inferSchema=True)
orders.write.format("csv")\
  .mode("overwrite")\
  .save("/data_project/products.csv")

# Item Table 
file_path = "/FileStore/order_items.csv"
items = spark.read.csv(file_path, header=True, inferSchema=True)

# Payment Table
file_path = "/FileStore/order_payments.csv"
payments = spark.read.csv(file_path, header=True, inferSchema=True)

# Review Table
file_path = "/FileStore/order_reviews.csv"
reviews = spark.read.csv(file_path, header=True, inferSchema=True)

# COMMAND ----------

# DBTITLE 1,Function Deployment
### All the functions needed to be deployed

def detect_outliers_and_plot(df, columns):
    """
    Detect outliers using the IQR method and visualize the columns using boxplots.

    Args:
        df (DataFrame): Spark DataFrame to analyze.
        Columns (list): List of numerical column names to check for outliers.
    """
    for column in columns:
        print(f"\nAnalyzing column: {column}")
        
        # Calculate Q1, Q3, and IQR
        quantiles = df.approxQuantile(column, [0.25, 0.75], 0.05) 
        q1, q3 = quantiles[0], quantiles[1]
        #iqr = q3 - q1
        lower_bound = q1 
        upper_bound = q3

        # Count the number of outliers
        outliers = df.filter((col(column) < lower_bound) | (col(column) > upper_bound))
        outlier_count = outliers.count()
        
        print(f"Q1: {q1}, Q3: {q3}")
        print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
        print(f"Number of outliers: {outlier_count}")
        
        # Convert column to Pandas for visualization
        pandas_df = df.select(column).toPandas()
        
        # Boxplot Visualization
        plt.figure(figsize=(8, 6))
        plt.boxplot(pandas_df[column].dropna(), vert=False)
        plt.title(f'Boxplot for {column}')
        plt.xlabel(column)
        plt.grid(True)
        plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ## Product Table Part

# COMMAND ----------

# DBTITLE 1,Exploration Tables
# Analysis of the variables in the Product Table
products.printSchema()
# Exploration of the table
products.show(5,False)
products.describe().show()


# COMMAND ----------

# DBTITLE 1,Null Values Exploration
#Count the number of nulls per column
products.select([count(when(col(c).isNull(), c)).alias(c) for c in products.columns]).show()

# COMMAND ----------

# DBTITLE 1,Categorical Variables Analysis
#Count the number of distinct 'product_category_name' that we have
print('Nbr of distinct product category name:')
count_prod_category_name=products.select(col("product_category_name")).distinct().count()
print(count_prod_category_name)

#Count the number of distinct "product_id" that we have
print('Nbr of distinct product id:')
count_productID=products.select(col("product_id")).distinct().count()
print(count_productID)

# COMMAND ----------

# DBTITLE 1,Feautres creation
#Add some features 
products=products\
    .withColumn("product_volume",col("product_length_cm") * col("product_width_cm") * col("product_height_cm"))\
    .withColumn("size_to_weight_ratio",col("product_volume") / col("product_weight_g"))\
    .withColumn("product_density", col("product_weight_g") / col("product_volume"))

# COMMAND ----------

# DBTITLE 1,Null / Duplicates / Verification
# Drop Null Raws with nulls values
products = products.na.drop("any")
# Drop Duplicates
products = products.dropDuplicates()
products.count()
# Verification of the nb of Nulls per column
products.count()
products.select([count(when(col(c).isNull(), c)).alias(c) for c in products.columns]).show()

# COMMAND ----------

# DBTITLE 1,Product Category Distribution Visualisation
#The distribution of 'product_category_name' in the products table
products_pd = products.select("product_category_name").toPandas()
category_counts = products_pd["product_category_name"].value_counts()

# Plot a bar chart
plt.figure(figsize=(12, 6))
category_counts.plot(kind="bar", color="skyblue", edgecolor="black")
plt.xlabel("Product Category")
plt.ylabel("Count")
plt.title("Distribution of Product Categories")
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ## Order Table Part

# COMMAND ----------

# DBTITLE 1,Order Table Exploration
# Analysis of the variables in the Order Table
orders.printSchema()
# Exploration of the table
orders.show(5,False)
orders.describe().show()

# COMMAND ----------

# DBTITLE 1,Null Values Exploration
#Count the number of nulls per column
orders.select([count(when(col(c).isNull(), c)).alias(c) for c in orders.columns]).show()

# COMMAND ----------

# DBTITLE 1,To Timestamp Convertion
#Initialize Columns to convert
columns_to_convert = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]
#Convertion
for column in columns_to_convert:
    orders = orders.withColumn(column, to_timestamp(col(column), "yyyy-MM-dd HH:mm:ss"))
#Verification
orders.printSchema()

# COMMAND ----------

# DBTITLE 1,NULL/ Duplicates / Verification
# Drop Null Raws with nulls values
orders = orders.na.drop("any")
# Drop Duplicates
orders=orders.dropDuplicates()
# Verification of the nb of Nulls per column
orders.select([count(when(col(c).isNull(), c)).alias(c) for c in orders.columns]).show()

# COMMAND ----------

# DBTITLE 1,Date/TimeStamp Features Creation
# Count orders per customer
customer_order_counts = orders.groupBy("customer_id").agg(
    count("order_id").alias("order_count"),
    min("order_purchase_timestamp").alias("first_order_date")
)

# Get the most recent purchase date for each customer
last_order_date_per_customer = orders.groupBy("customer_id") \
                                        .agg(max(col("order_purchase_timestamp")).alias("last_order_date"))

# Calculate the time since the last purchase
last_order_date_per_customer = last_order_date_per_customer.withColumn(
    "recency",
    datediff(current_date(), col("last_order_date"))
)

last_order_date_per_customer.select("customer_id", "last_order_date", "recency").show()
# Join the created data frame to orders
orders=orders.join(last_order_date_per_customer, on="customer_id", how="left")
# Join back to orders and add repeat customer features
# When we use max("order_purchase_timestamp") over the window, it will include the current order
orders = orders.join(customer_order_counts, "customer_id","left")
orders = orders.withColumn(
    "is_repeated_customer", 
    when(col("order_count") > 1, 1).otherwise(0)
).withColumn("delivery_duration", datediff(col("order_delivered_customer_date"), col("order_purchase_timestamp"))) \
                     .withColumn("approval_duration", datediff(col("order_approved_at"), col("order_purchase_timestamp"))) \
                     .withColumn("delivery_delay", datediff(col("order_estimated_delivery_date"), col("order_delivered_customer_date")))
orders.printSchema()

# COMMAND ----------

# DBTITLE 1,Last Verifications
# Last verifications 
orders.show(3)
print('Distinct order id :')
orders.select(col("order_id")).distinct().count()
print('Number of rows:')
orders.count()

#Count the number of nulls per column
orders.select([count(when(col(c).isNull(), c)).alias(c) for c in orders.columns]).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ## Item Table Part

# COMMAND ----------

# DBTITLE 1,Item Table Exploration
# Analysis of the variables in the Item Table
items.printSchema()
# Exploration of the table
items.show(5,False)
items.describe().show()

# COMMAND ----------

# DBTITLE 1,Cost Feature Creation
# Item_total_cost creation & # shipping_to_price_ratio creation
items=items\
    .withColumn("item_total_cost",col("price")+col("shipping_cost"))\
        .withColumn("shipping_to_price_ratio",col("shipping_cost")/col("price"))

# COMMAND ----------

# DBTITLE 1,Features Creation Item Table
#Since order_items and order_payments have multiple rows per order_id, we aggregate them before joining.
#Aggregate items at the order_id level

items_agg = items.groupBy("order_id").agg(
    count("order_item_id").alias("nbr_items"),
    first("product_id").alias("product_id"),  # Keeps only one product ID per order
    min("price").alias("min_price"),
    max("price").alias("max_price"),
    avg("price").alias("avg_price"),
    sum("price").alias("total_price"),
    min("shipping_cost").alias("min_shipping_cost"),
    max("shipping_cost").alias("max_shipping_cost"),
    avg("shipping_cost").alias("avg_shipping_cost"),
    sum("shipping_cost").alias("total_shipping_cost"),
    min("item_total_cost").alias("min_item_total_cost"),
    max("item_total_cost").alias("max_item_total_cost"),
    avg("item_total_cost").alias("avg_item_total_cost"),
    sum("item_total_cost").alias("total_item_total_cost"),
    min("shipping_to_price_ratio").alias("min_shipping_to_price_ratio"),
    max("shipping_to_price_ratio").alias("max_shipping_to_price_ratio"),
    avg("shipping_to_price_ratio").alias("avg_shipping_to_price_ratio"),
    sum("shipping_to_price_ratio").alias("total_shipping_to_price_ratio")
)

items_agg.show(3)

# COMMAND ----------

# DBTITLE 1,NULL/Duplicates/Verification
# Drop Null Raws with nulls values
items_agg=items_agg.na.drop("any")
# Drop Duplicates
items_agg = items_agg.dropDuplicates()
# Verification of the nb of Nulls per column
items_agg.select([count(when(col(c).isNull(), c)).alias(c) for c in items_agg.columns]).show()

# COMMAND ----------

# DBTITLE 1,Last Verification
#Last verification
items_agg.count()
#Count the number of distinct products per order 
print('Nbr of distinct products per order  :')
count_orderID=items_agg.select(col("product_id")).distinct().count()
print(count_orderID)

# COMMAND ----------

# DBTITLE 1,Joining with products table/ Verification
items_product = items_agg.join(products,"product_id","left")
#Verification
items_product.show(5)
items_product.count()
items_product.printSchema()

# COMMAND ----------

# DBTITLE 1,Drop NA before Feature Creation
items_product=items_product.na.drop("any")
items_product = items_product.dropDuplicates()
items_product.select([count(when(isnull(c) | isnan(c), c)).alias(c) for c in items_product.columns]).show()

# COMMAND ----------

# DBTITLE 1,Feature Creation Item_Product Table
# Group by order_id and perform multiple aggregations
items_product = items_product.groupBy("order_id").agg(
    avg("product_name_lenght").alias("avg_name_length"),
    avg("product_description_lenght").alias("avg_description_length"),
    sum("product_photos_qty").alias("total_photos"),
    sum("product_weight_g").alias("total_weight_g"),
    sum("product_volume").alias("total_volume"),
    avg("product_volume").alias("avg_product_volume_per_order"),
    sum("product_density").alias("total_product_density"),
    avg("product_density").alias("avg_product_density"),
    concat_ws(", ", collect_set("product_category_name")).alias("category"),  # Concatenated categories
    countDistinct("product_category_name").alias("distincy_category_count")  # Count of unique categories
)

items_product.show(truncate=False)

# COMMAND ----------

# DBTITLE 1,Last Verification
items_product.select(col("order_id")).distinct().count()
items_product.select([count(when(isnull(c) | isnan(c), c)).alias(c) for c in items_product.columns]).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ### Payment Table 

# COMMAND ----------

# DBTITLE 1,Exploration Payment Table
# Analysis of the variables in the Order Table
payments.printSchema()
# Exploration of the table
payments.show(5,False)
payments.describe().show()

# COMMAND ----------

# DBTITLE 1,Feature Creation Payment Table
# Categorical :
payments_sorted = payments.orderBy("order_id")

print('********* installment_value computing **********') # = payment_value / payment_installments
payments = payments.withColumn(
    "installment_value",
    when(col("payment_installments") > 0, col("payment_value") / col("payment_installments"))
    .otherwise(None)  # Avoid division by zero
)

print('********* monetary_value computing **********')
order_total_value = payments.groupBy("order_id") \
                             .agg(sum(col("payment_value")).alias("monetary_value"))

print('********* joining monetary_value to payments **********')
payments = payments.join(order_total_value, on="order_id", how="left")

print('********* Identifying all the possible values on payment_type **********')
distinct_values = payments.select("payment_type").distinct().rdd.flatMap(lambda x: x).collect()
print(distinct_values)

print('********* creation of the dummy **********')
for payment in distinct_values:
    payments = payments.withColumn(f"dummy_{payment}", when(col("payment_type") == payment, 1).otherwise(0))

print('********* Aggregation on order_id by keeping the max of each dummy **********')
dummy_cols = ["dummy_mobile", "dummy_credit_card", "dummy_voucher", "dummy_debit_card"]
payments_agg = payments.groupBy("order_id").agg(
    *[max(col).alias(col) for col in dummy_cols],
    max("payment_sequential").alias('total_payment_sequential'),
    sum("payment_value").alias("total_payment_value"),
    avg("payment_installments").alias("avg_payment_installments"),
    max("payment_installments").alias("max_payment_installments")
)

print('********* show the final table :  **********')
payments_agg.show(20,truncate=False)

print('********* Final Table analysis (Describe):  **********')
payments.printSchema()

# COMMAND ----------

# DBTITLE 1,NULL/Duplicates/Verification
# Drop Null Raws with nulls values
payments = payments.na.drop("any")
# Drop Duplicates
payments  = payments.dropDuplicates()
# Verification of the nb of Nulls per column
payments.select([count(when(col(c).isNull(), c)).alias(c) for c in payments.columns]).show()

# COMMAND ----------

# DBTITLE 1,Last Verification
# Last verifications 
orders.show(3)
print('Distinct order id :')
print(payments.select(col("order_id")).distinct().count())
print('Number of rows:')
payments.count()

# COMMAND ----------

# DBTITLE 1,Payment Methods Histogram Visualisation
# Count the occurrences of each payment type
payment_counts = payments.groupBy("payment_type").count().toPandas()

# Plot the bar chart
plt.figure(figsize=(8, 5))
plt.bar(payment_counts["payment_type"], payment_counts["count"], color='skyblue')
plt.xlabel("Payment Type")
plt.ylabel("Count")
plt.title("Distribution of Payment Types")
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# DBTITLE 1,Payment Methods Pie-Chart Visualisation
# Convert payment type count to Pandas DataFrame
payment_counts = payments.groupBy("payment_type").count().toPandas()

# Define different shades of blue for better visualization
colors = ['#1f77b4', '#aec7e8', '#6baed6', '#3182bd', '#08519c']

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(payment_counts["count"], labels=payment_counts["payment_type"], autopct='%1.1f%%', 
        colors=colors)

plt.title("Distribution of Payment Types")

# Show the pie chart
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Review Table**

# COMMAND ----------

# DBTITLE 1,Exploration ReviewTable
#Exploration Review Table
# Analysis of the variables in the Review Table
reviews.printSchema()
# Exploration of the table

print('Overview :')
reviews.show(5,False)
print('Statistics : ')
reviews.describe().show()
print('Number of line :')
reviews.count()

# COMMAND ----------

# DBTITLE 1,Processing Date Variables to TimeStamp
reviews= reviews.withColumn("review_answer_timestamp",to_timestamp(col("review_answer_timestamp"),"yyyy-MM-dd HH:mm:ss"))
reviews = reviews.withColumn("review_creation_date", to_timestamp(col("review_creation_date"), "yyyy-MM-dd HH:mm:ss"))

# COMMAND ----------

# DBTITLE 1,Filtering the Review Table
# Find the most recent review_answer_timestamp per order_id
df_latest = reviews.groupBy("order_id").agg(max("review_answer_timestamp").alias("latest_timestamp"))
# Keeping only the last review done per customers
reviews=reviews.join(df_latest,"order_id","left")

# COMMAND ----------

# DBTITLE 1,Review Scores Distribution Visualisation
df = reviews.select("review_score").dropna()
# Convert to Pandas for visualization
review_pd = df.toPandas()

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(review_pd['review_score'], bins=5, edgecolor='black', alpha=0.7, align='left')
plt.xticks([1, 2, 3, 4, 5])
plt.xlabel("Review Score")
plt.ylabel("Frequency")
plt.title("Distribution of Review Scores")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

# COMMAND ----------

# DBTITLE 1,Feature Creation Review Table
# Assign "positive" for scores 4-5 and "negative" for scores 1-3
reviews = reviews.withColumn("review",when((col("review_score") >= 4), 1).otherwise(0)).drop("review_score")

print('*** Computing review responsiveness ****')
reviews = reviews.withColumn("review_responsiveness", 
                             (col("review_answer_timestamp").cast("long") - col("review_creation_date").cast("long")) / 86400) # (:Seconds)



# COMMAND ----------

# DBTITLE 1,NULL/Duplicates
# Drop Null Raws with nulls values
reviews = reviews.na.drop("any")
# Drop Duplicates
reviews  = reviews.dropDuplicates()
# Verification of the nb of Nulls per column
reviews.select([count(when(col(c).isNull(), c)).alias(c) for c in reviews.columns]).show()

# COMMAND ----------

# DBTITLE 1,Last Verification
# Last verifications 
reviews.show()

print('Distinct order id :')
print(reviews.select(col("order_id")).distinct().count())
print('Number of rows:')
reviews.count()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Basetable**

# COMMAND ----------

# DBTITLE 1,Basetable Creation
# Basetable cration by joining the differents tables.
basetable = orders.join(items_product,"order_id","left")\
    .join(payments,"order_id","left")\
    .join(reviews,"order_id","left")

basetable.printSchema()

# COMMAND ----------

# DBTITLE 1,BaseTable Cleaning
# Drop Null Raws with nulls values
basetable=basetable.na.drop("any")
# Drop Duplicates
basetable=basetable.dropDuplicates()

# Droping unwhished features
basetable = basetable.drop("order_purchase_timestamp","order_approved_at","order_delivered_carrier_date","order_delivered_customer_date","order_estimated_delivery_date","payment_installments", "review_creation_date", "review_answer_timestamp","product_id","customer_id","order_status","review_id",'review_responsiveness','delivery_delay')


# Verification of the nb of Nulls per column
basetable.select([count(when(col(c).isNull(), c)).alias(c) for c in basetable.columns]).show()

# COMMAND ----------

# DBTITLE 1,Basetable Verification
# Display Final Table
display(basetable)

print('Distinct order id :')
print(basetable.select(col("order_id")).distinct().count())
print('Number of rows:')
basetable.count()

# COMMAND ----------

# DBTITLE 1,Reviews Distrubution (Postive/Negative)
df = basetable.select("review").dropna()

# Convert to Pandas for visualization
review_pd = df.toPandas()

# Count occurrences of each review type (1 = positive, 0 = negative)
review_counts = review_pd['review'].value_counts()

# Create a bar chart
plt.figure(figsize=(6, 5))
plt.bar(review_counts.index.astype(str), review_counts.values, color=['skyblue', 'steelblue'])
plt.xlabel("Review (1 = Positive, 0 = Negative)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Distribution of Reviews", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

# COMMAND ----------

# DBTITLE 1,Review Distribution (Postive/Negative) For Repeated Customer
# Count occurrences of review scores for repeated and non-repeated customers
review_counts = basetable.groupBy("is_repeated_customer", "review").count().toPandas()

# Pivot the table to get counts for 0 (negative) and 1 (positive) reviews
review_pivot = review_counts.pivot(index="is_repeated_customer", columns="review", values="count").fillna(0)

# Bar chart creation
review_pivot.plot(kind="bar", figsize=(7, 5), color=["skyblue", "steelblue"], edgecolor='black')

#Plot
plt.xlabel("Repeated Customer (0 = No, 1 = Yes)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Review Distribution by Repeated Customers", fontsize=14)
plt.xticks(rotation=0)
plt.legend(["Negative Review (0)", "Positive Review (1)"], title="Review", fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Features Analysis**

# COMMAND ----------

# DBTITLE 1,Correlation Matrix
# CORRELATION MATRIX

# Selecting numerical columns from the basetable
numerical_cols = [
    "recency", "order_count",  "approval_duration", 
    "delivery_delay", "avg_name_length", "avg_description_length", "total_photos", 
    "total_volume", "avg_product_volume_per_order", 
    "total_product_density", "avg_product_density", "payment_value", 
    "installment_value", "review", "review_responsiveness"
]

# Vector assembler to create feature vector
vector_col = "features"
assembler = VectorAssembler(inputCols=numerical_cols, outputCol=vector_col)
df_vector = assembler.transform(basetable).select(vector_col)

# Compute the correlation matrix
correlation_matrix = Correlation.corr(df_vector, vector_col).head()[0]

# Convert to pandas dataframe for visualization
corr_matrix_pd = pd.DataFrame(correlation_matrix.toArray(), index=numerical_cols, columns=numerical_cols)

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix_pd, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix of Features")
plt.show()


# COMMAND ----------

# DBTITLE 1,VIF (Variance Inflation Factor)
# Variance Inflation Factor

# Numeric features list
features = [
    "order_count",  "approval_duration", 
    "delivery_delay", "avg_description_length", "total_photos", 
    "total_volume", "avg_product_volume_per_order", 
    "total_product_density", "avg_product_density", "payment_value", 
    "installment_value", "review", "review_responsiveness"]

# VectorAssembler to group features in one variable
vector_assembler = VectorAssembler(inputCols=features, outputCol="features")
df_vif = vector_assembler.transform(basetable).select("features")

# Convert in Pandas DF
df_pandas = basetable.select(features).toPandas()

# Compute VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = df_pandas.columns
vif_data["VIF"] = [variance_inflation_factor(df_pandas.values, i) for i in range(df_pandas.shape[1])]

# Print results
print(vif_data.sort_values(by="VIF", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Pipeline Creation**

# COMMAND ----------

# DBTITLE 1,Pipeline
# RFormula: handles categorical and numerical features
formula = RFormula(formula="review ~ .-order_id", 
                   featuresCol="features", 
                   labelCol="label",
                   handleInvalid="skip")

# StandardScaler: Normalizes numerical features
scaler = StandardScaler(inputCol="features", 
                        outputCol="scaledFeatures")

# ChiSqSelector: Selects a subset of features after scaling
selector = ChiSqSelector(numTopFeatures=15, 
                         featuresCol="scaledFeatures",
                         outputCol="features")

# Create a Pipeline
pipeline = Pipeline(stages =
                    [formula,
                     #selector,
                     scaler])

# COMMAND ----------

# DBTITLE 1,Pipeline fit
# Last Step to get our Final table ready for modeling
final_table = pipeline.fit(basetable).transform(basetable).select("features","label")
# Verification
final_table.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### *Data Splitting*

# COMMAND ----------

# DBTITLE 1,Splitting & Repartition
#Create a train and test set with a 70% train, 30% test split
train, test = final_table.randomSplit([0.7, 0.3], seed=123)
'''
# Sample of the Data to evaluate faster :
test = test.sample(fraction=0.5, seed=42) 
train = train.sample(fraction=0.5, seed=42) 
'''
# Repartition and memory cache for efficiency
train = train.repartition(15)
train.cache()

print(f'Basetable :{final_table.count()}')
print(f'Train Set :{train.count()}')
print(f'Test Set :{test.count()}')

# COMMAND ----------

# MAGIC %md
# MAGIC ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Modelisation Part :**

# COMMAND ----------

# DBTITLE 1,Classification / Evaluation-Metrics / Tuning IMPORTATION
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, LinearSVC, GBTClassifier

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from sklearn.metrics import roc_curve, auc, log_loss

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder 

# COMMAND ----------

# MAGIC %md
# MAGIC ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Logistic Regression**

# COMMAND ----------

# DBTITLE 1,Logistic Regression Training
#Create logistic regression model to data train
lr_model = LogisticRegression().fit(train)

#Transform model to data test
lr_result = lr_model.transform(test)

#View Result
lr_result.show(5)

# COMMAND ----------

# DBTITLE 1,LR - Evaluation
#Evaluation of the model
lr_eval = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="label")
lr_eval2= MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")

#AUC
lr_AUC  = lr_eval.evaluate(lr_result)

#Accuracy
lr_ACC  = lr_eval2.evaluate(lr_result, {lr_eval2.metricName:"accuracy"})

#ROC Grafik
#Create ROC grafic from lr_result
PredAndLabels           = lr_result.select("probability", "label")
PredAndLabels_collect   = PredAndLabels.collect()
PredAndLabels_list      = [(float(i[0][0]), 1.0-float(i[1])) for i in PredAndLabels_collect]
PredAndLabels           = sc.parallelize(PredAndLabels_list)

metrics = BinaryClassificationMetrics(PredAndLabels)

# ROC
FPR = dict()                                                        # FPR: False Positive Rate
tpr = dict()                                                        # TPR: True Positive Rate
roc_auc = dict()
 
y_test = [i[1] for i in PredAndLabels_list]
y_score = [i[0] for i in PredAndLabels_list]
 
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
 
# Plot GRAPHIC
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc="lower right")
plt.show()

# Area under ROC
print("Logistic Regression Area Under ROC")
print("Area under ROC = %.2f" % metrics.areaUnderROC)

print("Logistic Regression Performance Measure")
print("Accuracy = %0.2f" % lr_ACC)
print("AUC = %.2f" % lr_AUC)

# COMMAND ----------

# DBTITLE 1,LR - Confusion Matrix
#CONFUSION MATRIX LR 
cm_lr_result = lr_result.crosstab("prediction", "label")
cm_lr_result = cm_lr_result.toPandas()

print(cm_lr_result)

# COMMAND ----------

# DBTITLE 1,LR - Evaluation Metrics
# Calculate Accuracy, Sensitivity, Specificity, Precision
TP = cm_lr_result["1.0"][0]
FP = cm_lr_result["0.0"][0]
TN = cm_lr_result["0.0"][1]
FN = cm_lr_result["1.0"][1]
Accuracy = (TP+TN)/(TP+FP+TN+FN)
Sensitivity = TP/(TP+FN)
Specificity = TN/(TN+FP)
Precision = TP/(TP+FP)

print ("Accuracy = %0.2f" %Accuracy )
print ("Sensitivity = %0.2f" %Sensitivity )
print ("Specificity = %0.2f" %Specificity )
print ("Precision = %0.2f" %Precision )

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Logistic Regression  -- HYPER-PARAMETER Tuning**

# COMMAND ----------

# DBTITLE 1,LR - Hyperparameter Tuning
#Logistic Regression With Hyper-Parameter Tuning
#Define logistic regression model
lr_hyper=LogisticRegression(featuresCol='features', labelCol='label')

#Hyper-Parameter Tuning
paramGrid_lr = ParamGridBuilder() \
    .addGrid(lr_hyper.regParam, [0.1, 0.01]) \
    .addGrid(lr_hyper.elasticNetParam, [0.8, 0.7]) \
    .build()
crossval_lr = CrossValidator(estimator=lr_hyper,
                             estimatorParamMaps=paramGrid_lr,
                             evaluator=BinaryClassificationEvaluator(),
                             numFolds=3)
#fit model to data train
lr_model_hyper = crossval_lr.fit(train)

#Transform model to data test
lr_result_hyper = lr_model_hyper.transform(test)

#View Result
lr_result_hyper.show(5)

# COMMAND ----------

# DBTITLE 1,Evaluation
#Logistic Regression With Hyper-Parameter Tuning Evaluation
#Evaluate model by checking accuracy and AUC value
lr_hyper_eval = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="label")
lr_hyper_eval2= MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")
lr_hyper_AUC  = lr_hyper_eval.evaluate(lr_result_hyper)
lr_hyper_ACC  = lr_hyper_eval2.evaluate(lr_result_hyper, {lr_hyper_eval2.metricName:"accuracy"})

#ROC Grafik
PredAndLabels           = lr_result_hyper.select("probability", "label")
PredAndLabels_collect   = PredAndLabels.collect()
PredAndLabels_list      = [(float(i[0][0]), 1.0-float(i[1])) for i in PredAndLabels_collect]
PredAndLabels           = sc.parallelize(PredAndLabels_list)

metrics = BinaryClassificationMetrics(PredAndLabels)

# Visualization
FPR = dict()                                                        # FPR: False Positive Rate
tpr = dict()                                                        # TPR: True Positive Rate
roc_auc = dict()
 
y_test = [i[1] for i in PredAndLabels_list]
y_score = [i[0] for i in PredAndLabels_list]
 
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# Plot Matrice
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc="lower right")
plt.show()

# Area under ROC
print("Logistic Regression Area Under ROC")
print("Area under ROC = %.2f" % metrics.areaUnderROC)
# AUC Accuracy
print("Logistic Regression Performance Measure")
print("Accuracy = %0.2f" % lr_hyper_ACC)
print("AUC = %.2f" % lr_hyper_AUC)



# COMMAND ----------

#CONFUSION MATRIX LR HYPER PARAMETERS
cm_lr_result_hyper = lr_result_hyper.crosstab("prediction", "label")
cm_lr_result_hyper = cm_lr_result_hyper.toPandas()
print(cm_lr_result_hyper)

# COMMAND ----------

# MAGIC %md
# MAGIC ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Decision Tree**

# COMMAND ----------

# DBTITLE 1,Decision Tree - Training
#Create decision tree model to data train
dt=DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 10, maxBins=128)
dt_model = dt.fit(train)

##Transform model to data test
dt_result = dt_model.transform(test)

#View Result
dt_result.show(5)

# COMMAND ----------

# DBTITLE 1,Decision Tree - Evaluation
# Decision Tree Evaluation
# Evaluate model by calculating accuracy and area under curve (AUC)
dt_eval = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="label")
dt_eval2= MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")
dt_AUC  = dt_eval.evaluate(dt_result)
dt_ACC  = dt_eval2.evaluate(dt_result, {dt_eval2.metricName:"accuracy"})

# ROC Grafic
PredAndLabels           = dt_result.select("probability", "label")
PredAndLabels_collect   = PredAndLabels.collect()
PredAndLabels_list      = [(float(i[0][0]), 1.0-float(i[1])) for i in PredAndLabels_collect]
PredAndLabels           = sc.parallelize(PredAndLabels_list)

metrics = BinaryClassificationMetrics(PredAndLabels)

# Visualization
FPR = dict()                                                        # FPR: False Positive Rate
tpr = dict()                                                        # TPR: True Positive Rate
roc_auc = dict()
 
y_test = [i[1] for i in PredAndLabels_list]
y_score = [i[0] for i in PredAndLabels_list]
 
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
 
#Plot ROC
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.legend(loc="lower right")
plt.show()

# Area under ROC
print("Decision Tree Area Under ROC")
print("Area under ROC = %.2f" % metrics.areaUnderROC)

print("Decision Tree Performance Measure")
print("Accuracy = %0.2f" % dt_ACC)
print("AUC = %.2f" % dt_AUC)


# COMMAND ----------

# DBTITLE 1,Decision Tree - Confusion Matrix
#CONFUSION MATRIX DT 
cm_dt_result = dt_result.crosstab("prediction", "label")
cm_dt_result = cm_dt_result.toPandas()
print(cm_dt_result)

# COMMAND ----------

# Calculate Accuracy, Sensitivity, Specificity, Precision
TP = cm_dt_result["1.0"][0]
FP = cm_dt_result["0.0"][0]
TN = cm_dt_result["0.0"][1]
FN = cm_dt_result["1.0"][1]
Accuracy = (TP+TN)/(TP+FP+TN+FN)
Sensitivity = TP/(TP+FN)
Specificity = TN/(TN+FP)
Precision = TP/(TP+FP)

print ("Accuracy = %0.2f" %Accuracy )
print ("Sensitivity = %0.2f" %Sensitivity )
print ("Specificity = %0.2f" %Specificity )
print ("Precision = %0.2f" %Precision )

# COMMAND ----------

# MAGIC %md
# MAGIC ####**DECISION TREE HYPER-PARAMETER TUNING**

# COMMAND ----------

# DBTITLE 1,Decision tree with HyperGrid - Training
# Decision Tree With Hyper-Parameter Tuning
# Define decision tree model
dt_hyper=DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', impurity='gini')

# Hyper-Parameter Tuning
paramGrid_dt = ParamGridBuilder() \
    .addGrid(dt_hyper.maxDepth, [10, 20]) \
    .addGrid(dt_hyper.maxBins, [32,64]) \
    .build()
crossval_dt = CrossValidator(estimator=dt_hyper,
                             estimatorParamMaps=paramGrid_dt,
                             evaluator=BinaryClassificationEvaluator(),
                             numFolds=5)
#fit model to data train
dt_model_hyper = crossval_dt.fit(train)

#transform model to data test
dt_result_hyper = dt_model_hyper.transform(test)

#View Result
dt_result_hyper.show(5)

# COMMAND ----------

# DBTITLE 1,Decision Tree Hyperparams- Evaluation
#Decision Tree With Hyper-Parameter Tuning Evaluation
#Evaluate model by calculating accuracy and area under curve (AUC)
dt_hyper_eval = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="label")
dt_hyper_eval2= MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")
dt_hyper_AUC  = dt_hyper_eval.evaluate(dt_result_hyper)
dt_hyper_ACC  = dt_hyper_eval2.evaluate(dt_result_hyper, {dt_hyper_eval2.metricName:"accuracy"})


#ROC Grafic
PredAndLabels           = dt_result_hyper.select("probability", "label")
PredAndLabels_collect   = PredAndLabels.collect()
PredAndLabels_list      = [(float(i[0][0]), 1.0-float(i[1])) for i in PredAndLabels_collect]
PredAndLabels           = sc.parallelize(PredAndLabels_list)

metrics = BinaryClassificationMetrics(PredAndLabels)

# Visualization
FPR = dict()                                                        # FPR: False Positive Rate
tpr = dict()                                                        # TPR: True Positive Rate
roc_auc = dict()
 
y_test = [i[1] for i in PredAndLabels_list]
y_score = [i[0] for i in PredAndLabels_list]
 
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

#Plot ROC Graphic
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.legend(loc="lower right")
plt.show()


# Area under ROC
print("Decision Tree Area Under ROC")
print("Area under ROC = %.2f" % metrics.areaUnderROC)

print("Decision Tree Performance Measure")
print("Accuracy = %0.2f" % dt_hyper_ACC)
print("AUC = %.2f" % dt_hyper_AUC)


# COMMAND ----------

# DBTITLE 1,Decision Tree Hyperparams - Confusion Matrix
#CONFUSION MATRIX DT HYPER- PARAMETERS
cm_dt_result_hyper = dt_result_hyper.crosstab("prediction", "label")
cm_dt_result_hyper = cm_dt_result_hyper.toPandas()
print(cm_dt_result_hyper)

# COMMAND ----------

# DBTITLE 1,Other Evaluations Metrics
# Calculatation of Accuracy, Sensitivity, Specificity and Precision
TP = cm_dt_result_hyper["1.0"][0]
FP = cm_dt_result_hyper["0.0"][0]
TN = cm_dt_result_hyper["0.0"][1]
FN = cm_dt_result_hyper["1.0"][1]
Accuracy = (TP+TN)/(TP+FP+TN+FN)
Sensitivity = TP/(TP+FN)
Specificity = TN/(TN+FP)
Precision = TP/(TP+FP)

print ("Sensitivity = %0.2f" %Sensitivity )
print ("Specificity = %0.2f" %Specificity )
print ("Precision = %0.2f" %Precision )

# COMMAND ----------

# DBTITLE 1,DT Hyperparams - Best Parameters / Feature Importance
# Get the best model from CrossValidator
best_dt_model = dt_model_hyper.bestModel  # Ensure rf_result_hyper is CrossValidatorModel

# Print Best Model Parameters
print("\nBest Decision Tree Parameters:")
print(f"Max Depth: {best_dt_model.getMaxDepth()}")  # Corrected
print(f"Max Bins: {best_dt_model.getMaxBins()}")  # Corrected

# COMMAND ----------

# MAGIC %md
# MAGIC ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ####**Random Forest**

# COMMAND ----------

# DBTITLE 1,Random Forest - Training
#Create decision tree model to data train
rf = RandomForestClassifier(featuresCol='features', labelCol="label",numTrees=100, maxDepth = 20, maxBins=128, seed=42)
rf_model = rf.fit(train)

#Transform model to data test
rf_result = rf_model.transform(test)

#View Result
rf_result.show(5)

# COMMAND ----------

# DBTITLE 1,Random Forest - Evaluation
#Random Forest Evaluation
#Evaluate model by calculatin accuracy and area under curve (AUC)
rf_eval = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="label")
rf_eval2= MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")
rf_AUC  = rf_eval.evaluate(rf_result)
rf_ACC  = rf_eval2.evaluate(rf_result, {rf_eval2.metricName:"accuracy"})

#Create Dataframe to Calculate Log Loss
y_test= test.select('label')
rf_proba=rf_result.select('probability')
#Convert pyspark dataframe to numpy array
rf_proba= np.array(rf_proba.select('probability').collect())
rf_proba=rf_proba.reshape(-1, rf_proba.shape[-1])                #Convert numpy array 3 dimentional to 2 dimentional
y_test=y_test.toPandas()                                            #Convert y_test dataframe to pandas dataframe
y_test=pd.Series(y_test['label'].values)                            #Convert y_test pandas dataframe to pandas series
#Calculate log loss from Gradient Boosting
LogLoss = log_loss(y_test, rf_proba) 

#ROC Grafik
PredAndLabels           = rf_result.select("probability", "label")
PredAndLabels_collect   = PredAndLabels.collect()
PredAndLabels_list      = [(float(i[0][0]), 1.0-float(i[1])) for i in PredAndLabels_collect]
PredAndLabels           = sc.parallelize(PredAndLabels_list)

metrics = BinaryClassificationMetrics(PredAndLabels)

# Visualization
FPR = dict()                                                        # FPR: False Positive Rate
tpr = dict()                                                        # TPR: True Positive Rate
roc_auc = dict()
 
y_test = [i[1] for i in PredAndLabels_list]
y_score = [i[0] for i in PredAndLabels_list]
 
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# Plot ROC Graphic
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc="lower right")
plt.show()

# Area under ROC
print("Random Forest Area Under ROC")
print("Area under ROC = %.2f" % metrics.areaUnderROC)
print("Random Forest Performance Measure")
print("Accuracy = %0.2f" % rf_ACC)
print("AUC = %.2f" % rf_AUC)
print("Log Loss Random Forest:%.4f" % LogLoss)


# COMMAND ----------

# DBTITLE 1,Random Forest - Confusion Matrix
#CONFUSION MATRIX RF
cm_rf_result = rf_result.crosstab("prediction", "label")
cm_rf_result = cm_rf_result.toPandas()
print(cm_rf_result)

# COMMAND ----------

# Calculatation of Accuracy, Sensitivity, Specificity and Precision
TP = cm_rf_result["1.0"][0]
FP = cm_rf_result["0.0"][0]
TN = cm_rf_result["0.0"][1]
FN = cm_rf_result["1.0"][1]
Accuracy = (TP+TN)/(TP+FP+TN+FN)
Sensitivity = TP/(TP+FN)
Specificity = TN/(TN+FP)
Precision = TP/(TP+FP)

print ("Sensitivity = %0.2f" %Sensitivity )
print ("Specificity = %0.2f" %Specificity )
print ("Precision = %0.2f" %Precision )

# COMMAND ----------

# DBTITLE 1,RF- Feature Importance overview
# Extract feature importances
feature_importance = rf_model.featureImportances.toArray()

# Get feature names
feature_metadata = train.schema["features"].metadata["ml_attr"]["attrs"]
feature_names = []

for key in feature_metadata:
    feature_names += [f["name"] for f in feature_metadata[key]]

# Create DataFrame for Feature Importances
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})

# Identify and Sum Categorical Features
category_prefixes = ("cat", "cate", "categ", "category")  # Define prefixes
category_mask = importance_df["Feature"].str.startswith(category_prefixes)  # Find matches

# Sum importance of all categorical features
category_importance_sum = importance_df.loc[category_mask, "Importance"].sum()

# Remove individual categorical features
importance_df = importance_df[~category_mask]

# Append the new aggregated feature
importance_df = importance_df.append({"Feature": "category_product", "Importance": category_importance_sum}, ignore_index=True)

# Sort and Display
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Plot the Top 10 Features
plt.figure(figsize=(10, 5))
plt.barh(importance_df["Feature"][:10], importance_df["Importance"][:10])
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("Top 10 Features Importance (Random Forest) - Grouped")
plt.gca().invert_yaxis()
plt.show()

# Display DataFrame
display(importance_df)

# COMMAND ----------

# DBTITLE 1,RF- Feature importance Product Category
# Extract Feature Importances
feature_importance = rf_model.featureImportances.toArray()

# Get feature names
feature_metadata = train.schema["features"].metadata["ml_attr"]["attrs"]
feature_names = []

for key in feature_metadata:
    feature_names += [f["name"] for f in feature_metadata[key]]

# Create DataFrame for Feature Importances
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})

# Filter Only Product Category Features
category_prefixes = ("cat", "cate", "categ", "category", "product_category")  # Adjust prefixes if needed
product_category_df = importance_df[importance_df["Feature"].str.startswith(category_prefixes)]

# Sort by Importance
product_category_df = product_category_df.sort_values(by="Importance", ascending=False)

# Plot the Top Product Categories
plt.figure(figsize=(10, 5))
plt.barh(product_category_df["Feature"][:10], product_category_df["Importance"][:10])
plt.xlabel("Feature Importance")
plt.ylabel("Product Category")
plt.title("Top 10 Most Impactful Product Categories on Review Answer")
plt.gca().invert_yaxis()
plt.show()

# Display the filtered dataframe
display(product_category_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ####**Random Forest Hyper-Parameters Tuning**

# COMMAND ----------

# DBTITLE 1,Random Forest Hyperparams - Training
#Random Forest With Hyper-Parameter Tuning

rf_hyper= RandomForestClassifier(featuresCol='features', labelCol="label")

# Hyper-Parameter Tuning
paramGrid_rf = ParamGridBuilder() \
    .addGrid(rf.numTrees, [40, 60, 80, 100])\
    .addGrid(rf.maxDepth, [10, 15, 20])\
    .build()
crossval_rf = CrossValidator(estimator=rf_hyper,
                             estimatorParamMaps=paramGrid_rf,
                             evaluator=BinaryClassificationEvaluator(),
                             numFolds=3) 
#fit model to data train
rf_model_hyper=crossval_rf.fit(train)

#transfrom model to data test
rf_result_hyper = rf_model_hyper.transform(test)

#View Result
rf_result_hyper.show(5)

# COMMAND ----------

# DBTITLE 1,Random Forest Hyperparams - Evaluation
#Random Forest With Hyper-Parameter Evaluation
#Evaluate model by calculating accuracy and area under curve (AUC)
rf_hyper_eval = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="label")
rf_hyper_eval2= MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")
rf_hyper_AUC  = rf_hyper_eval.evaluate(rf_result_hyper)
rf_hyper_ACC  = rf_hyper_eval2.evaluate(rf_result_hyper, {rf_hyper_eval2.metricName:"accuracy"})


print("Random Forest Performance Measure")
print("Accuracy = %0.2f" % rf_hyper_ACC)
print("AUC = %.2f" % rf_hyper_AUC)

# COMMAND ----------

# DBTITLE 1,Random Forest Hyperparams - Confusion Matrix
#CONFUSION MATRIX RF HYPERPARAMETER
cm_rf_result_hyper = rf_result_hyper.crosstab("prediction", "label")
cm_rf_result_hyper = cm_rf_result_hyper.toPandas()
print(cm_rf_result_hyper)


# Calculatation of Accuracy, Sensitivity, Specificity and Precision
TP = cm_rf_result_hyper["1.0"][0]
FP = cm_rf_result_hyper["0.0"][0]
TN = cm_rf_result_hyper["0.0"][1]
FN = cm_rf_result_hyper["1.0"][1]
Accuracy = (TP+TN)/(TP+FP+TN+FN)
Sensitivity = TP/(TP+FN)
Specificity = TN/(TN+FP)
Precision = TP/(TP+FP)

print ("Sensitivity = %0.2f" %Sensitivity )
print ("Specificity = %0.2f" %Specificity )
print ("Precision = %0.2f" %Precision )

# COMMAND ----------

# DBTITLE 1,RF Hyperparams - Best Parameters / Feature Importance
# Get the best model from CrossValidator
best_rf_model = rf_model_hyper.bestModel  # Ensure rf_result_hyper is CrossValidatorModel

# Print Best Model Parameters
print("\nBest Random Forest Parameters:")
print(f"Number of Trees: {best_rf_model.getNumTrees}")  # Corrected
print(f"Max Depth: {best_rf_model.getMaxDepth()}")  # Corrected
print(f"Impurity: {best_rf_model.getImpurity()}")  # Corrected

# Print Feature Importances
print("\nFeature Importances:")
print(best_rf_model.featureImportances)

# COMMAND ----------

# MAGIC %md
# MAGIC ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Super Vector Machine (SVM)**

# COMMAND ----------

# DBTITLE 1,SVM - Training
# Define your classifier
lsvc = LinearSVC(maxIter=10, regParam=0.1)

# Fit the model
lsvcModel = lsvc.fit(train)

# Print the coefficients and intercept for linearsSVC
coefficients = lsvcModel.coefficients
intercept = lsvcModel.intercept
print("Coefficients: " + str(coefficients))
print("Intercept: " + str(intercept))

# Make predictions on test data
predictions = lsvcModel.transform(test)

# Show some example predictions
predictions.show(10)

# COMMAND ----------

# DBTITLE 1,SVM - Evaluation
# Evaluate the model using BinaryClassificationEvaluator (AUC)
binary_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = binary_evaluator.evaluate(predictions)


# Evaluate using MulticlassClassificationEvaluator (Accuracy, Precision, Recall, F1)
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = accuracy_evaluator.evaluate(predictions)

precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
precision = precision_evaluator.evaluate(predictions)

recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
recall = recall_evaluator.evaluate(predictions)

f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = f1_evaluator.evaluate(predictions)

print(f"Area Under ROC Curve (AUC): {auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ####**SVM HYPER-PARAMETERS TUNING**

# COMMAND ----------

# DBTITLE 1,SVM Hyperparams - Training
# Define the SVM model
lsvc = LinearSVC(labelCol="label", featuresCol="features")

# Create a parameter grid for hyperparameter tuning
paramGrid = (ParamGridBuilder()
             .addGrid(lsvc.regParam, [0.01, 0.1, 1.0])  # Regularization parameter
             .addGrid(lsvc.maxIter, [10, 50, 100])  # Number of iterations
             .build())

# Define evaluator (use AUC to compare models)
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

# Define cross-validation with 3 folds
crossval = CrossValidator(estimator=lsvc,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)  # 3-fold cross-validation

# Train the model using cross-validation
cvModel = crossval.fit(train)
cvModel_hyper = cvModel.transform(test)

cvModel_hyper.show()

# COMMAND ----------

# DBTITLE 1,SVM Hyperparams - Evaluation
#SVM With Hyper-Parameter Evaluation

#Evaluate model by calculating accuracy and area under curve (AUC)
cvModel_eval = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="label")
cvModel_hyper2= MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")
#cv_hyper_AUC  = cvModel_hyper.evaluate(cvModel_hyper)
cv_hyper_ACC  = cvModel_hyper2.evaluate(cvModel_hyper, {cvModel_hyper2.metricName:"accuracy"})

print("Decision Tree Performance Measure")
print("Accuracy = %0.2f" % rf_hyper_ACC)
print("AUC = %.2f" % rf_hyper_AUC)

# COMMAND ----------

# DBTITLE 1,SVM Hyperparams - Best Parameters
# Get the best model from cross-validation
bestModel = cvModel.bestModel

# Print best parameters
print(f"Best Regularization Parameter (regParam): {bestModel._java_obj.getRegParam()}")
print(f"Best Max Iterations (maxIter): {bestModel._java_obj.getMaxIter()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ####**Gradient Boosting**

# COMMAND ----------

# DBTITLE 1,Gradient Boosting - Train
# Intialization
gbt = GBTClassifier(featuresCol="features", labelCol="label", 
                    maxDepth=15,
                    maxIter=15,
                    maxBins=80,
                    stepSize=0.1)
#Fit to the train
gbt_model = gbt.fit(train)

#Transfrom model to data test
gbt_result = gbt_model.transform(test)

#View Result
gbt_result.show(5)

# COMMAND ----------

# DBTITLE 1,Gradient Boosting - Evaluation
#Gradient Boosting Evaluation
#Evaluate model by calculating accuracy and area under curve (AUC)
gbt_eval = BinaryClassificationEvaluator(rawPredictionCol="probability",labelCol="label")
gbt_eval2= MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")
gbt_AUC  = gbt_eval.evaluate(gbt_result)
gbt_ACC  = gbt_eval2.evaluate(gbt_result, {gbt_eval2.metricName:"accuracy"})

#ROC Grafic
PredAndLabels           = gbt_result.select("probability", "label")
PredAndLabels_collect   = PredAndLabels.collect()
PredAndLabels_list      = [(float(i[0][0]), 1.0-float(i[1])) for i in PredAndLabels_collect]
PredAndLabels           = sc.parallelize(PredAndLabels_list)

metrics = BinaryClassificationMetrics(PredAndLabels)


#Create Dataframe to Calculate Log Loss
y_test= test.select('label')
gbt_proba=gbt_result.select('probability')
#Convert pyspark dataframe to numpy array
gbt_proba= np.array(gbt_proba.select('probability').collect())
gbt_proba=gbt_proba.reshape(-1, gbt_proba.shape[-1])                #Convert numpy array 3 dimentional to 2 dimentional
y_test=y_test.toPandas()                                            #Convert y_test dataframe to pandas dataframe
y_test=pd.Series(y_test['label'].values)                            #Convert y_test pandas dataframe to pandas series
#Calculate log loss from Gradient Boosting
LogLoss = log_loss(y_test, gbt_proba) 


# Visualization
FPR = dict()                                                        # FPR: False Positive Rate
tpr = dict()                                                        # TPR: True Positive Rate
roc_auc = dict()
 
y_test = [i[1] for i in PredAndLabels_list]
y_score = [i[0] for i in PredAndLabels_list]
 
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

#Plot ROC Graphic
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Gradient Boosting')
plt.legend(loc="lower right")
plt.show()



# Area under ROC
print("Gradient Boosting Area Under ROC")
print("Area under ROC = %.2f" % metrics.areaUnderROC)
print("Gradient Boosted Tree Performance Measure")
print("Accuracy = %0.2f" % gbt_ACC)
print("AUC = %.2f" % gbt_AUC)

# COMMAND ----------

# DBTITLE 1,Gradient Boosting - Confusion Matrix
#Confusion Matrix
cm_gbt_result = gbt_result.crosstab("prediction", "label")
cm_gbt_result = cm_gbt_result.toPandas()
print(cm_gbt_result)

# COMMAND ----------

# DBTITLE 1,Additional Evaluation Metrics
# Calculate Accuracy, Sensitivity, Specificity and Precision
TP = cm_gbt_result["1.0"][0]
FP = cm_gbt_result["0.0"][0]
TN = cm_gbt_result["0.0"][1]
FN = cm_gbt_result["1.0"][1]
Accuracy = (TP+TN)/(TP+FP+TN+FN)
Sensitivity = TP/(TP+FN)
Specificity = TN/(TN+FP)
Precision = TP/(TP+FP)

print ("Sensitivity = %0.2f" %Sensitivity )
print ("Specificity = %0.2f" %Specificity )
print ("Precision = %0.2f" %Precision )
print("Log Loss Gradient Boosting:%.4f" % LogLoss)

# COMMAND ----------

# MAGIC %md
# MAGIC ####**GRADIENT BOOSTING HYPER-PARAMETER TUNING**

# COMMAND ----------

# DBTITLE 1,GB Hyperparams - Training
#Gradient Boosting With Hyper-Parameter
gbt_hyper= GBTClassifier(featuresCol="features", labelCol="label")


# Hyper-Parameter Tuning
paramGrid_gbt = (ParamGridBuilder()
                 .addGrid(gbt.maxDepth, [10, 15, 25])  # Tree depth (controls complexity)
                 .addGrid(gbt.maxIter, [15, 30])  # Number of boosting iterations
                 .addGrid(gbt.maxBins, [75]) # Max number of bins for feature discretization
                 .addGrid(gbt.stepSize, [0.01, 0.1])  # Learning rate
                 .build())

crossval_gbt = CrossValidator(estimator=gbt_hyper,
                             estimatorParamMaps=paramGrid_gbt,
                             evaluator=BinaryClassificationEvaluator(),
                             numFolds=3)


#fit model to data train
gbt_model_hyper = crossval_gbt.fit(train)

#transfrom model to data test
gbt_result_hyper = gbt_model_hyper.transform(test)

#View Result
gbt_result_hyper.show(5)

# COMMAND ----------

# DBTITLE 1,GB Hyperparams - Evaluation
#Gradient Boosting With Hyper-Parameter Evaluation
#Evaluate model by calculating accuracy and area under curve (AUC)
gbt_eval_hyper = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="label")
gbt_eval_hyper2= MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")
gbt_hyper_AUC  = gbt_eval_hyper.evaluate(gbt_result_hyper)
gbt_hyper_ACC  = gbt_eval_hyper2.evaluate(gbt_result_hyper, {gbt_eval_hyper2.metricName:"accuracy"})

#ROC Grafic
PredAndLabels           = gbt_result_hyper.select("probability", "label")
PredAndLabels_collect   = PredAndLabels.collect()
PredAndLabels_list      = [(float(i[0][0]), 1.0-float(i[1])) for i in PredAndLabels_collect]
PredAndLabels           = sc.parallelize(PredAndLabels_list)

metrics = BinaryClassificationMetrics(PredAndLabels)

#Create Dataframe to Calculate Log Loss
y_test= test.select('label')
gbt_hyper_proba=gbt_result_hyper.select('probability')
#Convert pyspark dataframe to numpy array
gbt_hyper_proba= np.array(gbt_hyper_proba.select('probability').collect())
gbt_hyper_proba=gbt_hyper_proba.reshape(-1, gbt_hyper_proba.shape[-1])          #Convert numpy array 3 dimentional to 2 dimentional
y_test=y_test.toPandas()                                                        #Convert y_test dataframe to pandas dataframe
y_test=pd.Series(y_test['label'].values)                                        #Convert y_test pandas dataframe to pandas series
#Calculate log loss from Gradient Boosting
LogLoss = log_loss(y_test, gbt_hyper_proba) 

# Visualization
FPR = dict()                                                        # FPR: False Positive Rate
tpr = dict()                                                        # TPR: True Positive Rate
roc_auc = dict()
 
y_test = [i[1] for i in PredAndLabels_list]
y_score = [i[0] for i in PredAndLabels_list]
 
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

#Plot ROC Graphic
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Gradient Boosting')
plt.legend(loc="lower right")
plt.show()

# Area under ROC
print("Gradient Boosting Area Under ROC")
print("Area under ROC = %.2f" % metrics.areaUnderROC)
print("Gradient Boosted Tree Performance Measure")
print("Accuracy = %0.2f" % gbt_hyper_ACC)
print("AUC = %.2f" % gbt_hyper_AUC)

# COMMAND ----------

# DBTITLE 1,SVM Hyperparams - Best Parameters
bestModel_gbt = gbt_model_hyper.bestModel

# Print best hyperparameters for GBTClassifier
print("\n Best Hyperparameters for GBTClassifier:")
print(f"Best Max Depth (maxDepth): {bestModel_gbt.getMaxDepth()}")
print(f"Best Max Iterations (maxIter): {bestModel_gbt.getMaxIter()}")
print(f"Best Max Bins (maxBins): {bestModel_gbt.getMaxBins()}")
print(f"Best Step Size (stepSize): {bestModel_gbt.getStepSize()}")

# COMMAND ----------

# DBTITLE 1,Confusion Metrics
#Confusion Matrix
cm_gbt_result_hyper = gbt_result_hyper.crosstab("prediction", "label")
cm_gbt_result_hyper = cm_gbt_result_hyper.toPandas()
print(cm_gbt_result_hyper)


# COMMAND ----------

# DBTITLE 1,Additional Metrics
# Calculate Accuracy, Sensitivity, Specificity and Precision
TP = cm_gbt_result_hyper["1.0"][0]
FP = cm_gbt_result_hyper["0.0"][0]
TN = cm_gbt_result_hyper["0.0"][1]
FN = cm_gbt_result_hyper["1.0"][1]
Accuracy = (TP+TN)/(TP+FP+TN+FN)
Sensitivity = TP/(TP+FN)
Specificity = TN/(TN+FP)
Precision = TP/(TP+FP)

print ("Sensitivity = %0.2f" %Sensitivity )
print ("Specificity = %0.2f" %Specificity )
print ("Precision = %0.2f" %Precision )
print("Log Loss Gradient Boosting:%.4f" % LogLoss)

# COMMAND ----------

# MAGIC %md
# MAGIC ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC #**Test Part**

# COMMAND ----------

# DBTITLE 1,Test Tables Importation
# FILE IMPORTATION
# Product Table
file_path = "/FileStore/test_products.csv"
test_products = spark.read.csv(file_path, header=True, inferSchema=True)
test_products.write.format("csv")\
  .mode("overwrite")\
  .save("/data_project/test_products.csv")

# Order Table
file_path = "/FileStore/test_orders.csv"
test_orders = spark.read.csv(file_path, header=True, inferSchema=True)
test_orders.write.format("csv")\
  .mode("overwrite")\
  .save("/data_project/test_orders.csv")

# Item Table 
file_path = "/FileStore/test_order_items.csv"
test_items = spark.read.csv(file_path, header=True, inferSchema=True)
test_items.write.format("csv")\
  .mode("overwrite")\
  .save("/data_project/test_items.csv")

# Payment Table
file_path = "/FileStore/test_order_payments.csv"
test_payments = spark.read.csv(file_path, header=True, inferSchema=True)
test_payments.write.format("csv")\
  .mode("overwrite")\
  .save("/data_project/test_payments.csv")

# COMMAND ----------

# DBTITLE 1,Product Test Table Preprocess
#Count the number of distinct 'product_category_name' that we have
print('Nbr of distinct product category name:')
count_prod_category_name=test_products.select(col("product_category_name")).distinct().count()
print(count_prod_category_name)

#Count the number of distinct "product_id" that we have
print('Nbr of distinct product id:')
count_productID=test_products.select(col("product_id")).distinct().count()
print(count_productID)

#Add some features 
test_products=test_products\
    .withColumn("product_volume",col("product_length_cm") * col("product_width_cm") * col("product_height_cm"))\
    .withColumn("size_to_weight_ratio",col("product_volume") / col("product_weight_g"))\
    .withColumn("product_density", col("product_weight_g") / col("product_volume"))

# Drop Null Raws with nulls values
test_products = test_products.na.drop("any")
# Drop Duplicates
test_products = test_products.dropDuplicates()
test_products.count()
# Verification of the nb of Nulls per column
test_products.count()
test_products.select([count(when(col(c).isNull(), c)).alias(c) for c in test_products.columns]).show()

# COMMAND ----------

# DBTITLE 1,Order Test Table Preprocess
# Initialize Columns to convert
columns_to_convert = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]
# Convertion
for column in columns_to_convert:
    test_orders = test_orders.withColumn(column, to_timestamp(col(column), "yyyy-MM-dd HH:mm:ss"))
#Verification
test_orders.printSchema()

# Drop Null Raws with nulls values
test_orders = test_orders.na.drop("any")
# Drop Duplicates
test_orders=test_orders.dropDuplicates()
# Verification of the nb of Nulls per column
test_orders.select([count(when(col(c).isNull(), c)).alias(c) for c in test_orders.columns]).show()

# Count orders per customer
customer_order_counts = test_orders.groupBy("customer_id").agg(
    count("order_id").alias("order_count"),
    min("order_purchase_timestamp").alias("first_order_date")
)

# Get the most recent purchase date for each customer
last_order_date_per_customer = test_orders.groupBy("customer_id") \
                                        .agg(max(col("order_purchase_timestamp")).alias("last_order_date"))

# Calculate the time since the last purchase
last_order_date_per_customer = last_order_date_per_customer.withColumn(
    "recency",
    datediff(current_date(), col("last_order_date"))
)

last_order_date_per_customer.select("customer_id", "last_order_date", "recency").show()
# Join the created data frame to orders
test_orders=test_orders.join(last_order_date_per_customer, on="customer_id", how="left")
# Join back to orders and add repeat customer features
# When we use max("order_purchase_timestamp") over the window, it will include the current order
test_orders = test_orders.join(customer_order_counts, "customer_id","left")
test_orders = test_orders.withColumn(
    "is_repeated_customer", 
    when(col("order_count") > 1, 1).otherwise(0)
).withColumn("delivery_duration", datediff(col("order_delivered_customer_date"), col("order_purchase_timestamp"))) \
                     .withColumn("approval_duration", datediff(col("order_approved_at"), col("order_purchase_timestamp"))) \
                     .withColumn("delivery_delay", datediff(col("order_estimated_delivery_date"), col("order_delivered_customer_date")))
test_orders.printSchema()

# Last verifications 
test_orders.show(3)
print('Distinct order id :')
test_orders.select(col("order_id")).distinct().count()
print('Number of rows:')
test_orders.count()

#Count the number of nulls per column
test_orders.select([count(when(col(c).isNull(), c)).alias(c) for c in orders.columns]).show()

# COMMAND ----------

# DBTITLE 1,Item Test Table Preprocess
# Item_total_cost creation & # shipping_to_price_ratio creation
test_items=test_items\
    .withColumn("item_total_cost",col("price")+col("shipping_cost"))\
        .withColumn("shipping_to_price_ratio",col("shipping_cost")/col("price"))

#Since order_items and order_payments have multiple rows per order_id, we aggregate them before joining.
#Aggregate items at the order_id level

items_agg = test_items.groupBy("order_id").agg(
    count("order_item_id").alias("nbr_items"),
    first("product_id").alias("product_id"),  # Keeps only one product ID per order
    min("price").alias("min_price"),
    max("price").alias("max_price"),
    avg("price").alias("avg_price"),
    sum("price").alias("total_price"),
    min("shipping_cost").alias("min_shipping_cost"),
    max("shipping_cost").alias("max_shipping_cost"),
    avg("shipping_cost").alias("avg_shipping_cost"),
    sum("shipping_cost").alias("total_shipping_cost"),
    min("item_total_cost").alias("min_item_total_cost"),
    max("item_total_cost").alias("max_item_total_cost"),
    avg("item_total_cost").alias("avg_item_total_cost"),
    sum("item_total_cost").alias("total_item_total_cost"),
    min("shipping_to_price_ratio").alias("min_shipping_to_price_ratio"),
    max("shipping_to_price_ratio").alias("max_shipping_to_price_ratio"),
    avg("shipping_to_price_ratio").alias("avg_shipping_to_price_ratio"),
    sum("shipping_to_price_ratio").alias("total_shipping_to_price_ratio")
)

items_agg.show(3)

# Drop Null Raws with nulls values
items_agg=items_agg.na.drop("any")
# Drop Duplicates
items_agg = items_agg.dropDuplicates()
# Verification of the nb of Nulls per column
items_agg.select([count(when(col(c).isNull(), c)).alias(c) for c in items_agg.columns]).show()

# COMMAND ----------

# DBTITLE 1,Items_Product Test Table Preprocess
test_items_product = items_agg.join(test_products,"product_id","left")
#Verification
test_items_product.show(5)
test_items_product.count()
test_items_product.printSchema()

test_items_product=test_items_product.na.drop("any")
test_items_product = test_items_product.dropDuplicates()
test_items_product.select([count(when(isnull(c) | isnan(c), c)).alias(c) for c in test_items_product.columns]).show()

# Group by order_id and perform multiple aggregations
test_items_product = test_items_product.groupBy("order_id").agg(
    avg("product_name_lenght").alias("avg_name_length"),
    avg("product_description_lenght").alias("avg_description_length"),
    sum("product_photos_qty").alias("total_photos"),
    sum("product_weight_g").alias("total_weight_g"),
    sum("product_volume").alias("total_volume"),
    avg("product_volume").alias("avg_product_volume_per_order"),
    sum("product_density").alias("total_product_density"),
    avg("product_density").alias("avg_product_density"),
    concat_ws(", ", collect_set("product_category_name")).alias("category"),  # Concatenated categories
    countDistinct("product_category_name").alias("distincy_category_count")  # Count of unique categories
)

test_items_product.show(truncate=False)

# COMMAND ----------

# DBTITLE 1,Payment Test Table Preprocess
# Categorical :
payments_sorted = test_payments.orderBy("order_id")
test_payments = test_payments.na.drop("any")
print('********* installment_value computing **********') # = payment_value / payment_installments
test_payments = test_payments.withColumn(
    "installment_value",
    when(col("payment_installments") > 0, col("payment_value") / col("payment_installments"))
    .otherwise(None)  # Avoid division by zero
    )
print('********* monetary_value computing **********')
order_total_value = test_payments.groupBy("order_id") \
                             .agg(sum(col("payment_value")).alias("monetary_value"))
print('********* joining monetary_value to payments **********')
test_payments = test_payments.join(order_total_value, on="order_id", how="left")
print('********* Identifying all the possible values on payment_type **********')
distinct_values = test_payments.select("payment_type").distinct().rdd.flatMap(lambda x: x).collect()
print(distinct_values)

print('********* creation of the dummy **********')
for payment in distinct_values:

    test_payments = test_payments.withColumn(f"dummy_{payment}", when(col("payment_type") == payment, 1).otherwise(0))

print('********* Aggregation on order_id by keeping the max of each dummy **********')
dummy_cols = ["dummy_mobile", "dummy_credit_card", "dummy_voucher", "dummy_debit_card"]
payments_agg = test_payments.groupBy("order_id").agg(
    *[max(col).alias(col) for col in dummy_cols],
    max("payment_sequential").alias('total_payment_sequential'),
    sum("payment_value").alias("total_payment_value"),
    avg("payment_installments").alias("avg_payment_installments"),
    max("payment_installments").alias("max_payment_installments")
)
print('********* show the final table :  **********')
payments_agg.show(20,truncate=False)
print('********* Final Table analysis (Describe):  **********')
test_payments.printSchema()

# Drop Null Raws with nulls values
test_payments = test_payments.na.drop("any")
# Drop Duplicates
test_payments  = test_payments.dropDuplicates()
# Verification of the nb of Nulls per column
test_payments.select([count(when(col(c).isNull(), c)).alias(c) for c in test_payments.columns]).show()

# COMMAND ----------

# DBTITLE 1,Basetable Test Part
# Basetable cration by joining the differents tables.
test_basetable = test_orders.join(test_items_product,"order_id","left")\
    .join(test_payments,"order_id","left")

test_basetable.printSchema()

# Drop Null Raws with nulls values
test_basetable=test_basetable.na.drop("any")
# Drop Duplicates
test_basetable=test_basetable.dropDuplicates()


# Droping unwhished features
test_basetable = test_basetable.drop("order_purchase_timestamp","order_approved_at","order_delivered_carrier_date","order_delivered_customer_date","order_estimated_delivery_date","payment_installments", "review_creation_date", "review_answer_timestamp","product_id","customer_id","order_status","review_id",'dummy_not_defined','delivery_delay')

# Verification of the nb of Nulls per column
test_basetable.select([count(when(col(c).isNull(), c)).alias(c) for c in test_basetable.columns]).show()

print('Distinct order id :')
print(test_basetable.select(col("order_id")).distinct().count())
print('Number of rows:')
test_basetable.count()

display(test_basetable)

# COMMAND ----------

# MAGIC %md
# MAGIC ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Pipeline train & test**

# COMMAND ----------

# Pipeline for TRAIN SET (have "review")
formula_train = RFormula(formula="review ~ . - order_id", 
                         featuresCol="features", 
                         labelCol="label",
                         handleInvalid="skip")

scaler = StandardScaler(inputCol="features", 
                        outputCol="scaledFeatures")

selector = ChiSqSelector(numTopFeatures=15, 
                         featuresCol="scaledFeatures",
                         outputCol="features")

pipeline_train = Pipeline(stages=[formula_train, scaler]) 

# Fit and Transform on TRAIN
pipeline_model = pipeline_train.fit(basetable)
basetable_transformed = pipeline_model.transform(basetable).select("features", "label")
basetable_transformed.show()

# Pipeline for the TEST SET (DO NOT contain "review")
formula_test = RFormula(formula="~ . - order_id",  
                        featuresCol="features",
                        handleInvalid="skip")

pipeline_test = Pipeline(stages=[formula_test, scaler])  # Same pipeline Without label

# Transform on TEST (without re fit)
test_basetable_transformed = pipeline_model.transform(test_basetable).select("order_id","features")  # No label here
test_basetable_transformed.show()

# COMMAND ----------

print(f'Train Set :{basetable.count()}')
print(f'Test Set :{test_basetable.count()}')

# COMMAND ----------

# MAGIC %md
# MAGIC ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Gradient Boosting on Test Data**

# COMMAND ----------

# DBTITLE 1,GBT - Init / Fit / Transform test
from pyspark.ml.classification import GBTClassifier

# Define the GBT model with a custom threshold
gbt = GBTClassifier(
    featuresCol="features",
    labelCol="label",
    maxDepth=15,
    maxBins=80,
    maxIter=15,
    stepSize=0.1,
    seed=42  
)

# Train the model on basetable (trainset)
gbt_model = gbt.fit(basetable_transformed)

# Transform the test_basetable (testset) 
gbt_result = gbt_model.transform(test_basetable_transformed)


# COMMAND ----------

# DBTITLE 1,Show Test Predictions
# Display Predictions
gbt_result.select("features","order_id", "custom_prediction", "probability").show(10)

# COMMAND ----------

# DBTITLE 1,Prediction File Creation
# Select and collect the data
gbt_predictions = gbt_result.select("order_id", "prediction")

# Convert to Pandas and create the csv
gbt_predictions_pd = gbt_predictions.toPandas()
gbt_predictions_pd.to_csv("BDT_2025_Aboudi_Gaillard.csv", index=False)

print("✅ File 'BDT_2025_Aboudi_Gaillard.csv' suffesfully generated")


# COMMAND ----------

# DBTITLE 1,Move the csv File to DBFS & DownLoad Button
#Move the csv predictions file to FileStore DBFS
dbutils.fs.cp("file:/databricks/driver/BDT_2025_Aboudi_Gaillard.csv", "dbfs:/FileStore/BDT_2025_Aboudi_Gaillard.csv")
#Download button creation
displayHTML(f'<a href="/files/BDT_2025_Aboudi_Gaillard.csv" target="_blank">Download BDT_2025_Aboudi_Gaillard.csv</a>')

# COMMAND ----------

# DBTITLE 1,GBT - Feature Importance Overview
# Extract feature importances
feature_importance = gbt_model.featureImportances.toArray()

# Get feature names
feature_metadata = basetable_transformed.schema["features"].metadata["ml_attr"]["attrs"]
feature_names = []

for key in feature_metadata:
    feature_names += [f["name"] for f in feature_metadata[key]]

# Create DataFrame for Feature Importances
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})

# Identify and Sum Categorical Features
category_prefixes = ("cat", "cate", "categ", "category")  # Define prefixes
category_mask = importance_df["Feature"].str.startswith(category_prefixes)  # Find matches

# Sum importance of all categorical features
category_importance_sum = importance_df.loc[category_mask, "Importance"].sum()

# Remove individual categorical features
importance_df = importance_df[~category_mask]

# Append the new aggregated feature
importance_df = importance_df.append({"Feature": "category_product", "Importance": category_importance_sum}, ignore_index=True)

# Sort and Display
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Plot the Top 10 Features
plt.figure(figsize=(10, 5))
plt.barh(importance_df["Feature"][:10], importance_df["Importance"][:10])
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("Top 10 Features Importance (GBTClassifier) - Grouped")
plt.gca().invert_yaxis()
plt.show()

# Display DataFrame
display(importance_df)



# COMMAND ----------

# DBTITLE 1,GBT - Feature Importance Plot with all values
# Feature Importance Extraction
feature_importance = gbt_model.featureImportances.toArray()

# Get the features name
feature_metadata = basetable_transformed.schema["features"].metadata["ml_attr"]["attrs"]
feature_names = []

for key in feature_metadata:
    feature_names += [f["name"] for f in feature_metadata[key]]

# DataFrame Creation to Display the most importants features
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Plot
plt.figure(figsize=(10, 5))
plt.barh(importance_df["Feature"][:10], importance_df["Importance"][:10])
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("Top 10 Features Importance (GBTClassifier)")
plt.gca().invert_yaxis()
plt.show()

# Display Importance Feature DataFrame
display(importance_df)


# COMMAND ----------

# DBTITLE 1,GBT - Product Category Importance

# Extract Feature Importances
feature_importance = gbt_model.featureImportances.toArray()

# Get feature names
feature_metadata = basetable_transformed.schema["features"].metadata["ml_attr"]["attrs"]
feature_names = []

for key in feature_metadata:
    feature_names += [f["name"] for f in feature_metadata[key]]

# Create DataFrame for Feature Importances
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})

# Filter Only Product Category Features
category_prefixes = ("cat", "cate", "categ", "category", "product_category")  # Adjust prefixes if needed
product_category_df = importance_df[importance_df["Feature"].str.startswith(category_prefixes)]

# Sort by Importance
product_category_df = product_category_df.sort_values(by="Importance", ascending=False)

# Plot the Top Product Categories
plt.figure(figsize=(10, 5))
plt.barh(product_category_df["Feature"][:10], product_category_df["Importance"][:10])
plt.xlabel("Feature Importance")
plt.ylabel("Product Category")
plt.title("Top 10 Most Impactful Product Categories on Review Answer")
plt.gca().invert_yaxis()
plt.show()

# Display the filtered dataframe
display(product_category_df)


# COMMAND ----------

# MAGIC %md
# MAGIC ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Random Forest on Test Data**

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

# Define the Random Forest model
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100, maxDepth=20, maxBins=128, seed=42)

# Train the model on basetable (trainset)
rf_model = rf.fit(basetable_transformed)

# Transform the test_basetable (testset) 
rf_result = rf_model.transform(test_basetable_transformed)

# COMMAND ----------

# Display Predictions
rf_result.select("features", "order_id", "prediction", "probability").show(10)

# COMMAND ----------

# Select and collect the data
rf_predictions = rf_result.select("order_id", "prediction")

# Convert to Pandas and create the csv
rf_predictions_pd = rf_predictions.toPandas()
rf_predictions_pd.to_csv("predictions_rf.csv", index=False)

print("✅ File 'predictions_rf.csv' successfully generated")

# COMMAND ----------

# Move the CSV predictions file to FileStore DBFS
dbutils.fs.cp("file:/databricks/driver/predictions_rf.csv", "dbfs:/FileStore/predictions_rf.csv")

# Download button creation
displayHTML(f'<a href="/files/predictions_rf.csv" target="_blank">Download predictions_rf.csv</a>')

# COMMAND ----------

# DBTITLE 1,RF - Feature Importance Overview
# Extract feature importances
feature_importance = rf_model.featureImportances.toArray()

# Get feature names
feature_metadata = basetable_transformed.schema["features"].metadata["ml_attr"]["attrs"]
feature_names = []

for key in feature_metadata:
    feature_names += [f["name"] for f in feature_metadata[key]]

# Create DataFrame for Feature Importances
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})

# dentify and Sum Categorical Features
category_prefixes = ("cat", "cate", "categ", "category")  # Define prefixes
category_mask = importance_df["Feature"].str.startswith(category_prefixes)  # Find matches

# Sum importance of all categorical features
category_importance_sum = importance_df.loc[category_mask, "Importance"].sum()

# Remove individual categorical features
importance_df = importance_df[~category_mask]

# Append the new aggregated feature
importance_df = importance_df.append({"Feature": "category_product", "Importance": category_importance_sum}, ignore_index=True)

# Sort and Display
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Plot the Top 10 Features
plt.figure(figsize=(10, 5))
plt.barh(importance_df["Feature"][:10], importance_df["Importance"][:10])
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("Top 10 Features Importance (Random Forest) - Grouped")
plt.gca().invert_yaxis()
plt.show()

# Display DataFrame
display(importance_df)

# COMMAND ----------

# DBTITLE 1,RF - Feature Importance All Values
# Feature Importance Extraction
feature_importance = rf_model.featureImportances.toArray()

# Get the features name
feature_metadata = basetable_transformed.schema["features"].metadata["ml_attr"]["attrs"]
feature_names = []

for key in feature_metadata:
    feature_names += [f["name"] for f in feature_metadata[key]]

# DataFrame Creation to Display the most important features
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

#Plot
plt.figure(figsize=(10, 5))
plt.barh(importance_df["Feature"][:10], importance_df["Importance"][:10])
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("Top 10 Features Importance (RandomForestClassifier)")
plt.gca().invert_yaxis()
plt.show()

# Display Importance Feature DataFrame
display(importance_df)

# COMMAND ----------

# DBTITLE 1,RF - Most important category product on Review Answer

#Extract Feature Importances
feature_importance = rf_model.featureImportances.toArray()

# Get feature names
feature_metadata = basetable_transformed.schema["features"].metadata["ml_attr"]["attrs"]
feature_names = []

for key in feature_metadata:
    feature_names += [f["name"] for f in feature_metadata[key]]

# Create DataFrame for Feature Importances
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})

# Filter Only Product Category Features
category_prefixes = ("cat", "cate", "categ", "category", "product_category")  # Adjust prefixes if needed
product_category_df = importance_df[importance_df["Feature"].str.startswith(category_prefixes)]

# Sort by Importance
product_category_df = product_category_df.sort_values(by="Importance", ascending=False)

# Plot the Top Product Categories
plt.figure(figsize=(10, 5))
plt.barh(product_category_df["Feature"][:10], product_category_df["Importance"][:10])
plt.xlabel("Feature Importance")
plt.ylabel("Product Category")
plt.title("Top 10 Most Impactful Product Categories on Review Answer")
plt.gca().invert_yaxis()
plt.show()

# Display the filtered dataframe
display(product_category_df)

# COMMAND ----------

# MAGIC %md
# MAGIC -----------------------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC #### Additonal Ressources :
# MAGIC https://bbejournal.com/BBE/article/view/908
# MAGIC
# MAGIC
# MAGIC https://www.sciencedirect.com/science/article/abs/pii/S0969698924001619?via%3Dihub
