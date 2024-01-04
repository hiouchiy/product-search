# Databricks notebook source
# MAGIC %md このノートの目的は、Product Search ソリューション・アクセラレータ用のデータを準備することです。 このノートはhttps://github.com/databricks-industry-solutions/product-search

# COMMAND ----------

# MAGIC %md ##はじめに
# MAGIC
# MAGIC このノートブックでは、[Wayfair](https://www.wayfair.com/)がMITライセンスの下で公開している[Wayfair Annotation Dataset (WANDS)](https://www.aboutwayfair.com/careers/tech-blog/wayfair-releases-wands-the-largest-and-richest-publicly-available-dataset-for-e-commerce-product-search-relevance)にアクセスします。
# MAGIC
# MAGIC このデータセットは3種類のファイルから構成されています：
# MAGIC </p>
# MAGIC
# MAGIC * Product - Wayfairのウェブサイトに掲載されている42,000以上の商品
# MAGIC * Query - 商品検索に使用された480の顧客クエリ
# MAGIC * Label - 提供されたクエリに対する233,000以上の商品結果
# MAGIC
# MAGIC データセットに付随する[Annotations Guidelines document](https://github.com/wayfair/WANDS/blob/main/Product%20Search%20Relevance%20Annotation%20Guidelines.pdf)において、Wayfairはクエリのラベル付け方法について言及しています。 クエリ結果に割り当てられたラベルは以下の3つです：
# MAGIC </p>
# MAGIC
# MAGIC * 完全一致(Exact) - このラベルは、検索クエリに完全に一致する商品を表します。
# MAGIC * 部分一致(Partial) - このラベルは、検索クエリに完全に一致しないことを表します。
# MAGIC * 無関係(Irrelevant) - このラベルは、製品がクエリに無関係であることを示します。
# MAGIC
# MAGIC ドキュメントで説明されているように、これらのラベルの割り当てには若干の主観が入りますが、ここでのゴールは、真実を把握することではなく、情報に基づいた人間の判断を把握することです。

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.sql.types import *
import pyspark.sql.functions as fn

import os

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./00_Intro_and_Config"

# COMMAND ----------

# MAGIC %md ##ステップ 1: データセットファイルのダウンロード
# MAGIC
# MAGIC このステップでは、データセットファイルをDatabricksワークスペース内でアクセス可能なディレクトリにダウンロードします：

# COMMAND ----------

# DBTITLE 1,Set Path Variable for Script
os.environ['WANDS_DOWNLOADS_PATH'] = '/dbfs'+ config['dbfs_path'] + '/downloads' 

# COMMAND ----------

# DBTITLE 1,Download Dataset Files
# MAGIC %sh 
# MAGIC
# MAGIC # delete any old copies of temp data
# MAGIC rm -rf $WANDS_DOWNLOADS_PATH
# MAGIC
# MAGIC # make directory for temp tiles
# MAGIC mkdir -p $WANDS_DOWNLOADS_PATH
# MAGIC
# MAGIC # move to temp directory
# MAGIC cd $WANDS_DOWNLOADS_PATH
# MAGIC
# MAGIC # download datasets
# MAGIC wget -q https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/label.csv
# MAGIC wget -q https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/product.csv
# MAGIC wget -q https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/query.csv
# MAGIC
# MAGIC # show folder contents
# MAGIC pwd
# MAGIC ls -l

# COMMAND ----------

# MAGIC %md ##ステップ2：データをテーブルに書き込む
# MAGIC
# MAGIC このステップでは、以前にダウンロードした各ファイルからデータを読み込み、その後のアクセスをより簡単かつ高速にするために、データをテーブルに書き込みます：

# COMMAND ----------

# DBTITLE 1,Process Products
products_schema = StructType([
  StructField('product_id', IntegerType()),
  StructField('product_name', StringType()),
  StructField('product_class', StringType()),
  StructField('category_hierarchy', StringType()),
  StructField('product_description', StringType()),
  StructField('product_features', StringType()),
  StructField('rating_count', FloatType()),
  StructField('average_rating', FloatType()),
  StructField('review_count', FloatType())
  ])

_ = (
  spark
    .read
      .csv(
        path='dbfs:/wands/downloads/product.csv',
        sep='\t',
        header=True,
        schema=products_schema
        )
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('products')
  )

display(
  spark.table('products')
  )

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT product_class, count(*) AS count FROM products GROUP BY product_class ORDER BY count ASC

# COMMAND ----------

# DBTITLE 1,Process Queries
queries_schema = StructType([
  StructField('query_id', IntegerType()),
  StructField('query', StringType()),
  StructField('query_class', StringType())
  ])

_ = (
  spark
    .read
    .csv(
      path='dbfs:/wands/downloads/query.csv',
      sep='\t',
      header=True,
      schema=queries_schema
      )
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('queries')
  )

display(
  spark.table('queries')
  )

# COMMAND ----------

# DBTITLE 1,Process Labels
labels_schema = StructType([
  StructField('id', IntegerType()),
  StructField('query_id', IntegerType()),
  StructField('product_id', IntegerType()),
  StructField('label', StringType())
  ])

_ = (
  spark
    .read
    .csv(
      path='dbfs:/wands/downloads/label.csv',
      sep='\t',
      header=True,
      schema=labels_schema
      )
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('labels')
  )

display(spark.table('labels'))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT label FROM labels GROUP BY label

# COMMAND ----------

# MAGIC %md ##ステップ3：ラベルスコアの割り当て
# MAGIC
# MAGIC クエリによって返された商品に割り当てられたテキストベースのラベルをアルゴリズムで使用するために準備するために、これらのラベルがどのように重み付けされるべきかの判断に基づいてラベルを数値スコアに変換する：
# MAGIC
# MAGIC **NOTE** [この記事](https://medium.com/@nikhilbd/how-to-measure-the-relevance-of-search-engines-18862479ebc)では、検索結果の関連性のスコアリングのアプローチ方法に関する素晴らしい議論を提供している。

# COMMAND ----------

# DBTITLE 1,Add Label Score Column to Labels Table
if 'label_score' not in spark.table('labels').columns:
  _ = spark.sql('ALTER TABLE labels ADD COLUMN label_score FLOAT')

# COMMAND ----------

# DBTITLE 1,Assign Label Scores
# MAGIC %sql
# MAGIC
# MAGIC UPDATE labels
# MAGIC SET label_score = 
# MAGIC   CASE lower(label)
# MAGIC     WHEN 'exact' THEN 1.0
# MAGIC     WHEN 'partial' THEN 0.75
# MAGIC     WHEN 'irrelevant' THEN 0.0
# MAGIC     ELSE NULL
# MAGIC     END;

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC |  WANDS | Wayfair product search relevance data | MIT  | https://github.com/wayfair/WANDS   |
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | chromadb | An open source embedding database |  Apache |  https://pypi.org/project/chromadb/  |
# MAGIC | sentence-transformers | Compute dense vector representations for sentences, paragraphs, and images | Apache 2.0 |https://pypi.org/project/sentence-transformers/ |
