# Databricks notebook source
# MAGIC %md このノートブックの目的は、Product Search ソリューション アクセラレータで使用される設定データへのアクセスを提供することです。 このノートブックは https://github.com/databricks-industry-solutions/product-search にあります。

# COMMAND ----------

# MAGIC %md ##はじめに
# MAGIC
# MAGIC このソリューション・アクセラレータの目的は、大規模言語モデル（LLM）とその小さな同胞が、商品検索を可能にするためにどのように使用できるかを示すことです。 今日ほとんどのサイトで使われている、キーワードのマッチに依存する商品検索とは異なり、LLMは一般的にセマンティック検索と呼ばれる、単語の*概念的な類似性*が重要となる検索を可能にします。
# MAGIC
# MAGIC 単語間の*概念的類似性*に関するモデルの知識は、様々な文書に触れ、それらの文書から、特定の単語が互いに密接な関係を持つ傾向があることを学習することから得られる。 例えば、ある文書が*子供（Children）*にとっての遊びの重要性を論じ、*子供（Child）*という用語を使うことで、*子供（Children）*と*子供（Child）*には何らかの関係があることをモデルに教えるかもしれません。 他の文書では、これらの用語が似たような近さで使われ、同じトピックを論じる他の文書では、*kid*または*kids*という用語が導入されるかもしれません。 文書によっては4つの用語がすべて出てくる可能性もありますが、たとえそのようなことがなかったとしても、これらの用語を囲む単語には十分な重複があり、モデルはこれらすべての用語の間に密接な関連があると認識するようになります。
# MAGIC
# MAGIC オープンソースコミュニティから入手可能なLLMの多くは、事前に学習されたモデルとして提供されており、このような単語の関連は、一般に入手可能な幅広い情報からすでに学習されています。これらのモデルがすでに蓄積している知識を使って、商品カタログの商品説明テキストを検索し、ユーザーが入力した検索語やフレーズと一致すると思われる商品を探すことができます。サイト上で紹介される商品が、小売業者や紹介するサプライヤーのトーンやスタイルを反映した独自の関連パターンを持つ、より具体的な用語のセットを使用する傾向がある場合、これらのモデルは、使用されている言語に対する理解を形成するために、サイトに固有の追加データに触れることができます。 この*ファインチューニング*エクササイズは、既製のモデルを特定の製品カタログのニュアンスに合わせるために使用することができ、より効果的な検索結果を可能にします。
# MAGIC
# MAGIC このソリューションアクセラレータでは、既製のモデルと特定の商品テキストにチューニングしたモデルの両方のバージョンを紹介します。そして、ユーザーがDatabricks環境を通してセマンティック検索機能をどのようにデプロイできるかを確認できるように、モデルのデプロイメントに関する問題に取り組みます。
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/search_simple_architecture.png' width=800></img>
# MAGIC
# MAGIC これを探求するにあたり、GPUベースのクラスタとDatabricksのモデル提供機能をサポートするDatabricksワークスペースで実行する必要があることを認識することが重要です。 GPUクラスタの利用可否は、クラウドプロバイダと、そのプロバイダによってクラウドサブスクリプションに割り当てられたクォータに依存します。 Databricks モデルサービング機能のご利用は、現在以下の [AWS](https://docs.databricks.com/machine-learning/model-serving/index.html#limitations) および [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/#limitations) リージョンに限定されています。
# MAGIC
# MAGIC **注意** このソリューションでは、シングルノードクラスターを利用することができます。 必ずGPU対応クラスタ（および対応するDatabricks MLランタイム）を選択してください。 ノードサイズが大きいほど、より集中的なステップのパフォーマンスが向上します。

# COMMAND ----------

# MAGIC %md ##設定
# MAGIC
# MAGIC 以下のパラメータは、ノートブック全体で使用されるリソースを制御するために使用されます。 これらの変数を変更した場合、ノートブック内のマークダウンはこれらに関連する元の値を参照する可能性があることに注意してください：

# COMMAND ----------

# DBTITLE 1,設定変数の初期化
if 'config' not in locals().keys():
  config = {}

# COMMAND ----------

# DBTITLE 1,Database
config['database'] = 'wands'

# create database if not exists
_ = spark.sql('create database if not exists {0}'.format(config['database']))

# set current datebase context
_ = spark.catalog.setCurrentDatabase(config['database'])

# COMMAND ----------

# DBTITLE 1,Storage
config['dbfs_path'] = '/wands'

# COMMAND ----------

# DBTITLE 1,Models
config['basic_model_name'] = 'wands_basic_search'
config['tuned_model_name'] = 'wands_tuned_search'

# COMMAND ----------

# DBTITLE 1,DatabricksのURLとトークン
import os
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
config['databricks token'] = ctx.apiToken().getOrElse(None)
config['databricks url'] = ctx.apiUrl().getOrElse(None)
os.environ['DATABRICKS_TOKEN'] = config["databricks token"]
os.environ['DATABRICKS_URL'] = config["databricks url"]

# COMMAND ----------

# DBTITLE 1,mlflow experiment
import mlflow
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment('/Users/{}/prod_search'.format(username))

# COMMAND ----------

# DBTITLE 1,Model serving endpoint
config['serving_endpoint_name'] = 'wands_search'

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC |  WANDS | Wayfair product search relevance data | MIT  | https://github.com/wayfair/WANDS   |
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | chromadb | An open source embedding database |  Apache |  https://pypi.org/project/chromadb/  |
# MAGIC | sentence-transformers | Compute dense vector representations for sentences, paragraphs, and images | Apache 2.0 |https://pypi.org/project/sentence-transformers/ |
