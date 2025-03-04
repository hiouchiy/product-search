# Databricks notebook source
# MAGIC %md The purpose of this notebook is to deploy our model as part of the Product Search accelerator.  You may find this notebook on https://github.com/databricks-industry-solutions/product-search.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC At this point we have a basic model as well as a tuned model, packaged with access to product embeddings and ready for deployment.  In this notebook, we will show how this model can be deployed using [Databricks Model Serving](https://docs.databricks.com/machine-learning/model-serving/index.html) so that applications can simply call a REST API to perform a search in real-time.
# MAGIC
# MAGIC <img src='https://github.com/databricks-industry-solutions/product-search/raw/main/images/inference.png' width=800>
# MAGIC
# MAGIC
# MAGIC
# MAGIC はじめに
# MAGIC
# MAGIC この時点で、私たちは基本モデルとチューニングされたモデルを手に入れ、製品エンベッディングにアクセスできるようにパッケージ化し、デプロイする準備が整いました。 このノートブックでは、このモデルを[Databricks Model Serving](https://docs.databricks.com/machine-learning/model-serving/index.html)を使ってデプロイし、アプリケーションがREST APIを呼び出すだけでリアルタイムに検索を実行できるようにする方法を示します。
# MAGIC
# MAGIC <img src='https://github.com/databricks-industry-solutions/product-search/raw/main/images/inference.png' width=800>

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import mlflow
import os
import requests
import pandas as pd
import json
import time

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./00_Intro_and_Config"

# COMMAND ----------

# MAGIC %md ##Step 1: Review Model Names
# MAGIC
# MAGIC If we have successfully run the last two notebooks, we should have two models deployed to the MLflow model registry, each of which has been elevated to Production status.  You can select one or the other for Step 2:
# MAGIC
# MAGIC
# MAGIC ステップ 1：モデル名のレビュー
# MAGIC
# MAGIC 最後の 2 つのノートブックを成功裏に実行した場合、MLflow モデル・レジストリにデプロイされた 2 つのモデル があり、それぞれ本番ステータスに昇格しているはずです。 ステップ 2 では、どちらか一方を選択します：

# COMMAND ----------

# DBTITLE 1,Print Model Names
print(f"Basic Model: {config['basic_model_name']}")
print(f"Tuned Model: {config['tuned_model_name']}")

# COMMAND ----------

# DBTITLE 1,Select a model 
model_name = config['tuned_model_name']

# identify model version in registry
model_version = mlflow.tracking.MlflowClient().get_latest_versions(name = model_name, stages = ["Production"])[0].version

# COMMAND ----------

# MAGIC %md ##Step 2: Deploy Model to Model Serving Endpoint
# MAGIC
# MAGIC To deploy our model, we need to reconfigure our Databricks workspace for Machine Learning.  We can do this by clicking on the drop-down at the top of the left-hand navigation bar and selecting *Machine Learning*.
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/search_change_workspace.png'>
# MAGIC
# MAGIC Once we've done that, we should be able to select *Serving* from that same left-hand navigation bar.
# MAGIC
# MAGIC Within the Serving Endpoints page, click on the *Create Serving Endpoint* button.  Give your endpoint a name, select your model - it may help to start typing the model name to limit the search - and then select the model version.  Select the compute size based on the number of requests expected and select/deselect the *scale to zero* option based on whether you want the service to scale down completely during a sustained period of inactivity.  (Spinning back up from zero does take a little time once a request has been received.)
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/search_create_serving_endpoint2.png' width=90%>
# MAGIC
# MAGIC Click the *Create serving endpoint* button to deploy the endpoint and monitor the deployment process until the *Serving Endpoint State* is *Ready*:
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/search_endpoint_ready.png' width=90%>
# MAGIC
# MAGIC
# MAGIC Before leaving this page, be sure to click the *Query Endpoint* button in the upper right-hand corner.  In the resulting pane, click on the *Python* tab and copy the displayed code to the cell below:
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/search_query_endpoint.PNG' width=50%>
# MAGIC
# MAGIC
# MAGIC
# MAGIC ステップ2：モデルをModel Serving Endpointにデプロイする
# MAGIC
# MAGIC モデルをデプロイするために、DatabricksワークスペースをMachine Learning用に再設定する必要があります。 左側のナビゲーションバーの上部にあるドロップダウンをクリックし、*Machine Learning*を選択することでこれを行うことができます。
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/search_change_workspace.png'>
# MAGIC
# MAGIC そうしたら、同じ左側のナビゲーション・バーから *Serving* を選択できるはずです。
# MAGIC
# MAGIC Serving Endpoints」ページで、「*Create Serving Endpoint*」ボタンをクリックします。 エンドポイントに名前を付け、モデルを選択し（検索を制限するためにモデル名を入力し始めるとよいでしょう）、モデルのバージョンを選択します。 予想されるリクエスト数に基づいて計算サイズを選択し、持続的な非アクティブ期間中にサービスを完全にスケールダウンさせたいかどうかに基づいて、*scale to zero* オプションを選択/選択解除します。 (ゼロからスピンアップし直すには、リクエストを受け取ると少し時間がかかります)。
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/search_create_serving_endpoint2.png' width=90%> をクリックします。
# MAGIC
# MAGIC Create serving endpoint*ボタンをクリックしてエンドポイントをデプロイし、*Serving Endpoint State*が*Ready*になるまでデプロイプロセスを監視します：
# MAGIC
# MAGIC
# MAGIC </p> <img src='https://brysmiwasb.blob.core.windows.net/demos/images/search_endpoint_ready.png' width=90%>
# MAGIC
# MAGIC
# MAGIC このページを離れる前に、必ず右上隅の *Query Endpoint* ボタンをクリックしてください。 表示されたペインで、*Python*タブをクリックし、表示されたコードを下のセルにコピーしてください：
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/search_query_endpoint.PNG' width=50%>

# COMMAND ----------

# MAGIC %md Alternatively, we can use the API instead of the UI to [set up the model serving endpoint](https://docs.databricks.com/machine-learning/model-serving/create-manage-serving-endpoints.html#create-model-serving-endpoints). The effect of the following 3 blocks of code is equivalent to the steps described above. We provide this option to showcase automation and to make sure that this notebook can be consistently executed end-to-end without requiring manual intervention.
# MAGIC
# MAGIC To use the Databricks API, you need to create environmental variables named *DATABRICKS_URL* and *DATABRICKS_TOKEN* which must be your workspace url and a valid [personal access token](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/authentication). We have retrieved and set these values up for you in notebook *00* as part of the *config* setting.
# MAGIC
# MAGIC
# MAGIC あるいは、UIの代わりにAPIを使って[モデル提供エンドポイントを設定する](https://docs.databricks.com/machine-learning/model-serving/create-manage-serving-endpoints.html#create-model-serving-endpoints)こともできます。以下の3つのコードのブロックの効果は、上で説明したステップと同等です。このオプションを提供するのは、自動化を紹介するためと、このノートブックが手動介入を必要とせずにエンドツーエンドで一貫して実行できることを確認するためです。
# MAGIC
# MAGIC Databricks APIを使用するには、*DATABRICKS_URL*と*DATABRICKS_TOKEN*という環境変数を作成する必要があります。これは、ワークスペースのURLと有効な[パーソナルアクセストークン](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/authentication)でなければなりません。これらの値は、ノートブック *00* で *config* 設定の一部として取得され、設定されています。

# COMMAND ----------

# MAGIC %run ./util/create-update-serving-endpoint

# COMMAND ----------

# DBTITLE 1,Define model serving config
served_models = [
    {
      "name": "Product-Search",
      "model_name": model_name,
      "model_version": model_version,
      "workload_size": "Medium",
      "scale_to_zero_enabled": True
    }
]
traffic_config = {"routes": [{"served_model_name": "Product-Search", "traffic_percentage": "100"}]}

# COMMAND ----------

# DBTITLE 1,Create or update model serving endpoint
# kick off endpoint creation/update
if not endpoint_exists(config['serving_endpoint_name']):
  create_endpoint(config['serving_endpoint_name'], served_models)
else:
  update_endpoint(config['serving_endpoint_name'], served_models)

# COMMAND ----------

# MAGIC %md ##Step 3: Test the Model Serving Endpoint
# MAGIC
# MAGIC With the code for testing our endpoint in the cell below, we can now prepare to submit data against our endpoint:
# MAGIC
# MAGIC
# MAGIC ステップ 3: モデル提供エンドポイントのテスト
# MAGIC
# MAGIC 下のセルにエンドポイントをテストするコードがあるので、これでエンドポイントに対してデータを送信する準備ができます：

# COMMAND ----------

# DBTITLE 1,Query the Endpoint Code
import os
import requests
import numpy as np
import pandas as pd
import json

endpoint_url = f"""{config['databricks url']}/serving-endpoints/{config['serving_endpoint_name']}/invocations"""

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = endpoint_url
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')

  return response.json()

# COMMAND ----------

# MAGIC %md And now we can test the endpoint:
# MAGIC
# MAGIC これでエンドポイントをテストできる：

# COMMAND ----------

# DBTITLE 1,Test the Model Serving Endpoint
score_model( 
  pd.DataFrame({'query':['kid-proof rug']})
)

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC |  WANDS | Wayfair product search relevance data | MIT  | https://github.com/wayfair/WANDS   |
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | chromadb | An open source embedding database |  Apache |  https://pypi.org/project/chromadb/  |
# MAGIC | sentence-transformers | Compute dense vector representations for sentences, paragraphs, and images | Apache 2.0 |https://pypi.org/project/sentence-transformers/ |
