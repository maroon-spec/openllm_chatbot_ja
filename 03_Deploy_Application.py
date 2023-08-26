# Databricks notebook source
# MAGIC %md このノートブックの目的は、QA Botアクセラレータを構成するノートブックを制御するさまざまな設定値を設定することです。このノートブックは https://github.com/databricks-industry-solutions/diy-llm-qa-bot から利用できます。
# MAGIC
# MAGIC クラスターは、DBR13.2ML 以降をご利用ください。
# MAGIC
# MAGIC **注意** **利用するには Serverlessタイプのモデルサービングが利用できるワークスペースをご利用ください。**
# MAGIC
# MAGIC 現在 Serverlss Model Servingが利用できるリージョンはこちらから確認できます。<br>
# MAGIC AWS: https://docs.databricks.com/resources/supported-regions.html <br>
# MAGIC Azure: https://learn.microsoft.com/en-us/azure/databricks/resources/supported-regions <br>

# COMMAND ----------

# MAGIC %md ## イントロダクション
# MAGIC
# MAGIC このノートブックでは、以前のノートブックでMLflowに登録したカスタムモデルを、Databricksのモデルサービング([AWS](https://docs.databricks.com/machine-learning/model-serving/index.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/))にデプロイします。Databricksモデルサービングは、認証されたアプリケーションがREST API経由で登録されたモデルとインタラクションできるコンテナ化されたデプロイメントオプションを提供します。これによって、MLOpsチームはモデルを簡単にデプロイ、管理し、様々なアプリケーションとこれらのモデルをインテグレーションできるようになります。
# MAGIC
# MAGIC <img src='https://sajpstorage.blob.core.windows.net/maruyama/webinar/llm/qabot-openai.png' width='800'>
# MAGIC

# COMMAND ----------

# DBTITLE 1,設定の取得
# MAGIC %run "./util/notebook-config"

# COMMAND ----------

# DBTITLE 1,インポート
import mlflow
import requests
import json
import time
from mlflow.utils.databricks_utils import get_databricks_host_creds

# COMMAND ----------

# Unity Catalogにモデルを登録する場合
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# DBTITLE 1,デプロイする最新Productionモデルのバージョンを取得
from pyspark.sql.functions import max

# mlflowに接続
client = mlflow.MlflowClient()

#latest_version = mlflow.MlflowClient().get_latest_versions(config['registered_model_name'], stages=['Production'])[0].version
registered_model_name =f"{config['catalog_name']}.{config['schema_name']}.{config['registered_model_name']}"
latest_version = client.search_model_versions("name = '%s'" % registered_model_name)[0].version

# COMMAND ----------

# MAGIC %md ##Step 1: モデルサービングエンドポイントのデプロイ
# MAGIC
# MAGIC 通常、モデルはDatabricksワークスペースのUIかREST APIを用いて、モデル共有エンドポイントにデプロイされます。我々のモデルはセンシティブな環境変数のデプロイメントに依存しているので、現在REST API経由でのみ利用できる比較的新しいモデルサービングの機能を活用する必要あります。
# MAGIC

# COMMAND ----------

registered_model_name =f"{config['catalog_name']}.{config['schema_name']}.{config['registered_model_name']}"

served_models = [
    {
      "name": "current",
      "model_name": registered_model_name,
      "model_version": latest_version,
      "workload_type": "GPU_LARGE",
      "workload_size": "Small",
      "scale_to_zero_enabled": "False"
    }
]
traffic_config = {"routes": [{"served_model_name": "current", "traffic_percentage": "100"}]}

# COMMAND ----------

# DBTITLE 1,仕様に合わせてエンドポイントを作成、更新する関数の定義
# serving_endpoint_nameの名前のエンドポイントが存在するかどうかをチェック
def endpoint_exists():
  """serving_endpoint_nameの名前のエンドポイントが存在するかどうかをチェック"""
  url = f"https://{serving_host}/api/2.0/preview/serving-endpoints/{config['serving_endpoint_name']}"
  headers = { 'Authorization': f'Bearer {creds.token}' }
  response = requests.get(url, headers=headers)
  return response.status_code == 200


# デプロイメントの準備ができるまで待ち、エンドポイント設定を返却
def wait_for_endpoint():
  """デプロイメントの準備ができるまで待ち、エンドポイント設定を返却"""
  headers = { 'Authorization': f'Bearer {creds.token}' }
  endpoint_url = f"https://{serving_host}/api/2.0/preview/serving-endpoints/{config['serving_endpoint_name']}"
  response = requests.request(method='GET', headers=headers, url=endpoint_url)
  while response.json()["state"]["ready"] == "NOT_READY" or response.json()["state"]["config_update"] == "IN_PROGRESS" : # エンドポイントの準備ができていない、あるいは、設定更新中
    print("Waiting 30s for deployment or update to finish")
    time.sleep(30)
    response = requests.request(method='GET', headers=headers, url=endpoint_url)
    response.raise_for_status()
  return response.json()

# サービングエンドポイントを作成し、準備ができるまで待つ
def create_endpoint():
  """サービングエンドポイントを作成し、準備ができるまで待つ"""
  print(f"Creating new serving endpoint: {config['serving_endpoint_name']}")
  endpoint_url = f'https://{serving_host}/api/2.0/preview/serving-endpoints'
  headers = { 'Authorization': f'Bearer {creds.token}' }
  request_data = {"name": config['serving_endpoint_name'] , "config": {"served_models": served_models}}
  json_bytes = json.dumps(request_data).encode('utf-8')
  response = requests.post(endpoint_url, data=json_bytes, headers=headers)
  print(json.dumps(response.json(), indent=4))
  response.raise_for_status()
  wait_for_endpoint()
  displayHTML(f"""Created the <a href="/#mlflow/endpoints/{config['serving_endpoint_name']}" target="_blank">{config['serving_endpoint_name']}</a> serving endpoint""")

# サービングエンドポイントを更新し、準備ができるまで待つ
def update_endpoint():
  """サービングエンドポイントを更新し、準備ができるまで待つ"""
  print(f"Updating existing serving endpoint: {config['serving_endpoint_name']}")
  endpoint_url = f"https://{serving_host}/api/2.0/preview/serving-endpoints/{config['serving_endpoint_name']}/config"
  headers = { 'Authorization': f'Bearer {creds.token}' }
  request_data = { "served_models": served_models, "traffic_config": traffic_config }
  json_bytes = json.dumps(request_data).encode('utf-8')
  response = requests.put(endpoint_url, data=json_bytes, headers=headers)
  response.raise_for_status()
  wait_for_endpoint()
  displayHTML(f"""Updated the <a href="/#mlflow/endpoints/{config['serving_endpoint_name']}" target="_blank">{config['serving_endpoint_name']}</a> serving endpoint""")

# COMMAND ----------

# DBTITLE 1,エンドポイントの作成、更新に定義した関数を使用
# APIが必要とするその他の入力を収集
serving_host = spark.conf.get("spark.databricks.workspaceUrl")
creds = get_databricks_host_creds()

# エンドポイントの作成/更新のスタート
if not endpoint_exists():
  create_endpoint()
else:
  update_endpoint()

# COMMAND ----------

# MAGIC %md 
# MAGIC 作成したばかりのモデルサービングエンドポイントにアクセスするには上のリンクを使用できます。 
# MAGIC
# MAGIC <img src='https://github.com/databricks-industry-solutions/diy-llm-qa-bot/raw/main/image/model_serving_ui.png'>

# COMMAND ----------

# MAGIC %md ##Step 2: エンドポイントAPIのテスト

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC 1. 左メニューの **サービング** から、作成したサービングエンドポイントを選択します。<br>
# MAGIC 1. 右上の**サービングエンドポイントにクエリー** ボタンをクリックします<br>
# MAGIC 1. 以下のサンプル問い合わせを入力し、結果をチェックします。 
# MAGIC
# MAGIC ```
# MAGIC {"inputs" : 
# MAGIC   {"question" : ["レイクハウスとは？"]}
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC 以下のようにレスポンスが返ってくれば成功です。
# MAGIC
# MAGIC <img src='https://sajpstorage.blob.core.windows.net/maruyama/webinar/llm/query_test.png'  width='800'　>
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC いくかの制限があります：
# MAGIC - エンドポイントをゼロまでスケールするようにすると、クエリーがない際のbotのコストを削減できます。しかし、長い期間の後の最初のリクエストは、エンドポイントがゼロノードからスケールアップする必要があるため、数分を要します。
# MAGIC - サーバレスモデルサービングリクエストのタイムアウトリミットは60秒です。同じリクエストで3つの質問が送信されると、モデルがタイムアウトすることがあります。

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | tiktoken | Fast BPE tokeniser for use with OpenAI's models | MIT  |   https://pypi.org/project/tiktoken/ |
# MAGIC | faiss-cpu | Library for efficient similarity search and clustering of dense vectors | MIT  |   https://pypi.org/project/faiss-cpu/ |
# MAGIC | openai | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/openai/ |
