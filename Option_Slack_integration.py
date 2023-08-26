# Databricks notebook source
# MAGIC %md ## (オプション) 最後にSlackと連携する例をご紹介します。
# MAGIC
# MAGIC こちらのQiita記事: [Databricksで作成したLLM QA Chatbot をSlackと連携してみた](https://qiita.com/maroon-db/items/d05e1c7dc208228e004f) にある通り、SlackやTeamsなどのツールと連携することでユーザーから使いやすくすることができます。
# MAGIC
# MAGIC 今回はそのサンプルコードとSlack用のアプリケーションコードをノートブックで実行してみます。<br>
# MAGIC こちらのサンプルコードは動作を保証するものではありませんのでご留意ください。
# MAGIC
# MAGIC 以下の方法は、インタラクティブな開発時での使用が推奨されています。<br>
# MAGIC 本番環境では上記の記事を参考に、別の環境でSlack Volt用のコードを実行してください。

# COMMAND ----------

!pip install slack_bolt databricks-sql-connector openai langchain==0.0.205

# COMMAND ----------

# DBTITLE 1,設定の取得
# MAGIC %run "./util/notebook-config"

# COMMAND ----------

import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from PIL import Image
import base64, io, requests, json
import numpy as np
import pandas as pd
from mlflow.utils.databricks_utils import get_databricks_host_creds

# COMMAND ----------

# MAGIC %md ## 必要なトークンの設定
# MAGIC
# MAGIC Databricks Secretsを使って、Tokenを保護してください。
# MAGIC [Secrets の作成方法](https://qiita.com/maroon-db/items/6e2d86919a827bd61a9b)

# COMMAND ----------

DATABRICKS_TOKEN = dbutils.secrets.get("fieldeng", "modelserving-token") 

SLACK_BOT_TOKEN = dbutils.secrets.get("fieldeng", "slack_chatbot") 
SLACK_APP_TOKEN = dbutils.secrets.get("fieldeng", "slack_chatapp") 

# COMMAND ----------

# MAGIC %md ## Databricks Model Endpointとの接続設定
# MAGIC
# MAGIC Model Serving は Serverless の Model Serving を指定する必要があります。

# COMMAND ----------

# Model Serving Endpoint URL　& Token 取得
host = spark.conf.get("spark.databricks.workspaceUrl")
MODEL_SERVING_URL= f"https://{host}/serving-endpoints/{config['serving_endpoint_name']}/invocations"


## Databricks endpoint 連携の設定
def create_tf_serving_json(data):
    return {
        "inputs": {name: data[name].tolist() for name in data.keys()}
        if isinstance(data, dict)
        else data.tolist()
    }

def score_model(question):
  token = DATABRICKS_TOKEN
  url = MODEL_SERVING_URL

  headers = {'Authorization': f'Bearer {token}', "Content-Type": "application/json",}
  dataset = pd.DataFrame({'question':[question]})
  ds_dict = (
        {"dataframe_split": dataset.to_dict(orient="split")}
        if isinstance(dataset, pd.DataFrame)
        else create_tf_serving_json(dataset)
    )
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method="POST", headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception( f"Request failed with status {response.status_code}, {response.text}")
  return response.json()

# COMMAND ----------

# MAGIC %md ## Slack のBotと連携
# MAGIC
# MAGIC Slack Appや Botの追加についてはこちらの記事をご覧ください。<br>
# MAGIC https://qiita.com/maroon-db/items/d05e1c7dc208228e004f

# COMMAND ----------

app = App(token=SLACK_BOT_TOKEN)

# mentionされた場合メッセージを読み取ります
@app.event("app_mention")
def respond_to_mention(body, say):
    channel_id = body["event"]["channel"]
    user_id = body["event"]["user"]
    question = body["event"]["text"]

    response = score_model(question)
    answer = response['predictions'][0]["answer"]
    source = response['predictions'][0]["source"]

    # Send a preparatory message
    #say("回答中・・・, 少々お待ちください.", channel=channel_id)

    # Craft your response message
    message = f"{answer}.  \n Source: {source}"

    # Send the response message
    say(message, channel=channel_id)

# COMMAND ----------

# MAGIC %md ## 以下を実行すると、Bolt appが起動します。
# MAGIC
# MAGIC 以下の方法は、インタラクティブな開発時での使用が推奨されています。<br>

# COMMAND ----------

from flask import Flask, request

flask_app = Flask("dbchain")
handler = SocketModeHandler(app, SLACK_APP_TOKEN)

@flask_app.route('/', methods=['POST'])
def slack_events():
  return handler.start(request)

if __name__ == '__main__':
    handler.start()
    flask_app.run(host="0.0.0.0", port="7777")

