# Databricks notebook source
if 'config' not in locals():
  config = {}

# COMMAND ----------

# DBTITLE 1,ドキュメントパスの設定
# VectorDBを保存するパスを指定します。
config['vector_store_path'] = '/dbfs/tmp/qabot_openllm/vector_store' 

# COMMAND ----------

# DBTITLE 1,カタログ＆データベースの作成
config['catalog_name'] = 'jmaru_catalog'
config['schema_name'] = 'qabot_openllm_ja'

# create catalog & database if not exists
_ = spark.sql(f"create catalog if not exists {config['catalog_name']}")
_ = spark.sql(f"use catalog {config['catalog_name']}")
_ = spark.sql(f"create schema if not exists {config['schema_name']}")

# set current datebase context
_ = spark.sql(f"use {config['catalog_name']}.{config['schema_name']}")

# COMMAND ----------

# DBTITLE 1,mlflowの設定
import mlflow
# Model　Nameの指定
config['registered_model_name'] = 'databricks_openllm_qabot_jpn' 

# 以下は設定不要です。
config['model_uri'] = f"models:/{config['registered_model_name']}/production"
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
_ = mlflow.set_experiment('/Users/{}/{}'.format(username, config['registered_model_name']))

# COMMAND ----------

# DBTITLE 1,LLMモデルの設定
# 適宜、プロンプトなど変更ください。このままでも動作します。

# エンべディング変換を担うLLM
config['hf_embedding_model'] = 'sonoisa/t5-base-japanese'
config['hf_chat_model'] = 'mosaicml/mpt-7b-instruct'

# プロンプトテンプレート
config['prompt_template'] = """Below is an instruction with a question in it. Write a response that appropriately completes the request. 　### Instruction: You are a competent assistant developed by Databricks and are adept at answering questions based on the context specified, and context is a document. If the context does not provide enough information to determine an answer, say I don't know. If the context is not appropriate for the question, say I don't know. If the context does not provide a good answer, say you do not know. You do not repeat the same words. The following is a combination of a contextual document and a contextual question. Give an answer that adequately satisfies the question by summarizing the document. ### Context {context} ### Question: {question}  ### Response: """
config['temperature'] = 0.15

# COMMAND ----------

# DBTITLE 1,デプロイメントの設定
# Model Serving Endpoint Nameを指定ください。
config['serving_endpoint_name'] = "openllm-qabot-endpoint-jpn" 

# COMMAND ----------


