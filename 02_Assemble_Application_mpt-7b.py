# Databricks notebook source
# MAGIC %md このノートブックの目的は、QA Botアクセラレータを構成するノートブックを制御するさまざまな設定値を設定することです。このノートブックは https://github.com/databricks-industry-solutions/diy-llm-qa-bot から利用できます。
# MAGIC
# MAGIC クラスターは、DBR13.2ML 以降をご利用ください。

# COMMAND ----------

# MAGIC %md ## イントロダクション
# MAGIC
# MAGIC ドキュメントのインデックスを作成したので、コアアプリケーションのロジックの構築にフォーカスすることができます。このロジックは、ユーザーによる質問に基づいてベクトルストアからドキュメントを取得します。ドキュメントと質問にコンテキストが追加され、レスポンスを生成するためにモデルに送信されるプロンプトを構成するためにそれらを活用します。</p>
# MAGIC
# MAGIC <!---<img src='https://brysmiwasb.blob.core.windows.net/demos/images/bot_application.png' width=900> -->
# MAGIC <img src='https://sajpstorage.blob.core.windows.net/maruyama/webinar/llm/llm-flow.png' width=1100>
# MAGIC
# MAGIC
# MAGIC </p>
# MAGIC このノートブックでは、最初に何が行われているのかを把握するために一度ステップをウォークスルーします。そして、我々の作業をより簡単にカプセル化するためにクラスオブジェクトとしてこのロジックを再度パッケージングします。そして、このアクセラレーターの最後のノートブックで、モデルのデプロイをアシストするMLflowの中にモデルとしてこのオブジェクトを永続化します。

# COMMAND ----------

# DBTITLE 1,必要ライブラリのインストール
# MAGIC %pip install langchain==0.0.166 tiktoken==0.4.0 faiss-cpu==1.7.4
# MAGIC %pip install transformers==4.28.0 mlflow
# MAGIC %pip install sentence_transformers fugashi ipadic
# MAGIC %pip install xformers einops flash-attn==v1.0.3.post0 triton==2.0.0.dev20221202
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,必要ライブラリのインポート
import re, time, torch
import pandas as pd

import transformers 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, PreTrainedModel, PreTrainedTokenizer

from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings

import mlflow
from mlflow.models.signature import infer_signature

from langchain.vectorstores.faiss import FAISS
from langchain.schema import BaseRetriever
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain import LLMChain, PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain

from huggingface_hub import snapshot_download

# COMMAND ----------

# DBTITLE 1,設定の取得
# MAGIC %run "./util/notebook-config"

# COMMAND ----------

# MAGIC %md ##Step 1: 回答生成の探索
# MAGIC
# MAGIC まず初めに、どのようにしてユーザーが指定した質問に反応して回答を導き出すのかを探索しましょう。

# COMMAND ----------

# DBTITLE 1,環境変数呼び出し
repository = f"{config['catalog_name']}.{config['schema_name']}"
mlflow_model_name = config['registered_model_name']

# COMMAND ----------

# DBTITLE 1,適切なドキュメントの取得
# エンべディングにアクセスするためにベクトルストアをオープン
# embeddings = OpenAIEmbeddings(model=config['openai_embedding_model'])
embeddings = HuggingFaceEmbeddings(model_name=config['hf_embedding_model'])
vector_store = FAISS.load_local(embeddings=embeddings, folder_path=config['vector_store_path'])

# ドキュメント取得の設定 
n_documents = 5 # 取得するドキュメントの数 
retriever = vector_store.as_retriever(search_kwargs={'k': n_documents}) # 取得メカニズムの設定

# 適切なドキュメントの取得テスト
question = "Delta Lakeとは何ですか？"
docs = retriever.get_relevant_documents(question)
for doc in docs: 
  print(doc,'\n') 

# COMMAND ----------

# MAGIC %md ## ChatLLM の設定
# MAGIC
# MAGIC 今回利用するチャットLLMを設定し、pipeline化しておきます。今回は高速化のため tritonを利用するように設定しております。
# MAGIC パイプライン内で設定したパラメータはチューニングポイントになりますので、色々と試してください。
# MAGIC
# MAGIC またこのプロンプトには、ユーザーが送信する *question* と、回答の *context* を提供すると信じるドキュメントのプレースホルダーが必要です。
# MAGIC
# MAGIC プロンプトは複数のプロンプト要素から構成され、[prompt templates](https://python.langchain.com/en/latest/modules/prompts/chat_prompt_template.html)を用いて定義されることに注意してください。簡単に言えば、プロンプトテンプレートによって、プロンプトの基本的な構造を定義し、レスポンスをトリガーするために容易に変数データで置き換えることができるようになります。ここで示しているシステムメッセージは、モデルにどのように反応して欲しいのかの指示を当てます。人間によるメッセージテンプレートは、ユーザーが発端となるリクエストに関する詳細情報を提供します。
# MAGIC
# MAGIC プロンプトに対するレスポンスを行うモデルに関する詳細とプロンプトは、[LLMChain object](https://python.langchain.com/en/latest/modules/chains/generic/llm_chain.html)にカプセル化されます。このオブジェクトはクエリーの解決とレスポンスの返却に対する基本構造をシンプルに定義します：

# COMMAND ----------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# promt templateの読み込み
template = config['prompt_template']

# tritonベースのFlashAttentionを使用するように変更
model_name = config['hf_chat_model']
config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.attn_config['attn_impl'] = 'triton'
config.init_device = 'cuda'

model = transformers.AutoModelForCausalLM.from_pretrained(
  model_name,
  config=config,
  torch_dtype=torch.bfloat16,
  trust_remote_code=True,
  cache_dir="/local_disk0/.cache/huggingface/"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
end_key_token_id = tokenizer.encode("<|endoftext|>")[0]

instruct_pipeline = pipeline("text-generation", 
                             model=model, 
                             tokenizer=tokenizer, 
                             device=device, 
                             max_new_tokens=200, 
                             torch_dtype=torch.bfloat16,
                             top_p=0.7, 
                             top_k=50, 
                             pad_token_id=end_key_token_id, 
                             eos_token_id=end_key_token_id, 
                             use_cache = True)

llm = HuggingFacePipeline(pipeline=instruct_pipeline)

# Prompt設定とLLMChainによるカプセル化
prompt = PromptTemplate(template=template, input_variables=["context", "question"])
qa_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)


# COMMAND ----------

# MAGIC %md 
# MAGIC 実際にレスポンスをトリガーするには、適合性の高いものから低いものにドキュメントをループし、レスポンスを導き出すことを試みます。
# MAGIC
# MAGIC 次のセルでは、タイムアウトのハンドリングや、モデルからのレスポンスの検証を行なっていないことに注意してください。アプリケーションクラスを構築する際にはこのロジックをもっと堅牢にしますが、ここではコードを読みやすくするためにシンプルにしています：

# COMMAND ----------

# DBTITLE 1,レスポンスの生成テスト
# 指定されたドキュメントのそれぞれに対して
for doc in docs:

  # ドキュメントテキストの取得
  text = doc.page_content 

  # レスポンスの生成
  output = qa_chain.generate([{'context': text, 'question': question}])
 
  # 結果から回答の取得
  generation = output.generations[0][0]
  answer = generation.text
  link = doc.metadata['source']

  # 回答の表示
  if answer is not None:
    print(f"Question: {question}", '\n', f"Answer: {answer}")
    print(f"参考リンク： {link}")
    break

# COMMAND ----------

# MAGIC %md ##Step 2: デプロイするモデルの構築
# MAGIC
# MAGIC レスポンス生成に関連する基本的なステップを探索したら、デプロイメントを容易にするためにクラスの中にロジックをラップしましょう。我々のクラスは、LLMモデル定義、ベクトルストアの収集器、クラスに対するプロンプトを渡すことでインスタンスを生成します。*get_answer*メソッドは、質問を送信してレスポンスを取得するための主要なメソッドとして機能します：

# COMMAND ----------

# DBTITLE 1,QABotクラスの定義
import csv
from pyspark.sql import Row

class QABot():
  
  def __init__(self, llm, retriever, prompt):
    self.llm = llm
    self.retriever = retriever
    self.prompt = prompt
    self.qa_chain = LLMChain(llm = self.llm, prompt=prompt)
    self.abbreviations = { # 置換したい既知の略語
      "DBR": "Databricks Runtime",
      "ML": "Machine Learning",
      "UC": "Unity Catalog",
      "DLT": "Delta Live Table",
      "DBFS": "Databricks File Store",
      "HMS": "Hive Metastore",
      "UDF": "User Defined Function"
      } 


  def _is_good_answer(self, answer):

    ''' 回答が妥当かをチェック '''

    result = True # 初期値

    badanswer_phrases = [ # モデルが回答を生成しなかったことを示すフレーズ
      "わかりません", "コンテキストがありません", "知りません", "答えが明確でありません", "すみません", 
      "答えがありません", "説明がありません", "リマインダー", "コンテキストが提供されていません", "有用な回答がありません", 
      "指定されたコンテキスト", "有用でありません", "適切ではありません", "質問がありません", "明確でありません",
      "十分な情報がありません", "適切な情報がありません", "直接関係しているものが無いようです"
      ]
    
    if answer is None: # 回答がNoneの場合は不正な回答
      results = False
    else: # badanswer phraseを含んでいる場合は不正な回答
      for phrase in badanswer_phrases:
        if phrase in answer.lower():
          result = False
          break
    
    return result


  def _get_answer(self, context, question, timeout_sec=60):

    '''' タイムアウトハンドリングありのLLMからの回答取得 '''

    # デフォルトの結果
    result = None

    # 終了時間の定義
    end_time = time.time() + timeout_sec

    # タイムアウトに対するトライ
    while time.time() < end_time:

      # レスポンス取得の試行
      try: 
        result =  qa_chain.generate([{'context': context, 'question': question}])
        break # レスポンスが成功したらループをストップ

      # レートリミットのエラーが起きたら...
      except openai.error.RateLimitError as rate_limit_error:
        if time.time() < end_time: # 時間があるのであればsleep
          time.sleep(2)
          continue
        else: # そうでなければ例外を発生
          raise rate_limit_error

      # その他のエラーでも例外を発生
      except Exception as e:
        print(f'LLM QA Chain encountered unexpected error: {e}')
        raise e

    return result


  def get_answer(self, question):
    ''' 指定された質問の回答を取得 '''

    # デフォルトの結果
    result = {'answer':None, 'source':None, 'output_metadata':None}

    # 質問から一般的な略語を置換
    for abbreviation, full_text in self.abbreviations.items():
      pattern = re.compile(fr'\b({abbreviation}|{abbreviation.lower()})\b', re.IGNORECASE)
      question = pattern.sub(f"{abbreviation} ({full_text})", question)

    # VectorStoreから類似ドキュメントの取得
    docs = self.retriever.get_relevant_documents(question)

    # それぞれのドキュメントごとにいい回答が出るまでループ
    for doc in docs:

      # ドキュメントのキー要素を取得（ textとmetadata)
      text = doc.page_content
      source = doc.metadata['source']

      # LLMから回答を取得
      output = self._get_answer(text, question)
 
      # 回答結果からアウトプット(textとmetadata)を取得
      generation = output.generations[0][0]
      answer = generation.text
      output_metadata = output.llm_output

      # no_answer ではない場合には結果を構成
      if self._is_good_answer(answer):
        result['answer'] = answer
        result['source'] = source
        result['output_metadata'] = output_metadata
        break # 良い回答であればループをストップ
  

    return result

# COMMAND ----------

# MAGIC %md 
# MAGIC これで、以前インスタンス化したオブジェクトを用いてクラスをテストすることができます：

# COMMAND ----------

# DBTITLE 1,QABotクラスのテスト
# botオブジェクトのインスタンスを作成
qabot = QABot(llm, retriever, prompt)

# 質問に対するレスポンスの取得
qabot.get_answer("MLflowの特徴は？") 

# COMMAND ----------

# MAGIC %md ##Step 3: MLflowにモデルを永続化
# MAGIC
# MAGIC 我々のbotクラスが定義、検証されたので、MLflowにこれを永続化します。MLflowはモデルのトラッキングとロギングのためのオープンソースのリポジトリです。Databricksプラットフォームにはデフォルトでデプロイされており、簡単にモデルを記録することができます。
# MAGIC
# MAGIC 今では、MLflowはOpenAIとLangChainの両方のモデルフレーバーを[サポート](https://www.databricks.com/blog/2023/04/18/introducing-mlflow-23-enhanced-native-llm-support-and-new-features.html)していますが、我々のbotアプリケーションではカスタムロジックを記述しているので、より汎用的な[pyfunc](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#creating-custom-pyfunc-models)モデルフレーバーを活用しなくてはなりません。このpyfuncモデルフレーバーによって、標準的なMLflowのデプロイメントメカニズムを通じてデプロイされた際に、モデルがどのように反応するのかに関して非常に多くのコントロールを行えるようになります。
# MAGIC
# MAGIC カスタムMLflowモデルを作成するのに必要なことは、*mlflow.pyfunc.PythonModel*タイプのカスタムラッパーを定義することだけです。 *\_\_init__* メソッドは、*QABot*クラスのインスタンスを初期化し、クラス変数に永続化します。そして、 *predict* メソッドは、レスポンス生成の標準的なインタフェースとして動作します。このメソッドはpandasデータフレームとして入力を受け付けますが、ユーザーから一度に一つの質問を受け取るという知識を用いてロジックを記述することができます：

# COMMAND ----------

# DBTITLE 1,モデルのMLflowラッパーの定義
class MLflowQABot(mlflow.pyfunc.PythonModel):

  def __init__(self, llm, retriever, chat_prompt):
    self.qabot = QABot(llm, retriever, chat_prompt)

  def predict(self, context, inputs):
    questions = list(inputs['question'])

    # 回答の返却
    return [self.qabot.get_answer(q) for q in questions]

# COMMAND ----------

# MAGIC %md 
# MAGIC 次に、以下のようにモデルのインスタンスを作成し、[MLflow registry](https://docs.databricks.com/mlflow/model-registry.html)に記録します：<br>
# MAGIC

# COMMAND ----------

# DBTITLE 1,MLflowにモデルを永続化
# Unity Catalogにモデルを登録する場合
mlflow.set_registry_uri("databricks-uc")
registered_model_name = repository + "." + mlflow_model_name

# mlflowモデルのインスタンスを作成
model = MLflowQABot(llm, retriever, prompt)

# mlflowにモデルを永続化
with mlflow.start_run():
    # Signatureの作成
    input_df = pd.DataFrame({"question" : [""]})
    output_df = pd.DataFrame({'answer':[""], 'source':[""], 'output_metadata':[""]})
    model_signature = infer_signature(input_df, output_df)

    # Model の登録
    logged_model = mlflow.pyfunc.log_model(
      python_model=model,
      extra_pip_requirements=['langchain==0.0.166', 'tiktoken==0.4.0', 'faiss-cpu==1.7.4'],
      # artifacts={'repository' : model_location},
      artifact_path='mpt',
      registered_model_name = f"{registered_model_name}",
      signature=model_signature,
      # Registration can take awhile, so give it 40 minutes to finish just in case. 実際35分ほどかかった。
      await_registration_for=2500
    )

# COMMAND ----------

# MAGIC %md 
# MAGIC MLflowが始めてであれば、ロギングが何の役に立つのかと思うかもしれません。このノートブックに関連づけられているエクスペリメントに移動して、*log_model*の呼び出しによって記録されたものに対する詳細を確認するために、最新のエクスペリメントをクリックすることができます。エクスペリメントにアクセスするにはDatabricks環境の右側のナビゲーションにあるフラスコアイコンをクリックします。モデルのアーティファクトを展開すると、以前インスタンスを作成したMLflowQABotモデルのpickleを表現する*python_model.pkl*を確認することができます。これが(後で)本環境あるいは別環境でモデルをロードする際に取得されるモデルとなります：
# MAGIC </p>
# MAGIC
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/bot_mlflow_log_model.PNG" width=1000>

# COMMAND ----------

# MAGIC %md 
# MAGIC MLflowのモデルレジストリは、CI/CDワークフローを移動する際に登録されたモデルを管理するメカニズムを提供します。モデルを直接UI上でプロダクションのステータスにプッシュ(デモでは構いませんが、現実世界のシナリオでは推奨しません)したいのであれば、以下のようにプログラムから行うことができます：

# COMMAND ----------

# DBTITLE 1,モデルをプロダクションステータスに昇格 (Aliasの付与）
from pyspark.sql.functions import max

# mlflowに接続
client = mlflow.MlflowClient()

# 最新モデルバージョンの特定
latest_version = client.search_model_versions("name = '%s'" % registered_model_name)[0].version
print(latest_version)

# モデルをプロダクションに移行
client.set_registered_model_alias(
  name=registered_model_name,
  alias='Production',
  version=latest_version
)

# COMMAND ----------

# MAGIC %md 
# MAGIC 次に、レスポンスを確認するために、レジストリからモデルを取得し、いくつかの質問を送信することができます：

# COMMAND ----------

# DBTITLE 1,MLflowからModelをロード

# Unity Catalogにモデルを登録した場合
mlflow.set_registry_uri("databricks-uc")

# モデルが保存されている場所を指定
repository = "jmaru_catalog.qabot_openllm_ja"
model_name = "databricks_openllm_qabot_jpn"

# mlflowからモデルを取得
#registered_model_name =f"{config['catalog_name']}.{config['schema_name']}.{config['registered_model_name']}"
registered_model_name = repository + "." + model_name
model = mlflow.pyfunc.load_model(f"models:/{registered_model_name}@Production")

# COMMAND ----------

# DBTITLE 1,モデルの利用
model.predict({'question': "Delta Sharingとは何？"})

# COMMAND ----------

model.predict({'question': "MLflow の特徴を教えて"})

# COMMAND ----------

# MAGIC %md ## Step 4: モデルの評価
# MAGIC
# MAGIC MLflow Evaluation 機能を利用して、サンプルの質問でアウトプット結果を保存・比較することが可能です。
# MAGIC 詳しくは[こちらのBlog](https://www.databricks.com/jp/blog/announcing-mlflow-24-llmops-tools-robust-model-evaluation)をご覧ください。

# COMMAND ----------

# DBTITLE 1,モデルの評価
# 質問入力の構築
questions = pd.DataFrame({'question':[
  "Delta Sharingとは何？",
  "MLflowのメリットは？"
]})

# Model Evaluation 
mlflow.evaluate(
  model=model,
  model_type='question-answering',
  data=questions
)


# COMMAND ----------

# DBTITLE 1,評価結果の出力
# 評価結果をロードして調査
results: pd.DataFrame = mlflow.load_table(
    "eval_results_table.json", extra_columns=["run_id"]
)
results = spark.createDataFrame(results).sort("question")
display(results.select(results.question, results.run_id,  results.outputs.answer, results.outputs.source))


# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | tiktoken | Fast BPE tokeniser for use with OpenAI's models | MIT  |   https://pypi.org/project/tiktoken/ |
# MAGIC | faiss-cpu | Library for efficient similarity search and clustering of dense vectors | MIT  |   https://pypi.org/project/faiss-cpu/ |
# MAGIC | openai | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/openai/ |

# COMMAND ----------


