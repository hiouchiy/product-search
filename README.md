![image](https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo_wide.png)

[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

## LLM製品検索アクセラレーター

このソリューション・アクセラレータの目的は、大規模言語モデル（LLM）とその小さな同胞が、商品検索を可能にするためにどのように使用できるかを示すことです。 今日ほとんどのサイトで使われている、キーワードのマッチに依存する商品検索とは異なり、LLMは一般的にセマンティック検索と呼ばれる、単語の*概念的な類似性*が重要となる検索を可能にします。

単語間の*概念的類似性*に関するモデルの知識は、様々な文書に触れ、それらの文書から、特定の単語が互いに密接な関係を持つ傾向があることを学習することから得られる。 例えば、ある文書が*子供（Children）*にとっての遊びの重要性を論じ、*子供（Child）*という用語を使うことで、*子供（Children）*と*子供（Child）*には何らかの関係があることをモデルに教えるかもしれません。 他の文書では、これらの用語が似たような近さで使われ、同じトピックを論じる他の文書では、*kid*または*kids*という用語が導入されるかもしれません。 文書によっては4つの用語がすべて出てくる可能性もありますが、たとえそのようなことがなかったとしても、これらの用語を囲む単語には十分な重複があり、モデルはこれらすべての用語の間に密接な関連があると認識するようになります。

オープンソースコミュニティから入手可能なLLMの多くは、事前に学習されたモデルとして提供されており、このような単語の関連は、一般に入手可能な幅広い情報からすでに学習されています。これらのモデルがすでに蓄積している知識を使って、商品カタログの商品説明テキストを検索し、ユーザーが入力した検索語やフレーズと一致すると思われる商品を探すことができます。サイト上で紹介される商品が、小売業者や紹介するサプライヤーのトーンやスタイルを反映した独自の関連パターンを持つ、より具体的な用語のセットを使用する傾向がある場合、これらのモデルは、使用されている言語に対する理解を形成するために、サイトに固有の追加データに触れることができます。 この*ファインチューニング*エクササイズは、既製のモデルを特定の製品カタログのニュアンスに合わせるために使用することができ、より効果的な検索結果を可能にします。

このソリューションアクセラレータでは、既製のモデルと特定の商品テキストにチューニングしたモデルの両方のバージョンを紹介します。そして、ユーザーがDatabricks環境を通してセマンティック検索機能をどのようにデプロイできるかを確認できるように、モデルのデプロイメントに関する問題に取り組みます。

___
<tim.lortz@databricks.com> 
<mustafaali.sezer@databricks.com> 
<peyman@databricks.com>
<bryan.smith@databricks.com>
___


<img src='https://github.com/databricks-industry-solutions/product-search/raw/main/images/inference.png' width=800>

___

&copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
|  WANDS | Wayfair product search relevance data | MIT  | https://github.com/wayfair/WANDS   |
| langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
| chromadb | An open source embedding database |  Apache |  https://pypi.org/project/chromadb/  |
| sentence-transformers | Compute dense vector representations for sentences, paragraphs, and images | Apache 2.0 |https://pypi.org/project/sentence-transformers/ |

## Getting started

特定のソリューションはウェブサイトから .dbc アーカイブとしてダウンロードできますが、これらのリポジトリを databricks 環境にクローンすることをお勧めします。最新のコードにアクセスできるだけでなく、業界のベストプラクティスや再利用可能なソリューションを推進する専門家のコミュニティの一員となり、各業界に影響を与えることができます。

<img width="500" alt="add_repo" src="https://user-images.githubusercontent.com/4445837/177207338-65135b10-8ccc-4d17-be21-09416c861a76.png">

Databricksでソリューションアクセラレータを使い始めるには、以下の手順に従ってください： 

1. Databricks Repos](https://www.databricks.com/product/repos) を使って Databricks のソリューションアクセラレータリポジトリをクローンする。
2. RUNME` ノートブックを任意のクラスタにアタッチし、Run-All を使ってノートブックを実行する。アクセラレータパイプラインを記述したマルチステップジョブが作成され、リンクが提供される。ジョブの設定はRUNMEノートブックにjson形式で記述される。
3. マルチステップジョブを実行して、パイプラインがどのように実行されるかを確認します。
4. ソリューションアクセラレータのサンプルを自分のニーズに合わせて変更したり、他のユーザーと共同作業したり、自分のデータに対してコードサンプルを実行したりしたくなるかもしれません。そのためには、まず自分のリポジトリのGitリモートを自分の組織のリポジトリに変更することから始めましょう。これで、コードをコミットしてプッシュし、Gitを介して他のユーザーと共同作業し、コード開発のための組織のプロセスに従うことができます。

アクセラレーターの実行に関連する費用は、ユーザーの負担となります。


## サポートについて 

このプロジェクトに含まれるコードは、あくまで参考として提供されるものであり、Databricksによる正式なサービスレベルアグリーメント(SLA)でのサポートはありません。また、Databricksはいかなる保証も行いません。これらのプロジェクトの使用から生じるいかなる問題に関しても、サポートチケットを提出しないでください。このプロジェクトのソースは Databricks [License](./LICENSE) に従って提供されます。含まれる、または参照されるすべてのサードパーティライブラリは、以下に定めるライセンスに従うものとします。

このプロジェクトを使用して発見された問題は、GitHub IssuesとしてRepoに提出してください。時間の許す限りレビューされますが、サポートに関する正式なSLAはありません。
