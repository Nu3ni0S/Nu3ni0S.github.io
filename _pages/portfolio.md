---
layout: single
title: "Portfolio"
permalink: /portfolio/

cs_1:
  - image_path: /assets/images/projects/aws_eks_ms.jpg
    alt: "aws_ek_ms"
    title: "AWS EKS Microservices"
    text: "This project implements a Python-based Microservice Application on AWS Elastic Kubernetes Services (EKS) where a user can upload a video and get it processed into a sound file (mp3)."
    url: "https://github.com/maxlr8/aws_eks_microservices"
    btn_label: "Github"
    btn_class: "btn--primary"
    tags:
        - AWS
        - EKS
        - ELB
        - EC2
        - PostgresDB
        - MongoDB
        - Flask
        - API Gateway
        - Docker

cs_2:
  - image_path: /assets/images/projects/aws_eks_cn.jpg
    alt: "aws_ek_cn"
    title: "Cloud-Native App using AWS EKS"
    text: "This project implements an End-To-End Cloud-Native Voting application using AWS EKS (Amazon Elastic Kubernetes Services) where users can caste their votes."
    url: "https://github.com/maxlr8/aws_eks_cloud_native_app/"
    btn_label: "Github"
    btn_class: "btn--primary"
    tags:
        - AWS
        - EKS
        - ELB
        - EC2
        - MongoDB
        - GoLang
        - React

cs_3:
  - image_path: /assets/images/projects/kafka.jpg
    alt: "kafka"
    title: "Kafka Real-time Stock-Market Analysis"
    text: "This project implements an End-To-End Data Engineering Project on Real-Time Stock Market Data using Kafka and the dynamically generated data is stored in the S3 bucket which can is crawled using Glue, fetched using Athena and visualized using QuickSight."
    url: "https://github.com/maxlr8/stock_market_analysis"
    btn_label: "Github"
    btn_class: "btn--primary"
    tags:
        - Apache Kafka
        - Apache ZooKeeper
        - AWS
        - EC2
        - Boto3
        - S3
        - Glue 
        - Athena
        - QuickSight

cs_4:
  - image_path: /assets/images/projects/twitter_etl.jpg
    alt: "twitter_etl"
    title: "ETL using Apache Airflow"
    text: "This project focuses on extracting data from Twitter using the Twitter API, transforming the data using Python and loading it into S3-bucket."
    url: "https://github.com/maxlr8/aws_twitter_etl/"
    btn_label: "Github"
    btn_class: "btn--primary"
    tags:
        - Apache Airflow
        - AWS
        - EC2
        - S3

cs_5:
  - image_path: /assets/images/projects/psql.jpg
    alt: "psql"
    title: "Data Modeling with Postgres"
    text: "In this project, data modeling is performed with Postgres and an ETL (Extract, Transform, Load) pipeline is employed using Python."
    url: "https://github.com/maxlr8/data_modeling_with_postgres/"
    btn_label: "Github"
    btn_class: "btn--primary"
    tags:
        - Postgres
        - Python
        - Shell

aiml_1:
  - image_path: /assets/images/projects/langchain_dh.jpg
    alt: "langchain_dh"
    title: "LangChain Documentation Helper"
    text: "This project implements a LangChain based AI Web Application that is trained and deployed to answer ay question about LangChain (Sources from Official LangChain documentation)."
    url: "https://github.com/maxlr8/aws_eks_microservices"
    btn_label: "Github"
    btn_class: "btn--primary"
    tags:
        - OpenAI
        - LangChain
        - Pinecone
---

## Data Engineering Projects 

{% include feature_row id="cs_1" type="left" %}
<a name="AWS EKS Microservices"></a>
{% include feature_row id="cs_2" type="left" %}
<a name="Cloud-Native Voting App using AWS EKS"></a>
{% include feature_row id="cs_3" type="left" %}
<a name="Real-time Stock-Market Analysis using Kafka"></a>
{% include feature_row id="cs_4" type="left" %}
<a name="ETL using Apache Airflow"></a>
{% include feature_row id="cs_5" type="left" %}
<a name="Data Modeling with Postgres"></a>


## AI and ML Projects

&nbsp;
{% include feature_row id="aiml_1" type="left" %}
<a name="LangChain Documentation Helper">
