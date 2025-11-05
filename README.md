# Classificador de Chamados de Clientes (NLP)

Este projeto foi desenvolvido como trabalho final da disciplina de Processamento de Linguagem Natural (NLP) do MBA em Data Science & AI.

O objetivo é construir um *pipeline* de Machine Learning para classificar automaticamente chamados de clientes (textos) em 5 categorias de atendimento, otimizando o processo de triagem.

## 🚀 Metodologia e Pipeline

O projeto seguiu um pipeline completo de NLP:

1.  **Preparação dos Dados:** O dataset (`chamados.csv`) foi carregado e limpo (remoção de nulos).
2.  **Pré-processamento de Texto:** Foi criada uma função de limpeza usando **NLTK** e **Regex** para converter o texto para minúsculas, remover pontuações e *stopwords* em português.
3.  **Divisão:** O dataset foi dividido em 75% para treino e 25% para teste (`random_state=42`, `stratify=y`).
4.  **Engenharia de Features (Vetorização):** Foram testadas e comparadas duas abordagens principais:
    * **TF-IDF:** Usando `TfidfVectorizer` (com unigramas + bigramas).
    * **Word Embeddings:** Usando `Word2Vec` (Gensim) com a média dos vetores de palavras.
5.  **Modelagem e Avaliação:** Múltiplos classificadores supervisionados foram treinados e avaliados pela sua Acurácia e F1-Score.

## 📊 Resultados e Comparação

O modelo de **Regressão Logística** combinado com a vetorização **TF-IDF (Uni+Bigramas)** foi o grande vencedor, superando significativamente os outros *baselines*.

| Modelo | Vetorização | Acurácia | F1-Score (Ponderado) |
| :--- | :--- | :--- | :--- |
| **Regressão Logística (Vencedor)** | **TF-IDF (Uni+Bigramas)** | **90.6%** | **0.91** |
| RandomForest | Word2Vec (Média) | 82.0% | 0.81 |
| Multinomial Naive Bayes (Baseline) | TF-IDF (Uni+Bigramas) | 74.5% | 0.71 |

O Naive Bayes se mostrou ineficaz (F1 de 0.09) para a classe "Outros", enquanto a Regressão Logística manteve alta performance em todas as categorias.

## 🛠️ Tecnologias Utilizadas

* **Python**
* **Pandas:** Para manipulação de dados.
* **Scikit-learn:** Para `train_test_split`, `TfidfVectorizer`, `LogisticRegression`, `MultinomialNB` e métricas de avaliação.
* **NLTK:** Para pré-processamento e remoção de *stopwords*.
* **Gensim:** Para o treinamento do modelo `Word2Vec`.
* **Matplotlib / Seaborn:** Para visualização de dados.

## 🏁 Como Executar

1.  Clone este repositório.
2.  Instale as dependências (ex: `pip install pandas scikit-learn nltk gensim matplotlib seaborn`).
3.  Abra e execute o notebook Jupyter `Template_Trabalho_Final_NLP.ipynb` em um ambiente como o Google Colab.
