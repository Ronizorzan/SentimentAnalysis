import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from joblib import load
import nltk
nltk.download("vader_lexicon")




#Configuração do Layout da página
st.set_page_config(page_title="Análise de Sentimentos", layout="wide")



#Função de carregamento do modelo e dados
@st.cache_resource
def load_and_process_data():
    data = pd.read_csv("Twitter_Data.csv")
    
    #Tratamento dos dados
    data.dropna(inplace=True)

           
    #separação da classe
    X = data["clean_text"]
    y = data["category"]

    
    #Divisão entre treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=321)

    vectorizer = TfidfVectorizer().fit(X_train)

    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)

    #Carregamento do modelo treinado
    model = load("logistic_model.pkl")    
    predicted = model.predict(X_test)

    #Acurácia
    accuracy = accuracy_score(y_test, predicted)

    #Matriz de Confusão para visualização
    cm = confusion_matrix(y_test, predicted)
    confusion = ConfusionMatrixDisplay(cm, display_labels=["Negativos", "Neutros", "Positivos"]).plot(xticks_rotation=30, cmap="viridis")

    #Retorno
    return confusion, accuracy, data, model, vectorizer


#Função para cálculo da polaridade e subjetividade
@st.cache_resource
def subjectivity_and_polarity(data):
    analisador = SentimentIntensityAnalyzer()
    polarities = []
    for text in data["clean_text"]:
        polarity = analisador.polarity_scores(text)        
        polarities.append(polarity["compound"])
        
    return polarities

#Função para geração das nuvens de palavras
@st.cache_resource
def gerador_nuvem(data):    
    positive_texts = data.loc[data["category"]==  1, "clean_text"]
    neutral_texts = data.loc[data["category"]==   0, "clean_text"]
    negative_texts = data.loc[data["category"]== -1, "clean_text"]

    positive_wordcloud = WordCloud(width=400, height=300, background_color="black",colormap="viridis", max_words=100 ).generate(str(positive_texts))
    neutral_wordcloud = WordCloud(width=400, height=300, background_color="black",colormap="viridis", max_words=100).generate(str(neutral_texts))
    negative_wordcloud = WordCloud(width=400, height=300, background_color="black",colormap="viridis", max_words=100).generate(str(negative_texts))

    positive_fig, ax2 = plt.subplots() #Nuvem de palavras positivas
    plt.imshow(positive_wordcloud, interpolation="bilinear")
    plt.title("Nuvem de Palavras(Tweets Positivos)", fontsize=15, fontweight="bold", color="green")
    plt.grid(False)
    plt.axis("off")    
    
    
    negative_fig, ax = plt.subplots() #Nuvem de palavras negativas
    plt.imshow(negative_wordcloud, interpolation="bilinear")
    plt.title("Nuvem de Palavras(Tweets Negativos)", fontsize=15, fontweight="bold", color="red")
    plt.grid(False)
    plt.axis("off")
    

    neutral_fig, ax3 = plt.subplots()  #Nuvem de palavras neutras
    plt.imshow(neutral_wordcloud, interpolation="bilinear")
    plt.title("Nuvem de Palavras(Tweets Neutros)", fontsize=15, fontweight="bold", color="darkorange")
    plt.axis("off")
    plt.grid(False)
    

    return positive_fig, negative_fig, neutral_fig
    



#Configuração da barra lateral
with st.sidebar:        
    st.markdown("<hr style='border:2px solid white'>", unsafe_allow_html=True)
    st.markdown("<h1 style='color: gray;'>Analisar novas entradas </h1>", unsafe_allow_html=True)
    texto = st.text_area("Insira o texto para analisar", value="")
    processar = st.button("Processar")


if processar and texto:            
    progress = st.progress(20, text="Carregando o Modelo... Por favor Aguarde")    
    confusion, accuracy, data, model, vectorizer = load_and_process_data()
    progress.progress(60, text="Calculando Polaridade e subjetividade")    
    polarities = subjectivity_and_polarity(data)
    progress.progress(100, text="Gerando Nuvens de Palavras")
    positive_fig, negative_fig, neutral_fig = gerador_nuvem(data)    
    

#exibições dos resultados na barra lateral
    with st.sidebar:
        mapeamento = {-1.0: "Negativo", 0.0: "Neutro", 1.0: "Positivo"}            
        texto = vectorizer.transform([texto])
        previsao = model.predict(texto)            
        previsao = list(map(mapeamento.get, previsao))            
        st.markdown("<hr style='border:2px solid white'>", unsafe_allow_html=True)
        st.markdown(f"*Acurácia do Modelo:*  **{ accuracy*100:.2f}%**")
        st.markdown("*O Modelo obteve um ótimo desempenho acertando aproximadamente 9 de 10 previsões*")
        if previsao[0]=="Positivo":
            st.success(f"**Resultado da Análise:** *{ previsao[0]}*")
        elif previsao[0]=="Neutro":
            st.warning(f"**Resultado da Análise:**  *{ previsao[0]}*")
        else:
            st.error(f"*Resultado da Análise:*  **{ previsao[0]}**")
        

    #Tabulações
    tab1, tab2 = st.tabs(["Métricas e Nuvem de Palavras", "Polaridade e Subjetividade"])
    
    with tab1:    
        col1, col2 = st.columns(2, gap="large")  
        with col1:                        
            #Exibição da Matriz de Confusão na primeira coluna
            st.markdown("<h2 style='color: gray;'>Matriz de Confusão</h2>", unsafe_allow_html=True)                                    
            plt.grid(False)
            st.pyplot(confusion.figure_, use_container_width=True)
            st.markdown("<hr style='border:2px solid white'>", unsafe_allow_html=True)
            st.markdown("**A matriz acima exibe na diagonal principal os acertos do modelo...Indicando que o modelo \
                             generalizou bem para todas as classes conseguindo identificar corretamente a grande maioria dos Tweets (positivos, negativos e neutros)**")

        with col2:            
            #Exibição da nuvem de palavras de acordo com a previsão obtida
            wordcloud = negative_fig if previsao[0]=="Negativo" else neutral_fig if previsao[0]=="Neutro" else positive_fig
            st.pyplot(wordcloud)
            st.markdown("<hr style='border:2px solid white'>", unsafe_allow_html=True)
            st.markdown("**A Nuvem de palavras acima exibe as palavras mais frequentes nos textos de acordo com a classe prevista. \
                        Use para identificar os sentimentos e possíveis causas para insatisfação dos eleitores...**")
        
            
    with tab2:
        col1, col2 = st.columns(2, gap="large")
        with col1:
            #Exibição da polaridade
            fig, ax = plt.subplots()
            plt.style.use("seaborn-v0_8-muted")

            mean_polarity = sum(polarities) / len(polarities)

            plt.hist(x=polarities, color="SkyBlue", edgecolor="black", bins=40, alpha=0.7)
            plt.title("Histograma de Polaridade", fontsize=15, fontweight="bold", color="darkblue")
            plt.xlabel("Distribuição de Polaridade", fontsize=10, fontweight="bold", color="darkblue")
            plt.ylabel("Contagem de Polaridade", fontsize=10, fontweight="bold", color="darkblue")
            plt.axvline(mean_polarity, color="darkblue", linewidth=0.8, linestyle="-.", label=f"Polaridade Média: {mean_polarity:.2f}")
            plt.legend(fontsize=10)
            plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.8)            
            st.pyplot(fig)
            st.markdown("*O gráfico acima refere-se à Polaridade que é a atitude expressa nos textos, indicando se o conteúdo é positivo, negativo ou neutro. Ela é uma medida de sentimento. \
                        A polaridade varia de -1 a 1. Uma pontuação de -1 indica um sentimento extremamente negativo, 0 indica neutralidade e 1 representa um sentimento extremamente positivo.*")

        with col2:
            #Exibição da subjetividade
            subjectivity = [abs(p) for p in polarities]            
            mean_subjectivity = sum(subjectivity) / len(subjectivity)

            fig2, ax2 = plt.subplots()
            plt.style.use("seaborn-v0_8-muted")
            plt.hist(x=subjectivity, color="salmon", edgecolor="black", bins=40, alpha=0.7)
            plt.title("Histograma de Subjetividade", fontsize=15, fontweight="bold", color="darkred")
            plt.xlabel("Distribuição de Subjetividade", fontsize=10, fontweight="bold", color="darkred")
            plt.ylabel("Contagem de Subjetividade", fontsize=10, fontweight="bold", color="darkred")
            plt.axvline(mean_subjectivity, color="darkred", linewidth=0.8, linestyle="-.", label=f"Subjetividade Média: {mean_subjectivity:.2f}")
            plt.legend(fontsize=10)
            plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.8)
            st.pyplot(fig2)
            st.markdown("*O gráfico acima refere-se à subjetividade que indica o quão opinativo ou baseado em fatos é um texto. Textos com alta subjetividade contêm mais opiniões e emoções, \
                         enquanto textos com baixa subjetividade são mais objetivos e baseados em fatos. A subjetividade varia de 0 a 1. Uma pontuação de 0 indica \
                        que o texto é completamente objetivo e baseado em fatos, enquanto 1 indica que o texto é totalmente subjetivo baseando-se apenas em opiniões pessoais.*")
       
            

        



    


    

    



