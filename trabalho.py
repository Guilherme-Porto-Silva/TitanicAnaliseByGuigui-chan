# Instalando as ferramentas

import pandas

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



# Limpando os base_usada

Titanic = pandas.read_csv("Titanic-Dataset.csv")

Titanic = Titanic.fillna(Titanic.mean(numeric_only=True))

base_usada = Titanic[['Pclass', 'Age', 'Fare']]

previsto = Titanic['Survived'].astype(int)



# Interpretando os base_usada

# Para este exemplo, usaremos 'Pclass', 'Age' e 'Fare' para prever 'Survived'

# 'Survived' (Sobrevivente) é o "Churn" (variável a ser prevista: 0 ou 1).

trainamento_primeiro_teste, primeiro_teste, trainamento_segundo_teste, segundo_teste = train_test_split(base_usada, previsto, test_size=0.2, random_state=42)

logica_de_previsao = LogisticRegression()

logica_de_previsao.fit(base_usada, previsto)

previsoes_probabilidade = logica_de_previsao.predict_proba(primeiro_teste)

taxa_de_vazamento = previsoes_probabilidade[:, 1]



# Mostrando a previsão

resultados = pandas.DataFrame({
    
    'Taxa_de_vazamento': taxa_de_vazamento,
    
    'Vazamento_real': segundo_teste  # Para comparação
})

resultados_alto_risco = resultados.sort_values(by='Taxa_de_vazamento', ascending=False).head(10)

print("\n--- TOP 10 CLIENTES NA ZONA DE PERIGO ---")

print(resultados_alto_risco)



# Fazendo a previsão da CLASSE (0 ou 1)

previsoes_classe = logica_de_previsao.predict(primeiro_teste)

# Calculando a Acurácia (acertos/total de previsões)

acuracia = accuracy_score(segundo_teste, previsoes_classe)

# Imprimindo a Acurácia

print(f"\n--- MÉTRICA DE DESEMPENHO DO MODELO ---")

print(f"A acurácia do nosso Modelo de Churn é de: {acuracia * 100}%\n\n")



# Enviando os resultados para o Power BI

resultados = pandas.DataFrame({
    
    'Taxa_de_vazamento': taxa_de_vazamento,
    
    'Vazamento_real': segundo_teste,
    
    'Classe': previsoes_classe,
    
    'Acurácia': segundo_teste
})

resultados.to_excel("saida_modelo.xlsm", index=False)