import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy.stats import ks_2samp

# Configuração da página do Streamlit
st.set_page_config(page_title="Simulador de Crédito - Analytics", layout="wide", page_icon="💳")

st.title("💳 Plataforma Inteligente de Concessão & Simulação de Crédito")
st.markdown("---")

# Função para gerar dados fictícios para teste rápido
def gerar_dados_ficticios(n_samples=1000):
    np.random.seed(42)
    dados = {
        'idade': np.random.randint(18, 70, n_samples),
        'situacao_profissional': np.random.choice(['CLT', 'Autônomo', 'Empresário', 'Desempregado'], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
        'anos_emprego': np.random.randint(0, 30, n_samples),
        'renda_anual': np.random.normal(60000, 25000, n_samples).clip(12000, 300000),
        'score_credito': np.random.randint(300, 850, n_samples),
        'historico_credito': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'poupanca_ativos': np.random.normal(15000, 20000, n_samples).clip(0, 200000),
        'divida_atual': np.random.normal(20000, 15000, n_samples).clip(0, 150000),
        'inad_registradas': np.random.choice([0, 1, 2], n_samples, p=[0.85, 0.11, 0.04]),
        'atrasos_2_anos': np.random.choice([0, 1, 2, 3], n_samples, p=[0.7, 0.15, 0.1, 0.05]),
        'restricoes': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'tipo_produto': np.random.choice(['Cartão de Crédito', 'Empréstimo Pessoal', 'Financiamento Imobiliário', 'Consignado'], n_samples),
        'intencao_emprestimo': np.random.choice(['Pessoal', 'Educação', 'Consolidação de Dívida', 'Empreendimento'], n_samples),
        'valor_emprestimo': np.random.normal(15000, 10000, n_samples).clip(1000, 100000),
        'taxa_juros': np.random.uniform(5.0, 28.0, n_samples),
    }
    df = pd.DataFrame(dados)
    
    # Criando métricas derivadas
    df['divida_renda'] = df['divida_atual'] / df['renda_anual']
    df['emprestimo_renda'] = df['valor_emprestimo'] / df['renda_anual']
    df['pagamento_renda'] = (df['valor_emprestimo'] * (1 + df['taxa_juros']/100) / 12) / (df['renda_anual'] / 12)
    
    # Lógica para criar a variável alvo simulando um cenário real de risco
    score_efeito = (df['score_credito'] - 300) / 550
    renda_efeito = (df['renda_anual'] - 12000) / 288000
    divida_efeito = 1 - df['divida_renda'].clip(0, 1)
    restricao_efeito = 1 - df['restricoes']
    
    prob = (score_efeito * 0.4 + renda_efeito * 0.2 + divida_efeito * 0.2 + restricao_efeito * 0.2)
    df['emprestimo_aprovado'] = (prob > 0.45).astype(int)
    
    return df

# Inicialização de variáveis globais na sessão
if 'modelo_treinado' not in st.session_state:
    st.session_state.modelo_treinado = None
    st.session_state.scaler = None
    st.session_state.features_colunas = None
    st.session_state.top5_features = None
    st.session_state.df_original = None

# Sidebar para upload de dados
st.sidebar.header("📥 Carga de Dados")
upload_file = st.sidebar.file_uploader("Envie sua base de dados (CSV)", type=["csv"])

usar_ficticios = st.sidebar.button("✨ Usar Base Fictícia de Teste")

if upload_file is not None:
    st.session_state.df_original = pd.read_csv(upload_file)
    st.sidebar.success("Base carregada com sucesso!")
elif usar_ficticios:
    st.session_state.df_original = gerar_dados_ficticios()
    st.sidebar.success("Base fictícia gerada com sucesso!")

# Verificação se os dados existem
if st.session_state.df_original is not None:
    df = st.session_state.df_original.copy()
    
    # Menu de navegação principal por Tabs
    tab_dados, tab_treino, tab_interpreta, tab_simulador = st.tabs([
        "📊 Visão Geral dos Dados", 
        "⚙️ Treinamento e Otimização", 
        "🧠 Interpretabilidade", 
        "🔮 Simulador Interativo"
    ])
    
    with tab_dados:
        st.subheader("Amostra dos Dados Carregados")
        st.dataframe(df.head(10))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total de Registros", df.shape[0])
            st.metric("Total de Variáveis", df.shape[1])
        with col2:
            aprovados = df['emprestimo_aprovado'].value_counts(normalize=True)
            st.metric("Taxa de Aprovação Atual", f"{aprovados.get(1, 0)*100:.2f}%")
            st.metric("Taxa de Rejeição Atual", f"{aprovados.get(0, 0)*100:.2f}%")

    with tab_treino:
        st.subheader("Processamento do Pipeline de Machine Learning")
        
        if st.button("🚀 Executar Pipeline Completo (Limpeza, Treinamento e Tuning)"):
            with st.spinner("Processando..."):
                # 1. Limpeza de Dados
                df = df.drop_duplicates()
                num_cols = df.select_dtypes(include=['float64', 'int64']).columns
                cat_cols = df.select_dtypes(include=['object']).columns
                
                df[num_cols] = df[num_cols].fillna(df[num_cols].median())
                df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
                
                # Remoção de outliers (IQR) em colunas críticas
                cols_outliers = ['renda_anual', 'valor_emprestimo', 'divida_renda']
                for col in cols_outliers:
                    if col in df.columns:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]
                
                # 2. Engenharia de Features (One-Hot Encoding)
                cols_to_encode = [c for c in ['situacao_profissional', 'tipo_produto', 'intencao_emprestimo'] if c in df.columns]
                df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
                
                # Separação das Features e Target
                X = df_encoded.drop('emprestimo_aprovado', axis=1)
                y = df_encoded['emprestimo_aprovado']
                
                # Garantindo que todas as colunas sejam numéricas (booleanas para int)
                X = X.astype(float)
                
                # Divisão Treino e Teste
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
                
                # Escalonamento
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Treinamento dos 3 Modelos Básicos
                modelos = {
                    'Regressão Logística': LogisticRegression(max_iter=1000, random_state=42),
                    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
                }
                
                resultados = {}
                col_m1, col_m2, col_m3 = st.columns(3)
                cols_layout = [col_m1, col_m2, col_m3]
                
                for idx, (nome, modelo) in enumerate(modelos.items()):
                    modelo.fit(X_train_scaled, y_train)
                    y_pred = modelo.predict(X_test_scaled)
                    y_proba = modelo.predict_proba(X_test_scaled)[:, 1]
                    
                    # Métricas
                    recall = recall_score(y_true=y_test, y_pred=y_pred)
                    f1 = f1_score(y_test, y_pred)
                    auc_roc = roc_auc_score(y_test, y_proba)
                    
                    proba_0 = y_proba[y_test == 0]
                    proba_1 = y_proba[y_test == 1]
                    ks_stat, _ = ks_2samp(proba_0, proba_1)
                    
                    resultados[nome] = {'AUC': auc_roc, 'KS': ks_stat, 'Recall': recall, 'F1': f1, 'modelo': modelo}
                    
                    with cols_layout[idx]:
                        st.markdown(f"### {nome}")
                        st.write(f"**AUC-ROC:** {auc_roc:.4f}")
                        st.write(f"**KS:** {ks_stat:.4f}")
                        st.write(f"**Recall:** {recall:.4f}")
                        st.write(f"**F1-Score:** {f1:.4f}")
                        
                        # Matriz de Confusão simplificada
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots(figsize=(3, 2))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
                        ax.set_ylabel('Real')
                        ax.set_xlabel('Previsto')
                        st.pyplot(fig)
                
                # Escolha do melhor modelo baseado no AUC-ROC
                melhor_nome = max(resultados, key=lambda k: resultados[k]['AUC'])
                st.success(f"🏆 Melhor modelo detectado: **{melhor_nome}**")
                
                # Otimização de Hiperparâmetros (GridSearch Simplificado para Performance)
                st.markdown("### 🛠️ Otimizando o Melhor Modelo via GridSearchCV...")
                if melhor_nome == 'Gradient Boosting':
                    param_grid = {'n_estimators': [100, 150], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
                    base_model = GradientBoostingClassifier(random_state=42)
                elif melhor_nome == 'Random Forest':
                    param_grid = {'n_estimators': [100, 150], 'max_depth': [5, 10]}
                    base_model = RandomForestClassifier(random_state=42)
                else:
                    param_grid = {'C': [0.1, 1.0, 10.0]}
                    base_model = LogisticRegression(max_iter=1000, random_state=42)
                
                grid = GridSearchCV(base_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
                grid.fit(X_train_scaled, y_train)
                
                best_model = grid.best_estimator_
                y_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]
                auc_otimizado = roc_auc_score(y_test, y_proba_best)
                
                st.write(f"**Melhores parâmetros:** {grid.best_params_}")
                st.write(f"**AUC-ROC Antes:** {resultados[melhor_nome]['AUC']:.4f} ➡️ **AUC-ROC Otimizado:** {auc_otimizado:.4f}")
                
                # Importância das Features
                if hasattr(best_model, 'feature_importances_'):
                    importances = best_model.feature_importances_
                else:
                    importances = np.abs(best_model.coef_[0])
                    
                feat_df = pd.DataFrame({'Feature': X.columns, 'Importância': importances}).sort_values(by='Importância', ascending=False)
                
                # Salvando na sessão para as próximas abas
                st.session_state.modelo_treinado = best_model
                st.session_state.scaler = scaler
                st.session_state.features_colunas = X.columns.tolist()
                st.session_state.top5_features = feat_df.head(5)['Feature'].tolist()
                st.session_state.feat_df = feat_df
                
                st.balloons()
                st.success("Modelo pronto para uso nas abas de 'Interpretabilidade' e 'Simulador'!")

    with tab_interpreta:
        if st.session_state.modelo_treinado is None:
            st.warning("Por favor, execute o pipeline de treinamento na aba anterior primeiro.")
        else:
            st.subheader("🧠 Entendendo as Decisões do Modelo (XAI)")
            
            fig_feat, ax_feat = plt.subplots(figsize=(10, 5))
            sns.barplot(x='Importância', y='Feature', data=st.session_state.feat_df.head(10), palette='viridis', ax=ax_feat)
            ax_feat.set_title("Top 10 Variáveis de Maior Impacto")
            st.pyplot(fig_feat)
            
            st.markdown("### Significado de Negócio das Principais Variáveis:")
            for feature in st.session_state.top5_features:
                st.markdown(f"- **`{feature}`**: Esta variável apresentou alto peso na separação de risco. Mudanças drásticas neste valor influenciam diretamente a probabilidade de aprovação do cliente.")

    with tab_simulador:
        if st.session_state.modelo_treinado is None:
            st.warning("Por favor, execute o pipeline de treinamento na aba anterior primeiro.")
        else:
            st.subheader("🔮 Simulador de Score de Crédito em Tempo Real")
            st.markdown("Ajuste os valores abaixo para calcular a probabilidade de concessão de crédito do cliente:")
            
            # Criando inputs dinâmicos na interface baseados nas variáveis do modelo original
            inputs_usuario = {}
            
            # Vamos gerar sliders/inputs estruturados para facilitar a demonstração
            col_sim1, col_sim2 = st.columns(2)
            
            with col_sim1:
                if 'score_credito' in st.session_state.features_colunas:
                    inputs_usuario['score_credito'] = st.slider("Score de Crédito", 300, 850, 600)
                if 'renda_anual' in st.session_state.features_colunas:
                    inputs_usuario['renda_anual'] = st.number_input("Renda Anual ($)", min_value=0, value=50000)
                if 'valor_emprestimo' in st.session_state.features_colunas:
                    inputs_usuario['valor_emprestimo'] = st.number_input("Valor do Empréstimo ($)", min_value=0, value=15000)
                if 'idade' in st.session_state.features_colunas:
                    inputs_usuario['idade'] = st.slider("Idade", 18, 100, 35)
                if 'anos_emprego' in st.session_state.features_colunas:
                    inputs_usuario['anos_emprego'] = st.slider("Anos no Emprego Atual", 0, 45, 5)
                    
            with col_sim2:
                if 'divida_renda' in st.session_state.features_colunas:
                    inputs_usuario['divida_renda'] = st.slider("Comprometimento de Renda (Dívida/Renda)", 0.0, 2.0, 0.3, step=0.05)
                if 'emprestimo_renda' in st.session_state.features_colunas:
                    inputs_usuario['emprestimo_renda'] = st.slider("Razão Empréstimo por Renda", 0.0, 2.0, 0.2, step=0.05)
                if 'pagamento_renda' in st.session_state.features_colunas:
                    inputs_usuario['pagamento_renda'] = st.slider("Razão Parcela por Renda", 0.0, 1.0, 0.1, step=0.01)
                if 'taxa_juros' in st.session_state.features_colunas:
                    inputs_usuario['taxa_juros'] = st.slider("Taxa de Juros (%)", 1.0, 40.0, 12.0)
                if 'restricoes' in st.session_state.features_colunas:
                    inputs_usuario['restricoes'] = st.selectbox("Possui Restrições Cadastrais?", [0, 1], format_func=lambda x: "Sim" if x==1 else "Não")
                if 'historico_credito' in st.session_state.features_colunas:
                    inputs_usuario['historico_credito'] = st.selectbox("Histórico de Crédito Limpo?", [1, 0], format_func=lambda x: "Sim" if x==1 else "Não")
                if 'inad_registradas' in st.session_state.features_colunas:
                    inputs_usuario['inad_registradas'] = st.selectbox("Inadimplências Registradas", [0, 1, 2, 3])
                if 'atrasos_2_anos' in st.session_state.features_colunas:
                    inputs_usuario['atrasos_2_anos'] = st.selectbox("Atrasos nos Últimos 2 Anos", [0, 1, 2, 3, 4])
                if 'poupanca_ativos' in st.session_state.features_colunas:
                    inputs_usuario['poupanca_ativos'] = st.number_input("Poupança/Ativos ($)", min_value=0, value=5000)
                if 'divida_atual' in st.session_state.features_colunas:
                    inputs_usuario['divida_atual'] = st.number_input("Dívida Atual ($)", min_value=0, value=10000)

            # Para as variáveis dummy que foram criadas no One-Hot-Encoding, precisamos garantir que estejam presentes
            # Mapeamos o estado padrão como 0 para todas as outras colunas que não estão explícitas nos inputs acima
            for col in st.session_state.features_colunas:
                if col not in inputs_usuario:
                    inputs_usuario[col] = 0.0
                    
            # Criando o DataFrame correspondente a uma linha para predição
            df_pred = pd.DataFrame([inputs_usuario])[st.session_state.features_colunas]
            
            # Escalonando os dados de entrada
            df_pred_scaled = st.session_state.scaler.transform(df_pred)
            
            # Predição de Probabilidade
            probabilidade = st.session_state.modelo_treinado.predict_proba(df_pred_scaled)[0][1]
            
            # Exibição do Resultado de Forma Visual e Clara para Stakeholders
            st.markdown("---")
            st.subheader("📊 Resultado da Análise de Risco")
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.metric(label="Probabilidade de Aprovação", value=f"{probabilidade*100:.1f}%")
                
            with col_res2:
                if probabilidade >= 0.70:
                    st.success("🟢 STATUS: CRÉDITO APROVADO (Baixo Risco)")
                    st.info("Recomendação: Seguir com a emissão do contrato nas taxas vigentes de mercado.")
                elif probabilidade >= 0.50:
                    st.warning("🟡 STATUS: ENVIAR PARA ANÁLISE MANUAL (Médio Risco)")
                    st.info("Recomendação: Solicitar comprovantes adicionais ou aplicar uma garantia real/coobrigado.")
                else:
                    st.error("🔴 STATUS: CRÉDITO REJEITADO (Alto Risco)")
                    st.info("Recomendação: Recusa automática para evitar inadimplência na carteira.")
else:
    st.info("💡 Para começar, faça o upload de um arquivo CSV ou clique no botão 'Usar Base Fictícia de Teste' na barra lateral.")
