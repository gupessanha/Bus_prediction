import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine
from shapely.geometry import Point, LineString, MultiPoint
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
from itertools import combinations
import numpy as np
import json
import datetime
import traceback
import os
import joblib
import multiprocessing
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed

# =====================================================================================
# ||                           CONFIGURAÇÕES GLOBAIS                                 ||
# =====================================================================================

# Substitua com suas credenciais e configurações
DB_USER = 'myuser'
DB_PASSWORD = 'mypassword'
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'bus_predictions'
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

DATA_INICIO_TREINO = '2024-04-25'
DATA_FIM_TREINO = '2024-05-10'

NOME_ALUNO = "xxxxx xxxxx"
SENHA_SECRETA = "xxxx"

# Número de workers para paralelização
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)

# =====================================================================================
# ||                  FUNÇÕES OTIMIZADAS DE TREINAMENTO E PREVISÃO                   ||
# =====================================================================================

def processar_linha(linha_atual, gdf_linha_original, pasta_modelos, modo_treino):
    """
    Função auxiliar para processar uma linha específica (para paralelização)
    """
    try:
        print(f"\n--- Processando e treinando para a linha {linha_atual} ---")
        gdf_linha = gdf_linha_original.copy()
        
        # 3. CONSTRUÇÃO DO TRAJETO CANÔNICO
        projected_crs = "EPSG:32723" 
        terminal_A, terminal_B = None, None
        trajetos_finais = {}
        
        # Reduzir uso de memória selecionando apenas os pontos parados
        paradas = gdf_linha[gdf_linha['velocidade'] == 0].copy()
        paradas.loc[:, 'tempo_diff_min'] = paradas.groupby('ordem')['datahoraservidor'].diff().dt.total_seconds() / 60
        paradas_terminais = paradas[paradas['tempo_diff_min'].between(10, 60)].copy()
        
        # Liberar memória
        del paradas
        gc.collect()

        if len(paradas_terminais) >= 2:
            # Usar apenas as coordenadas para clustering para economizar memória
            coords = paradas_terminais[['latitude', 'longitude']].values
            clustering = DBSCAN(eps=0.001, min_samples=2).fit(coords)
            paradas_terminais.loc[:, 'terminal_cluster'] = clustering.labels_
            
            terminais_encontrados = {lbl: MultiPoint(paradas_terminais[paradas_terminais['terminal_cluster'] == lbl].geometry.tolist()).centroid
                                     for lbl in paradas_terminais['terminal_cluster'].unique() if lbl != -1}

            if len(terminais_encontrados) >= 2:
                pontos_terminais = list(terminais_encontrados.values())
                max_dist = 0
                for p1, p2 in combinations(pontos_terminais, 2):
                    dist = p1.distance(p2)
                    if dist > max_dist:
                        max_dist, (terminal_A, terminal_B) = dist, (p1, p2)

        if terminal_A and terminal_B:
            # Filtrar pontos em movimento de forma mais eficiente
            pontos_em_movimento_idx = gdf_linha.index[gdf_linha['velocidade'] > 0]
            if len(pontos_em_movimento_idx) > 2:
                # Usar apenas as coordenadas necessárias
                pontos_em_movimento = gdf_linha.loc[pontos_em_movimento_idx, ['latitude', 'longitude']]
                kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(pontos_em_movimento)
                gdf_linha.loc[pontos_em_movimento.index, 'sentido'] = kmeans.labels_

                centroides = kmeans.cluster_centers_
                dist_centroide0_A = Point(centroides[0]).distance(terminal_A)
                dist_centroide1_A = Point(centroides[1]).distance(terminal_A)
                sentido_ida = 0 if dist_centroide0_A < dist_centroide1_A else 1
                sentido_map = {sentido_ida: 0, 1 - sentido_ida: 1} # 0 = Ida (A->B), 1 = Volta (B->A)
                gdf_linha.loc[pontos_em_movimento.index, 'sentido'] = gdf_linha.loc[pontos_em_movimento.index, 'sentido'].map(sentido_map)

                for sentido_id in [0, 1]:
                    pontos = gdf_linha[(gdf_linha.index.isin(pontos_em_movimento.index)) & (gdf_linha['sentido'] == sentido_id)]
                    if len(pontos) < 2: continue
                    
                    terminal_partida = terminal_A if sentido_id == 0 else terminal_B
                    # Projetar apenas os pontos necessários
                    pontos_proj = pontos.to_crs(projected_crs)
                    terminal_partida_geoseries = gpd.GeoSeries([terminal_partida], crs=gdf_linha.crs)
                    terminal_partida_proj = terminal_partida_geoseries.to_crs(projected_crs).iloc[0]
                    distancias_em_metros = pontos_proj.geometry.distance(terminal_partida_proj)
                    
                    # Ordenar e criar o trajeto
                    pontos_ordenados = pontos.assign(dist=distancias_em_metros).sort_values('dist')
                    trajetos_finais[sentido_id] = LineString(pontos_ordenados.geometry.tolist()).simplify(0.0001)
                    
                   
                    del pontos_proj, pontos_ordenados
                    gc.collect()
        
        # Verificar se conseguiu construir os trajetos
        if not (terminal_A and terminal_B and trajetos_finais):
            print(f"  -> AVISO: Não foi possível construir os trajetos para a linha {linha_atual}. Pulando.")
            return None
        
        print(f"  -> Trajetos de ida e volta construídos para a linha {linha_atual}.")
        
        # 4. ENGENHARIA DE FEATURES
        # Reduzir o dataframe apenas para os registros com sentido definido
        gdf_com_sentido = gdf_linha[gdf_linha['sentido'] != -1].copy()
        del gdf_linha  
        gc.collect()

        # Features de tempo cíclicas e categóricas (colunas calculadas de forma vetorizada)
        gdf_com_sentido['hora_do_dia'] = gdf_com_sentido['datahoraservidor'].dt.hour
        gdf_com_sentido['dia_da_semana'] = gdf_com_sentido['datahoraservidor'].dt.dayofweek
        gdf_com_sentido['fim_de_semana'] = (gdf_com_sentido['dia_da_semana'] >= 5).astype(int)
        gdf_com_sentido['hora_sin'] = np.sin(2 * np.pi * gdf_com_sentido['hora_do_dia'] / 24.0)
        gdf_com_sentido['hora_cos'] = np.cos(2 * np.pi * gdf_com_sentido['hora_do_dia'] / 24.0)
        gdf_com_sentido['dia_semana_sin'] = np.sin(2 * np.pi * gdf_com_sentido['dia_da_semana'] / 7.0)
        gdf_com_sentido['dia_semana_cos'] = np.cos(2 * np.pi * gdf_com_sentido['dia_da_semana'] / 7.0)

        # Identificar viagens e calcular tempos relativos (partida e chegada)
        gdf_com_sentido = gdf_com_sentido.sort_values(by=['ordem', 'datahoraservidor'])
        gdf_com_sentido['viagem_id'] = (gdf_com_sentido['sentido'] != gdf_com_sentido['sentido'].shift(1)).cumsum()
        
        # Calcular estatísticas das viagens de forma mais otimizada
        viagem_stats = gdf_com_sentido.groupby('viagem_id')['datahoraservidor'].agg(['min', 'max']).reset_index()
        viagem_stats.columns = ['viagem_id', 'horario_partida', 'horario_chegada_previsto']
        gdf_com_sentido = pd.merge(gdf_com_sentido, viagem_stats, on='viagem_id')
        
        gdf_com_sentido['tempo_desde_partida'] = (gdf_com_sentido['datahoraservidor'] - gdf_com_sentido['horario_partida']).dt.total_seconds()
        gdf_com_sentido['tempo_ate_final'] = (gdf_com_sentido['horario_chegada_previsto'] - gdf_com_sentido['datahoraservidor']).dt.total_seconds()

        # Features de distância baseadas no trajeto canônico
        comprimento_trajetos = {s: t.length for s, t in trajetos_finais.items()}
        
        # Calcular features de distância
        features_distancia = []
        for idx, row in gdf_com_sentido.iterrows():
            trajeto_especifico = trajetos_finais.get(row['sentido'])
            if not trajeto_especifico:
                features_distancia.append([0, 0, 0])
                continue
                
            comprimento_total = comprimento_trajetos.get(row['sentido'], 0)
            distancia_no_trajeto = trajeto_especifico.project(row['geom'])
            percentual = distancia_no_trajeto / comprimento_total if comprimento_total > 0 else 0
            dist_final = comprimento_total - distancia_no_trajeto
            features_distancia.append([distancia_no_trajeto, percentual, dist_final])
            
        # Converter para DataFrame e concatenar
        dist_df = pd.DataFrame(features_distancia, 
                              index=gdf_com_sentido.index,
                              columns=['distancia_trajeto', 'percentual_trajeto', 'distancia_ate_final'])
        gdf_linha_final = pd.concat([gdf_com_sentido, dist_df], axis=1)
        
        # Liberação de memória
        del gdf_com_sentido, dist_df
        gc.collect()
        
        # Limpeza final
        gdf_linha_final.fillna(0, inplace=True)
        print("  -> Engenharia de features concluída.")

        # 5. Treinamento do Modelo
        if modo_treino == 'POSICAO':
            target_cols = ['latitude', 'longitude']
            features = ['tempo_desde_partida', 'sentido', 'hora_sin', 'hora_cos', 'dia_semana_sin', 'dia_semana_cos', 'fim_de_semana']
        else: # ETA
            target_cols = ['tempo_ate_final']
            features = ['latitude', 'longitude', 'sentido', 'hora_sin', 'hora_cos', 'fim_de_semana', 'distancia_trajeto', 'percentual_trajeto', 'distancia_ate_final']
        
        # Extrair apenas as colunas necessárias para o treinamento
        X = gdf_linha_final[features].copy()
        y = gdf_linha_final[target_cols].copy()
        
        # Limpar o dataframe completo para liberar memória
        del gdf_linha_final
        gc.collect()

        if X.empty or y.isnull().values.any():
            print(f"  -> AVISO: Sem dados válidos para treinar o modelo para a linha {linha_atual}. Pulando.")
            return None
            
        # Criar e treinar o modelo
        lgbm_base = lgb.LGBMRegressor(objective='regression_l1', seed=42)
        model = MultiOutputRegressor(lgbm_base) if modo_treino == 'POSICAO' else lgbm_base
        model.fit(X, y)

        # 6. Salvar os artefatos (modelo, trajetos e terminais)
        artefatos = {
            'model': model,
            'trajetos': trajetos_finais,
            'terminais': {'A': terminal_A, 'B': terminal_B},
            'features': features,
            'comprimento_trajetos': comprimento_trajetos
        }
        caminho_artefato = os.path.join(pasta_modelos, f'artefatos_{modo_treino.lower()}_linha_{linha_atual}.joblib')
        joblib.dump(artefatos, caminho_artefato)
        print(f"  -> Artefatos para linha {linha_atual} treinados e salvos em: {caminho_artefato}")
        
        return linha_atual
    except Exception as e:
        print(f"  -> ERRO ao processar linha {linha_atual}: {e}")
        traceback.print_exc()
        return None


def treinar_e_salvar_modelos(linhas_para_treinar, modo_treino, db_engine, pasta_modelos='modelos_salvos'):
    """
    Função completa para treinar um modelo para cada linha, incluindo a construção
    do trajeto canônico, engenharia de features e salvamento dos artefatos em disco.
    """
    print(f"--- INICIANDO TREINAMENTO EM MODO: {modo_treino} ---")
    os.makedirs(pasta_modelos, exist_ok=True)
    
    if not linhas_para_treinar:
        print(f"Nenhuma linha nova para treinar no modo {modo_treino}.")
        return

    # 1. Carregamento de Dados - processar em lotes para reduzir uso de memória
    linhas_sql = tuple(linhas_para_treinar)
    
    # Carregar dados por grupos de linhas se houver muitas
    if len(linhas_para_treinar) > 3:
        # Processar em lotes de 3 linhas para evitar sobrecarga de memória
        lotes = [linhas_para_treinar[i:i+3] for i in range(0, len(linhas_para_treinar), 3)]
        for lote in lotes:
            _treinar_linhas_lote(lote, modo_treino, db_engine, pasta_modelos)
    else:
        _treinar_linhas_lote(linhas_para_treinar, modo_treino, db_engine, pasta_modelos)


def _treinar_linhas_lote(lote_linhas, modo_treino, db_engine, pasta_modelos):
    """Função auxiliar para treinar um lote de linhas"""
    linhas_sql = tuple(lote_linhas)
    if len(lote_linhas) == 1:
        # SQLAlchemy precisa de sintaxe especial para tupla com um único elemento
        sql_query = f"""
            SELECT * FROM gps_data 
            WHERE linha = '{lote_linhas[0]}'
            AND datahoraservidor >= '{DATA_INICIO_TREINO}' AND datahoraservidor < '{DATA_FIM_TREINO}'
        """
    else:
        sql_query = f"""
            SELECT * FROM gps_data 
            WHERE linha IN {str(linhas_sql)}
            AND datahoraservidor >= '{DATA_INICIO_TREINO}' AND datahoraservidor < '{DATA_FIM_TREINO}'
        """
    
    try:
        # Carregar dados e aplicar filtragem inicial
        gdf = gpd.read_postgis(sql_query, db_engine, geom_col='geom')
        print(f"Total de {len(gdf)} registros carregados para o treinamento do modo {modo_treino}.")
    except Exception as e:
        print(f"Erro ao carregar dados do banco: {e}")
        return

    # 2. Limpeza e Pré-processamento
    gdf['datahoraservidor'] = pd.to_datetime(gdf['datahoraservidor'])
    gdf = gdf.sort_values(by=['linha', 'ordem', 'datahoraservidor'])
    gdf['hora'] = gdf['datahoraservidor'].dt.hour
    
    # Aplicar filtros para reduzir dados
    gdf_operacao = gdf[(gdf['hora'].between(4, 23)) & (gdf['velocidade'] <= 120)].copy()
    gdf_operacao = gdf_operacao.drop_duplicates()
    gdf_operacao['sentido'] = -1  # Inicializa a coluna sentido
    del gdf  
    gc.collect()
    
    print(f"Após a limpeza, restaram {len(gdf_operacao)} registros para a análise.")
    
    # Processar cada linha em paralelo usando ProcessPoolExecutor
    linhas_processadas = []
    tasks = []
    
    # Dividir o dataframe por linha
    grupos_por_linha = {linha: grupo.copy() for linha, grupo in gdf_operacao.groupby('linha')}
    del gdf_operacao  
    gc.collect()
    
    # Processar cada linha de forma paralela
    with ProcessPoolExecutor(max_workers=min(len(grupos_por_linha), NUM_WORKERS)) as executor:
        futures = []
        for linha, grupo in grupos_por_linha.items():
            futures.append(executor.submit(
                processar_linha, linha, grupo, pasta_modelos, modo_treino
            ))
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    linhas_processadas.append(result)
            except Exception as e:
                print(f"Erro durante processamento paralelo: {e}")
    
    print(f"Linhas processadas com sucesso: {linhas_processadas}")


def prever_com_modelo_salvo(caminho_arquivo_teste, pasta_modelos='modelos_salvos'):
    """
    Função otimizada que carrega um modelo e seus artefatos para fazer a previsão,
    replicando a engenharia de features do treino.
    """
    print(f"\n{'='*25}\nIniciando predição para: {os.path.basename(caminho_arquivo_teste)}\n{'='*25}")
    try:
        with open(caminho_arquivo_teste, 'r', encoding='utf-8') as f:
            teste_data = json.load(f)
            if not teste_data:
                print("Arquivo de teste vazio.")
                return
            teste_df = pd.DataFrame(teste_data)

        if 'datahora' in teste_df.columns:
            modo_necessario = 'POSICAO'
            print("Modo detectado: POSICAO")
        elif 'latitude' in teste_df.columns:
            modo_necessario = 'ETA'
            print("Modo detectado: ETA")
        else:
            raise ValueError("Formato de arquivo de teste inválido.")

        # Dividir as previsões por linha para processar em paralelo
        grupos_por_linha = {linha: grupo.copy() for linha, grupo in teste_df.groupby('linha')}
        previsoes_por_linha = []
        
        # Processar cada linha em paralelo
        with ProcessPoolExecutor(max_workers=min(len(grupos_por_linha), NUM_WORKERS)) as executor:
            futures = {}
            for linha, grupo in grupos_por_linha.items():
                futures[executor.submit(_prever_para_linha, linha, grupo, modo_necessario, pasta_modelos)] = linha
            
            for future in as_completed(futures):
                try:
                    resultado = future.result()
                    if resultado:
                        previsoes_por_linha.extend(resultado)
                except Exception as e:
                    linha = futures[future]
                    print(f"Erro ao prever para a linha {linha}: {e}")
        
        if previsoes_por_linha:
            now = datetime.datetime.now()
            payload_api = {
                "aluno": NOME_ALUNO, "senha": SENHA_SECRETA,
                "datahora": now.strftime('%Y-%m-%d %H:%M:%S'),
                "previsoes": sorted(previsoes_por_linha, key=lambda x: x[0])
            }
            print("\n--- Payload Final ---")
            print(json.dumps(payload_api, indent=2))
        else:
            print("Nenhuma previsão foi gerada.")

    except Exception as e:
        print(f"\nERRO GERAL: {e}")
        print(traceback.format_exc())


def _prever_para_linha(linha, grupo, modo_necessario, pasta_modelos):
    """Função auxiliar para fazer previsão para uma linha específica"""
    try:
        print(f"--- Processando linha {linha} ---")
        caminho_artefato = os.path.join(pasta_modelos, f'artefatos_{modo_necessario.lower()}_linha_{linha}.joblib')
        
        if not os.path.exists(caminho_artefato):
            print(f"  -> AVISO: Artefatos para a linha {linha} não encontrados. Pulando.")
            return []

        artefatos = joblib.load(caminho_artefato)
        modelo = artefatos['model']
        trajetos = artefatos['trajetos']
        features_treino = artefatos['features']
        comprimento_trajetos = artefatos['comprimento_trajetos']

        # Feature Engineering para o grupo de teste
        now = datetime.datetime.now()
        grupo['hora_do_dia'] = now.hour
        grupo['dia_da_semana'] = now.weekday()
        grupo['fim_de_semana'] = (grupo['dia_da_semana'] >= 5).astype(int)
        grupo['hora_sin'] = np.sin(2 * np.pi * grupo['hora_do_dia'] / 24.0)
        grupo['hora_cos'] = np.cos(2 * np.pi * grupo['hora_do_dia'] / 24.0)
        grupo['dia_semana_sin'] = np.sin(2 * np.pi * grupo['dia_da_semana'] / 7.0)
        grupo['dia_semana_cos'] = np.cos(2 * np.pi * grupo['dia_da_semana'] / 7.0)
        
        if modo_necessario == 'POSICAO':
            if 'sentido' not in grupo.columns:
                grupo['sentido'] = 0
            grupo['datahoraservidor'] = pd.to_datetime(grupo['datahora'], unit='ms')
            grupo['tempo_desde_partida'] = (grupo['datahoraservidor'] - grupo['datahoraservidor'].min()).dt.total_seconds()
        else: # ETA
            # Converter latitude/longitude para números
            grupo['latitude'] = pd.to_numeric(grupo['latitude'].str.replace(',', '.'))
            grupo['longitude'] = pd.to_numeric(grupo['longitude'].str.replace(',', '.'))
            grupo['geom'] = grupo.apply(lambda r: Point(r['longitude'], r['latitude']), axis=1)

            # Inferir sentido para cada ponto
            sentidos = []
            for _, row in grupo.iterrows():
                dist_ida = row['geom'].distance(trajetos.get(0, LineString()))
                dist_volta = row['geom'].distance(trajetos.get(1, LineString()))
                sentidos.append(0 if dist_ida < dist_volta else 1)
            grupo['sentido'] = sentidos
            
            # Calcular features de distância em lote
            features_distancia = []
            for _, row in grupo.iterrows():
                trajeto_especifico = trajetos.get(row['sentido'])
                if not trajeto_especifico:
                    features_distancia.append([0, 0, 0])
                    continue
                    
                comprimento_total = comprimento_trajetos.get(row['sentido'], 0)
                dist_trajeto = trajeto_especifico.project(row['geom'])
                percentual = dist_trajeto / comprimento_total if comprimento_total > 0 else 0
                dist_final = comprimento_total - dist_trajeto
                features_distancia.append([dist_trajeto, percentual, dist_final])
            
            # Adicionar as features calculadas ao dataframe
            dist_df = pd.DataFrame(features_distancia, 
                                  index=grupo.index,
                                  columns=['distancia_trajeto', 'percentual_trajeto', 'distancia_ate_final'])
            grupo = pd.concat([grupo, dist_df], axis=1)

        # Fazer a predição
        X_teste = grupo[features_treino]
        previsoes = modelo.predict(X_teste)

        # Formatar as previsões
        resultados = []
        for i, (idx, row) in enumerate(grupo.iterrows()):
            if modo_necessario == 'POSICAO':
                resultado = {"latitude": previsoes[i, 0], "longitude": previsoes[i, 1]}
            else: # ETA
                eta_timestamp = now + datetime.timedelta(seconds=int(previsoes[i]))
                resultado = int(eta_timestamp.timestamp() * 1000)
            
            resultados.append([row['id'], resultado])
        
        return resultados
    
    except Exception as e:
        print(f"Erro ao processar linha {linha}: {e}")
        traceback.print_exc()
        return []

# =====================================================================================
# ||                                     EXECUÇÃO                                    ||
# =====================================================================================

if __name__ == '__main__':
    pasta_modelos = r'Data-Mining\T3\src\modelos_salvos'
    pasta_testes = r'Data-Mining\T3\data\tests' 

    # --- ETAPA 1: VERIFICAR ARQUIVOS DE TESTE E TREINAR MODELOS NECESSÁRIOS ---
    print("--- INICIANDO PROCESSO AUTOMATIZADO DE TREINAMENTO E PREVISÃO ---")
    
    # Cria as pastas se não existirem
    os.makedirs(pasta_modelos, exist_ok=True)
    os.makedirs(pasta_testes, exist_ok=True)
    
    # Identificar modelos necessários a partir dos arquivos de teste
    arquivos_teste = [os.path.join(pasta_testes, f) for f in os.listdir(pasta_testes) if f.endswith('.json')]
    
    if not arquivos_teste:
        print(f"Nenhum arquivo de teste .json encontrado em '{pasta_testes}'. O script não continuará.")
    else:
        modelos_necessarios = set()
        arquivos_por_modo = {'POSICAO': [], 'ETA': []}
        
        print("\nAnalisando arquivos de teste para determinar modelos necessários...")
        for arquivo in arquivos_teste:
            try:
                with open(arquivo, 'r', encoding='utf-8') as f:
                    teste_data = json.load(f)
                    if not teste_data:
                        print(f"AVISO: Arquivo de teste '{os.path.basename(arquivo)}' está vazio. Pulando.")
                        continue
                    
                    # Carregar apenas as colunas necessárias para economizar memória
                    if 'datahora' in teste_data[0]:
                        modo = 'POSICAO'
                        df_teste = pd.DataFrame(teste_data)[['id', 'linha', 'datahora']]
                    elif 'latitude' in teste_data[0]:
                        modo = 'ETA'
                        df_teste = pd.DataFrame(teste_data)[['id', 'linha', 'latitude', 'longitude']]
                    else:
                        print(f"AVISO: Formato desconhecido no arquivo '{os.path.basename(arquivo)}'.")
                        continue

                    linhas_no_arquivo = df_teste['linha'].unique()
                    arquivos_por_modo[modo].append(arquivo)

                    for linha in linhas_no_arquivo:
                        modelos_necessarios.add((linha, modo))

            except Exception as e:
                print(f"Erro ao processar arquivo de teste '{os.path.basename(arquivo)}': {e}")
        
        # Verificar quais modelos precisam ser treinados
        modelos_a_treinar = {'POSICAO': set(), 'ETA': set()}
        for linha, modo in modelos_necessarios:
            caminho_artefato = os.path.join(pasta_modelos, f'artefatos_{modo.lower()}_linha_{linha}.joblib')
            if not os.path.exists(caminho_artefato):
                modelos_a_treinar[modo].add(linha)

        # Executar treinamento seletivo
        try:
            if modelos_a_treinar['POSICAO']:
                print(f"\nModelos de POSICAO faltando para as linhas: {list(modelos_a_treinar['POSICAO'])}")
                treinar_e_salvar_modelos(list(modelos_a_treinar['POSICAO']), 'POSICAO', engine, pasta_modelos)
            
            if modelos_a_treinar['ETA']:
                print(f"\nModelos de ETA faltando para as linhas: {list(modelos_a_treinar['ETA'])}")
                treinar_e_salvar_modelos(list(modelos_a_treinar['ETA']), 'ETA', engine, pasta_modelos)

            if not modelos_a_treinar['POSICAO'] and not modelos_a_treinar['ETA']:
                print("\nTodos os modelos necessários já estão treinados e prontos.")

        except Exception as e:
            print(f"Erro durante a etapa de treinamento automático: {e}")

        # --- ETAPA 2: REALIZAR PREVISÕES ---
        print("\n--- INICIANDO ETAPA DE PREVISÃO ---")
        for modo, lista_arquivos in arquivos_por_modo.items():
            if lista_arquivos:
                print(f"\nRealizando previsões para o modo: {modo}")
                for arquivo in lista_arquivos:
                    prever_com_modelo_salvo(arquivo, pasta_modelos)

    print("\n--- PROCESSO FINALIZADO ---")
  
    gc.collect()
