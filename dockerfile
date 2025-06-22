# Use uma imagem base do Python
FROM python:3.10-slim

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Instala dependências do sistema necessárias para o GeoPandas
RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Copia o arquivo de dependências
COPY requirements.txt .

# Instala as bibliotecas Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código da sua aplicação para o contêiner
COPY . .

# Comando padrão para manter o contêiner rodando (opcional)
CMD ["tail", "-f", "/dev/null"]