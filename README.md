# HERMES 🕵️

Assistente que **observa conversas** (das quais não participa), interpreta o
contexto e gera artefatos úteis — como resumos em PDF com referências.

> **Fase 0** — Prova de conceito: acionamento → gravação → transcrição →
> resumo (LLM) → PDF.

## ⚠️ Privacidade e consentimento
O HERMES grava conversas. **Use apenas com o consentimento de todos os
participantes** e avise quando estiver ativo. Respeite a LGPD.

## Instalação

\`\`\`bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # ajuste as variáveis
\`\`\`

> Dica Linux: se `sounddevice` falhar, instale `libportaudio2`.
> Para Ollama: baixe em https://ollama.com e rode `ollama pull llama3.1`.

## Uso

\`\`\`bash
python run.py
\`\`\`

1. Pressione **ENTER** (ou diga o wake word) para ativar o HERMES.
2. Fale naturalmente — o HERMES escuta a conversa inteira.
3. Acione novamente para parar e processar.
4. O resumo em PDF será salvo na pasta `output/`.

## Roadmap
- Fase 1: diariação, classificação de tema, referências acadêmicas reais.
- Fase 2: memória (RAG), modo 100% local.
- Fase 3: dashboard ao vivo, flashcards, TTS.

## Como rodar (passo a passo)

# 1. Crie o ambiente virtual
python -m venv .venv
source .venv/bin/activate        # no Windows: .venv\Scripts\activate

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Configure o .env
cp .env.example .env

# 4. (opcional, para LLM local e grátis) baixe o Ollama + modelo
ollama pull llama3.1

# 5. Rode
python run.py
