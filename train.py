import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
from tokenizer import CharTokenizer
from model import NanoGPT
import random

def load_questions():
    # Intenta cargar preguntas de feedback.txt, si no existe usa input.txt
    if os.path.exists('feedback.txt'):
        with open('feedback.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            preguntas = [line.split('||')[0].strip() for line in lines if '||' in line]
            return preguntas
    elif os.path.exists('input.txt'):
        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
            ejemplos = [ej.strip() for ej in text.split('\n\n') if ej.strip()]
            preguntas = [ej.split('\n')[0] for ej in ejemplos if '\n' in ej]
            return preguntas
    else:
        return []

def sample_candidates(model, tokenizer, device, context, num_candidates=5, length=100):
    model.eval()
    candidates = []
    for _ in range(num_candidates):
        idx = torch.tensor([tokenizer.encode(context)], dtype=torch.long).to(device)
        for _ in range(length):
            with torch.no_grad():
                logits = model(idx[:, -config['context_size']:])
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_id], dim=1)
        respuesta = tokenizer.decode(idx[0].tolist())[len(context):].strip()
        # Calcula la probabilidad promedio de la secuencia generada
        with torch.no_grad():
            logits = model(idx[:, :-1])
            log_probs = torch.log_softmax(logits, dim=-1)
            seq_probs = log_probs[0, torch.arange(len(context), idx.size(1)-1), idx[0, len(context)+1:]]
            avg_prob = seq_probs.mean().item() if seq_probs.numel() > 0 else -float('inf')
        candidates.append((respuesta, avg_prob))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates

# Configuración
config = {
    'batch_size': 32,
    'context_size': 128,
    'epochs': 10000,
    'lr': 3e-4,
    'd_model': 128,
    'n_layers': 4,
    'n_heads': 4,
    'd_ff': 512,
    'refine': True  # Si es True, continúa entrenando desde el modelo guardado
}

def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def get_batches(data, batch_size, context_size):
    n_batches = len(data) // (batch_size * context_size)
    data = data[:n_batches * batch_size * context_size]
    data = np.array(data).reshape(batch_size, -1)
    for i in range(0, data.shape[1] - context_size, context_size):
        x = data[:, i:i+context_size]
        y = data[:, i+1:i+context_size+1]
        yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def sample(model, tokenizer, device, context, length=200):
    model.eval()
    idx = torch.tensor([tokenizer.encode(context)], dtype=torch.long).to(device)
    for _ in range(length):
        with torch.no_grad():
            logits = model(idx[:, -config['context_size']:])
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
    return tokenizer.decode(idx[0].tolist())

def main():
    device = torch.device('cpu')
    text = load_text('input.txt')
    # Refinamiento: usar el mismo vocabulario que el modelo previo
    if config.get('refine', False) and os.path.exists('tokenizer.json'):
        print('Cargando vocabulario previo para refinamiento...')
        tokenizer = CharTokenizer()
        tokenizer.load('tokenizer.json')
    else:
        tokenizer = CharTokenizer(text)
        tokenizer.save('tokenizer.json')
    data = tokenizer.encode(text)
    vocab_size = len(tokenizer.stoi)
    model = NanoGPT(vocab_size, d_model=config['d_model'], n_layers=config['n_layers'],
                    n_heads=config['n_heads'], d_ff=config['d_ff'], context_size=config['context_size'])
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()

    # Refinamiento: cargar modelo previo si refine=True
    if config.get('refine', False) and os.path.exists('gpt_nano.pth'):
        print('Cargando modelo previo para refinamiento...')
        model.load_state_dict(torch.load('gpt_nano.pth', map_location=device))

    for epoch in range(config['epochs']):
        model.train()
        losses = []
        pbar = tqdm(get_batches(data, config['batch_size'], config['context_size']),
                    desc=f"Época {epoch+1}/{config['epochs']}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix({'loss': np.mean(losses)})
        # Generar muestra
        muestra = sample(model, tokenizer, device, context="Hola", length=100)
        print(f"\n--- Muestra: ---\n{muestra}\n")

        # --- Auto-supervisión al final de cada época ---
        preguntas = load_questions()
        if preguntas:
            # Selecciona hasta 5 preguntas aleatorias para auto-supervisión
            preguntas_sample = random.sample(preguntas, min(5, len(preguntas)))
            resultados = []
            for pregunta in preguntas_sample:
                candidatos = sample_candidates(model, tokenizer, device, pregunta, num_candidates=5, length=100)
                mejor_respuesta, score = candidatos[0]
                resultados.append(f"{pregunta} || {mejor_respuesta}")
                print(f"[Auto-supervisión] Pregunta: {pregunta}\nMejor respuesta: {mejor_respuesta}\nScore: {score:.4f}\n---")
            # Acumula en autosupervised.txt
            with open('autosupervised.txt', 'a', encoding='utf-8') as f:
                for r in resultados:
                    f.write(r + '\n')
            print('[Auto-supervisión] Resultados guardados en autosupervised.txt')
    # Guardar modelo
    torch.save(model.state_dict(), 'gpt_nano.pth')
    print("Modelo guardado en gpt_nano.pth")

if __name__ == '__main__':
    main()
