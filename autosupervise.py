import torch
import torch.nn as nn
import numpy as np
import os
from tokenizer import CharTokenizer
from model import NanoGPT

# Configuración
config = {
    'context_size': 128,
    'd_model': 128,
    'n_layers': 4,
    'n_heads': 4,
    'd_ff': 512,
    'num_candidates': 5,  # Cuántas respuestas generar por pregunta
    'max_length': 100
}

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
            # Divide por saltos de línea dobles como separador de ejemplos
            ejemplos = [ej.strip() for ej in text.split('\n\n') if ej.strip()]
            preguntas = [ej.split('\n')[0] for ej in ejemplos if '\n' in ej]
            return preguntas
    else:
        print('No se encontraron archivos de preguntas.')
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
    # Ordena por mayor probabilidad promedio
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates

def main():
    device = torch.device('cpu')
    if not os.path.exists('tokenizer.json') or not os.path.exists('gpt_nano.pth'):
        print('Faltan tokenizer.json o gpt_nano.pth. Entrena el modelo primero.')
        return
    tokenizer = CharTokenizer()
    tokenizer.load('tokenizer.json')
    vocab_size = len(tokenizer.stoi)
    model = NanoGPT(vocab_size, d_model=config['d_model'], n_layers=config['n_layers'],
                    n_heads=config['n_heads'], d_ff=config['d_ff'], context_size=config['context_size'])
    model.load_state_dict(torch.load('gpt_nano.pth', map_location=device))
    model.to(device)

    preguntas = load_questions()
    if not preguntas:
        print('No hay preguntas para auto-supervisar.')
        return
    resultados = []
    for pregunta in preguntas:
        candidatos = sample_candidates(model, tokenizer, device, pregunta, num_candidates=config['num_candidates'], length=config['max_length'])
        mejor_respuesta, score = candidatos[0]
        resultados.append(f"{pregunta} || {mejor_respuesta}")
        print(f"Pregunta: {pregunta}\nMejor respuesta: {mejor_respuesta}\nScore: {score:.4f}\n---")
    # Guarda los pares pregunta||respuesta auto-supervisados
    with open('autosupervised.txt', 'w', encoding='utf-8') as f:
        for r in resultados:
            f.write(r + '\n')
    print('Auto-supervisión completada. Resultados guardados en autosupervised.txt')

if __name__ == '__main__':
    main()
