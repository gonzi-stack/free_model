import torch
from tokenizer import CharTokenizer
from model import NanoGPT
import os


def load_model_and_tokenizer():
    tokenizer = CharTokenizer()
    tokenizer.load('tokenizer.json')
    vocab_size = len(tokenizer.stoi)
    model = NanoGPT(vocab_size)
    model.load_state_dict(torch.load('gpt_nano.pth', map_location='cpu'))
    model.eval()
    return model, tokenizer


def generate_text(model, tokenizer, prompt, context_size=128, length=200):
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    for _ in range(length):
        with torch.no_grad():
            logits = model(idx[:, -context_size:])
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
    return tokenizer.decode(idx[0].tolist())


def save_feedback(prompt, respuesta_ideal, path='feedback.txt'):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(f'{prompt.strip()} | {respuesta_ideal.strip()}\n')


def main():
    model, tokenizer = load_model_and_tokenizer()
    context_size = 128
    print("Escribe un prompt (Ctrl+C para salir):")
    prompt = ""
    while True:
        try:
            user_input = input("Usuario: ")
            prompt += user_input
            output = generate_text(model, tokenizer, prompt, context_size, length=100)
            respuesta = output[len(prompt):]
            print(f"Modelo: {respuesta}")
            # Feedback interactivo
            feedback = input("¿La respuesta fue la esperada? (S/N): ").strip().lower()
            if feedback == 'n':
                ideal = input("¿Cómo debería haber respondido Zeew?: ")
                save_feedback(prompt, ideal)
                print("¡Gracias! Respuesta guardada para refinar el modelo.")
            prompt += respuesta
        except KeyboardInterrupt:
            print("\nSaliendo...")
            break


if __name__ == '__main__':
    main()
