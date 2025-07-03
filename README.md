# NanoGPT desde cero en PyTorch

## Instrucciones

1. **Coloca tu texto en `input.txt`**
   - Usa cualquier texto plano para entrenar el modelo.

2. **Instala dependencias**
   ```bash
   pip install -r requirements.txt
   ```

3. **Entrena el modelo**
   ```bash
   python train.py
   ```

4. **Genera texto con el modelo entrenado**
   ```bash
   python generate.py
   ```

## Archivos
- `requirements.txt`: dependencias necesarias
- `tokenizer.py`: tokenizador de caracteres
- `model.py`: arquitectura NanoGPT
- `train.py`: entrenamiento desde cero
- `generate.py`: generación interactiva
- `input.txt`: tu corpus de texto

## Notas
- El modelo está optimizado para CPU y hardware básico.
- Todo el código es auto-contenido y comentado.
- Puedes ajustar los hiperparámetros en el diccionario `config` de `train.py`.
