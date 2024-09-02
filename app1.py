from flask import Flask, request, jsonify
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

# Load models and tokenizers for both translation directions
model_en_to_es = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-es')
tokenizer_en_to_es = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-es')

model_es_to_en = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-es-en')
tokenizer_es_to_en = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-en')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get('text')
    direction = data.get('direction')

    if direction == 'en_to_es':
        model = model_en_to_es
        tokenizer = tokenizer_en_to_es
    elif direction == 'es_to_en':
        model = model_es_to_en
        tokenizer = tokenizer_es_to_en
    else:
        return jsonify({'error': 'Invalid direction'}), 400

    input_ids = tokenizer.encode(text, return_tensors="pt")
    translated = model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
    translation = tokenizer.decode(translated[0], skip_special_tokens=True)

    return jsonify({'translation': translation})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
