from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from spellchecker import SpellChecker 

app = Flask(__name__)


tokenizer = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")

model = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction")
spell = SpellChecker() 

def correct_spelling(text):
    corrected_words = []
    for word in text.split():
        
        if word.lower() in spell:
            corrected_words.append(word)
        else:
            corrected_words.append(spell.correction(word) or word)
    return ' '.join(corrected_words)

def autocorrect_text(text):
   
    spelling_corrected = correct_spelling(text)
   
    input_text = "gec: " + spelling_corrected
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    outputs = model.generate(input_ids, max_length=128, num_beams=5, early_stopping=True)
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected

@app.route('/', methods=['GET', 'POST'])
def index():
    corrected = ""
    user_input = ""
    if request.method == 'POST':

        user_input = request.form['text']
        corrected = autocorrect_text(user_input)

    return render_template('Front.html', user_input=user_input, corrected=corrected)

if __name__ == '__main__':
    app.run(debug=True)