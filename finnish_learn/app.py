from flask import Flask, render_template, request, jsonify
import csv
import json
import os

app = Flask(__name__)

CSV_FILE = 'data/dictionary.csv'
STATS_FILE = 'data/stats.json'

def load_vocab():
    vocab = []
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                vocab.append({'word': row['Word'].strip(), 'definition': row['Definition'].strip()})
    return vocab

def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_stats(stats):
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=4)

@app.route('/')
def index():
    vocab = load_vocab()
    return render_template('index.html', vocab_count=len(vocab))

@app.route('/learn')
def learn():
    return render_template('learn.html', vocab=load_vocab())

@app.route('/flashcard')
def flashcard():
    return render_template('flashcard.html', vocab=load_vocab())

@app.route('/quiz/mc')
def quiz_mc():
    return render_template('mc.html', vocab=load_vocab())

@app.route('/quiz/typing')
def quiz_typing():
    return render_template('typing.html', vocab=load_vocab())

@app.route('/stats')
def stats():
    stats_data = load_stats()
    total_attempts = sum(d['attempts'] for d in stats_data.values())
    total_correct = sum(d['correct'] for d in stats_data.values())
    overall_acc = round((total_correct / total_attempts * 100), 1) if total_attempts > 0 else 0
    return render_template('stats.html', stats=stats_data, overall_acc=overall_acc, attempts=total_attempts)

@app.route('/api/stats', methods=['POST'])
def update_stats():
    data = request.json
    word = data.get('word')
    is_correct = data.get('correct')
    time_taken = data.get('time')

    stats = load_stats()
    if word not in stats:
        stats[word] = {'attempts': 0, 'correct': 0, 'fastest_time': float('inf')}
    
    stats[word]['attempts'] += 1
    if is_correct:
        stats[word]['correct'] += 1
        if time_taken and time_taken < stats[word]['fastest_time']:
            stats[word]['fastest_time'] = round(time_taken, 2)
    
    save_stats(stats)
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)