
import click 
import sqlite3
import os
import json
import itertools
import datetime
import textwrap
from typing import List
from sqlite3 import Connection
from dataclasses import dataclass
from tabulate import tabulate
import openai


with open("config.json", "r") as file:
    config = json.load(file)
openai.api_key = config["token"]


BASE_PATH = ".vocaby/"
MODEL_ID = "gpt-3.5-turbo"
PROMPT = """
I need you to translate some words to russian for me. I will send you lines, 
each containing the word that needs to be translated and a sentence in which 
it was used. The word and the sentence are separated by ::. Answer in JSON, 
where you should include the original normalized word (name the field 
"word"), translation of the word (name the field "translation"), transcription 
of the word (name the field "transcription"), and the sentence (name the field 
"usage") in which it was used. For the usage field you should highlight the 
original word in bold using html tag <b>. Do not translate the sentence. Only 
highlight the original word in the sentence, that was requested to be translated. 
Respond with a JSON list, where you should include all of the translations.
"""


@dataclass
class Lookup:
    word: str
    usage: str
    book: str
    timestamp: int


@dataclass
class Translation:
    word: str
    translation: str
    transcription: str
    usage: str
    book: str
    timestamp: int


@click.command()
@click.option("--limit", default=20, help="Number of lookup words to scan")
@click.option("--no-checkpoint", is_flag=True, default=False, help="Whether to update checkpoints")
@click.option("--stats", is_flag=True, default=False, help="Show only stats")
@click.argument("filename")
def vocaby(limit, no_checkpoint, stats, filename):
    """
    Parse Kindle vocab.db file and transform the lookup words
    into Anki flashcards.
    """
    timestamp = load_latest_timestamp()
    click.echo(f"Latest scanned timestamp: {datetime.datetime.fromtimestamp(timestamp // 1000)}")

    conn = sqlite3.connect(filename)
    num_unprocessed = get_unprocessed_words_count(conn, timestamp)
    history = load_history()
    by_book = get_unprocessed_words_count_by_books(conn, timestamp)
    if num_unprocessed == 0:
        return click.echo("No new words")
    click.echo(f"Words left unprocessed in the database: {num_unprocessed}")
    click.echo(tabulate(by_book))

    if not stats:
        confirmed: List[Translation] = []
        rejected: List[Translation] = []
        for i, tr in enumerate(query_gpt(query_db(conn, timestamp, limit))):
            click.clear()
            click.echo(f"{i + 1}/{limit}\n")
            click.echo(f"{tr.word} [{tr.transcription}] = {tr.translation}\n")
            if tr.word in history:
                click.echo("#"*80)
                click.echo("This word was already processed!\n\n".upper())
                click.echo("\n---\n".join(map(lambda x: textwrap.fill(x, width=80), history[tr.word])))
                click.echo("#"*80 + "\n")
            click.echo(textwrap.fill(tr.usage, width=80))
            click.echo("")
            click.echo(f"{tr.book}")
            click.echo("---\n")
            if click.confirm("Save this translation?"):
                confirmed.append(tr)
                history.setdefault(tr.word.lower(), []).append(tr.usage)
            else:
                rejected.append(tr)
        click.clear()

        combined = list(itertools.chain(confirmed, rejected))
        if len(combined) == 1:
            timestamp = combined[0]
        else:
            timestamp = max(map(lambda x: x.timestamp, combined))
        
        filename = dump(confirmed, "anki")
        click.echo(f"Saved new words to {filename}")
        
        if no_checkpoint:
            click.echo("Skipped updating checkpoints")
        else:
            click.echo("Updated checkpoints")
            save_latest_timestamp(timestamp)
            save_history(history)
            num_unprocessed = get_unprocessed_words_count(conn, timestamp)
            click.echo(f"Words left unprocessed in the database: {num_unprocessed}")
    conn.close()


def load_latest_timestamp() -> int:
    if not os.path.exists(os.path.join(BASE_PATH, "checkoint.json")):
        return 0
    with open(os.path.join(BASE_PATH, "checkoint.json")) as file:
        return int(json.load(file).get("timestamp", 0))


def save_latest_timestamp(timestamp: int):
    os.makedirs(BASE_PATH, exist_ok=True)
    with open(os.path.join(BASE_PATH, "checkoint.json"), "w") as file:
        json.dump({"timestamp": timestamp}, file)


def load_history():
    if not os.path.exists(os.path.join(BASE_PATH, "history.json")):
        return {}
    with open(os.path.join(BASE_PATH, "history.json")) as file:
        return json.load(file)


def save_history(history):
    os.makedirs(BASE_PATH, exist_ok=True)
    with open(os.path.join(BASE_PATH, "history.json"), "w") as file:
        json.dump(history, file, indent=2)
    

def get_unprocessed_words_count(conn: Connection, timestamp: int):
    cursor = conn.cursor()
    cursor.execute("""
        select count(*) from LOOKUPS l
        where l.timestamp > ? and l.word_key like 'en:%'
    """, (timestamp,))
    return cursor.fetchone()[0]


def get_unprocessed_words_count_by_books(conn: Connection, timestamp: int):
    cursor = conn.cursor()
    cursor.execute("""
        select b.title, count(*) from lookups l
        join BOOK_INFO b on l.book_key = b.id
        where l.timestamp > ? and l.word_key like 'en:%'
        group by b.title 
        order by count(*) desc
    """, (timestamp,))
    return cursor.fetchall()


def query_db(conn: Connection, timestamp: int, limit: int) -> List[Lookup]:
    click.echo(f"Retrieving {limit} words from database")
    cursor = conn.cursor()
    cursor.execute("""
        select w.word, l.usage, b.title, l.timestamp from LOOKUPS l
        left join BOOK_INFO b on l.book_key = b.id
        left join WORDS w on l.word_key = w.id
        where l.timestamp > ? and l.word_key like 'en:%'
        order by l.timestamp
        limit ?
    """, (timestamp, limit))
    return [Lookup(row[0], row[1], row[2], row[3]) for row in cursor.fetchall()]


def query_gpt(lookups: List[Lookup]) -> List[Translation]:
    click.echo("Querying ChatGPT for translations")
    lines = []
    for word in lookups:
        lines.append(f"{word.word} :: {word.usage}")
    prompt = "{}\n{}".format(PROMPT, "\n".join(lines))
    completion = openai.ChatCompletion.create(
        model=MODEL_ID, 
        messages=[{"role": "user", "content": prompt},
    ])
    response = completion.choices[0].message.content
    translations = json.loads(response)
    result = []
    for word, tr in zip(lookups, translations):
        result.append(Translation(
            tr["word"], 
            tr["translation"], 
            tr["transcription"], 
            tr["usage"], 
            word.book, 
            word.timestamp
        ))
    return result


def dump(words: List[Translation], prefix = ""):
    ts = datetime.datetime.now()
    filename = f"{prefix}_{ts.isoformat()}.txt"
    with open(filename, "w") as file:
        for word in words:
            word.usage
            file.write(f"{word.word.lower()} [{word.transcription}] <br><br>{word.usage}<br><br>{word.book}|{word.translation}\n")
    return filename


if __name__ == '__main__':
    vocaby()
