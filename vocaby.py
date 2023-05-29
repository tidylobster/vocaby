
import click, sqlite3, openai, openai.error
import os, time, json, itertools, datetime, textwrap
from glob import glob
from typing import List
from contextlib import contextmanager
from dataclasses import dataclass
from sqlite3 import Connection
from nltk.stem import PorterStemmer
from tabulate import tabulate


with open("config.json", "r") as file:
    config = json.load(file)
openai.api_key = config["token"]


BASE_PATH = ".vocaby/"
PREFIX = "anki"
MODEL_ID = "gpt-3.5-turbo"
PROMPT = """
I need you to translate some words to russian for me. I will send you lines, 
each containing the word that needs to be translated and a sentence in which 
it was used. The word and the sentence are separated by ::. Answer in JSON, 
where you should include the original normalized word (name the field 
"word"), translation of the word (name the field "translation"), phonetical 
transcription of the word (name the field "transcription" and do not separate the 
syllabels), and the sentence (name the field "usage") in which it was used. 
For the usage field you should highlight the original word in bold using html tag 
<b>. Only highlight one requested word in the original text (the one on the left 
of the :: separator). Do not translate the "usage" sentence. Respond with a JSON 
list, where you should include all of the translations.
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


@dataclass
class Processed:
    confirmed: List[Translation]
    rejected: List[Translation]


class History:
    def __init__(self, words: dict):
        self.words = words
        self.ps = PorterStemmer()

    def __contains__(self, word: Translation):
        return self.ps.stem(word.word) in self.words
    
    def __getitem__(self, word: Translation):
        stem = self.ps.stem(word.word)
        if stem in self.words:
            return self.words[stem]
        else: 
            raise KeyError(stem)

    def add(self, word: Translation):
        self.words.setdefault(self.ps.stem(word.word), []).append(f"{word.translation} :: {word.usage}")

    @classmethod
    def load(cls):
        if not os.path.exists(os.path.join(BASE_PATH, "history.json")):
            return cls({})
        with open(os.path.join(BASE_PATH, "history.json")) as file:
            return cls(json.load(file))

    def save(self):
        os.makedirs(BASE_PATH, exist_ok=True)
        with open(os.path.join(BASE_PATH, "history.json"), "w") as file:
            json.dump(dict(sorted(self.words.items(), key=lambda x: x[0])), file, indent=2)

            
@contextmanager
def db_resource(filename):
    """Acquire and release Sqlite3 connection."""
    conn = sqlite3.connect(filename)
    try:
        yield conn
    finally:
        conn.close()


@click.group()
def vocaby():
    pass


@vocaby.command()
@click.option("--limit", default=20, help="Number of lookup words to scan")
@click.option("--no-checkpoint","--no-checkpoints", is_flag=True, default=False, help="Flag to leave checkpoints unupdated.")
@click.argument("filename", default="vocab.db")
def translate(limit, no_checkpoint, filename):
    """Process vocab.db file, translate lookup words and transform them into Anki flashcards."""
    with db_resource(filename) as conn:
        timestamp = load_timestamp()
        history = History.load()
        print_stats(timestamp, conn)
        lookups = query_db(conn, timestamp, limit)
        processed = process_words(lookups, history)
        combined = list(itertools.chain(processed.confirmed, processed.rejected))
        match len(combined):
            case 0:
                return
            case 1:
                timestamp = combined[0]
            case _:
                timestamp = max(map(lambda x: x.timestamp, combined))
        filename = dump(processed.confirmed, PREFIX)
        click.echo(f"Saved new words to {filename}")
        if no_checkpoint:
            click.echo("Skipped updating checkpoints")
        else:
            click.echo("Updated checkpoints")
            save_timestamp(timestamp)
            history.save()
            num_unprocessed = get_unprocessed_words_count(conn, timestamp)
            click.echo(f"Words left unprocessed in the database: {num_unprocessed}")


@vocaby.command()
@click.argument("filename", default="vocab.db")
def stats(filename):
    """Show accumulated statistics."""
    with db_resource(filename) as conn:
        timestamp = load_timestamp()
        print_stats(timestamp, conn)


@vocaby.command()
@click.option("--delete", is_flag=True, default=False)
def merge(delete):
    """Merge Anki flashcards into a single batch."""
    lines = []
    for filename in glob(f"{PREFIX}_*.txt"):
        with open(filename, "r") as file:
            lines.extend(file.readlines())
    with open(f"{PREFIX}.txt", "a") as file:
        file.writelines(lines)
    if delete:
        for filename in glob(f"{PREFIX}_*.txt"):
            os.remove(filename)


def process_words(words: List[Lookup], history: History) -> Processed:
    """Process all words using LLM. Retry the prompt if requested."""
    def inner(words: List[Lookup], confirmed: List[Translation], rejected: List[Translation]):
        up_for_retry: List[Translation] = []
        for i, tr in enumerate(query_gpt(words)):
            match prompt_user(i, len(words), tr, history):
                case "y": 
                    confirmed.append(tr)
                    history.add(tr)
                case "h":
                    history.add(tr)
                case "n":
                    rejected.append(tr)
                case "r":
                    up_for_retry.append(tr)
        click.clear()
        if up_for_retry:
            click.echo(f"Retrying requested words: {[tr.word for tr in up_for_retry]}")
            return inner(up_for_retry, confirmed, rejected)
        return Processed(confirmed, rejected)
    return inner(words, [], [])


def prompt_user(i: int, limit: int, tr: Translation, history: History):
    """Ask user if the translation is good enough. Either save the translation, reject it or retry with LLM."""
    click.clear()
    click.echo(f"{i + 1}/{limit}\n")
    click.echo(f"{tr.word} [{tr.transcription}] = {tr.translation}\n")
    if tr in history:
        click.echo("#"*80)
        click.echo("Similar word was already processed!\n\n".upper())
        numbered = map(lambda i: f'{i[0]+1}. {i[1]}', enumerate(history[tr]))
        wrapped = map(lambda x: textwrap.fill(x, width=80), numbered)
        click.echo("\n---\n".join(wrapped))
        click.echo("#"*80 + "\n")
    click.echo(textwrap.fill(tr.usage, width=80))
    click.echo("")
    click.echo(f"{tr.book}")
    click.echo("---\n")
    text = "Save this translation?\n(y)es, (n)o, (r)etry, (h)istory"
    return click.prompt(
        text=text, 
        default="y", 
        type=click.Choice(["y", "n", "r", "h"], case_sensitive=False), 
        show_choices=False
    )
    

def print_stats(timestamp, conn):
    """Print accumulated statistics for user."""
    click.echo(f"Latest scanned timestamp: {datetime.datetime.fromtimestamp(timestamp // 1000)}")
    num_unprocessed = get_unprocessed_words_count(conn, timestamp)
    by_book = get_unprocessed_words_count_by_books(conn, timestamp)
    if num_unprocessed == 0:
        return click.echo("No words left unprocessed")
    click.echo(f"Words left unprocessed in the database: {num_unprocessed}")
    click.echo(tabulate(by_book))


def load_timestamp() -> int:
    """Load latest processed timestamp."""
    if not os.path.exists(os.path.join(BASE_PATH, "checkoint.json")):
        return 0
    with open(os.path.join(BASE_PATH, "checkoint.json")) as file:
        return int(json.load(file).get("timestamp", 0))


def save_timestamp(timestamp: int):
    """Save latest processed timestamp."""
    os.makedirs(BASE_PATH, exist_ok=True)
    with open(os.path.join(BASE_PATH, "checkoint.json"), "w") as file:
        json.dump({"timestamp": timestamp}, file)
    

def get_unprocessed_words_count(conn: Connection, timestamp: int):
    """Get total number of unprocessed words."""
    cursor = conn.cursor()
    cursor.execute("""
        select count(*) from LOOKUPS l
        where l.timestamp > ? and l.word_key like 'en:%'
    """, (timestamp,))
    return cursor.fetchone()[0]


def get_unprocessed_words_count_by_books(conn: Connection, timestamp: int):
    """Get total number of unprocessed words grouped by books."""
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
    """Query database to retrieve the next batch of unprocessed words."""
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


def query_gpt(lookups: List[Lookup], retry: int = 3) -> List[Translation]:
    """Query LLM to retrieve word translations."""
    click.echo("Querying ChatGPT for translations")
    lines = []
    for word in lookups:
        lines.append(f"{word.word} :: {word.usage}")
    prompt = "{}\n{}".format(PROMPT, "\n".join(lines))
    try:
        completion = openai.ChatCompletion.create(
            model=MODEL_ID, 
            messages=[{"role": "user", "content": prompt},
        ])
    except openai.error.RateLimitError as e:
        if retry > 0:
            click.echo("Hit RateLimitError, waiting for retry")
            time.sleep(5)
            return query_gpt(lookups, )
        else:
            click.echo("Hit RateLimitError, giving up")
            click.echo(e)
            raise e
    response = completion.choices[0].message.content
    try:
        translations = json.loads(response)
    except json.JSONDecodeError as e:
        if retry > 0:
            click.echo("Invalid response from ChatGPT, retrying")
            return query_gpt(lookups, retry-1)
        else:
            click.echo("Invalid response from ChatGPT, giving up")
            click.echo(response)
            raise e
    result = []
    for word, tr in zip(lookups, translations):
        result.append(Translation(
            tr["word"].lower(), 
            tr["translation"].lower(), 
            tr["transcription"], 
            tr["usage"], 
            word.book, 
            word.timestamp
        ))
    return result


def dump(words: List[Translation], prefix = ""):
    """Dump words into Anki flashcard form."""
    ts = datetime.datetime.now()
    filename = f"{prefix}_{ts.isoformat()}.txt"
    with open(filename, "w") as file:
        for word in words:
            word.usage
            file.write(f"{word.word} [{word.transcription}] <br><br>{word.usage}<br><br>{word.book}|{word.translation}\n")
    return filename


if __name__ == '__main__':
    vocaby()
