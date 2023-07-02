import click, sqlite3, openai, openai.error
import os, time, json, itertools, datetime, textwrap
from glob import glob
from functools import cached_property
from typing import List, Dict, Optional
from contextlib import contextmanager
from dataclasses import dataclass, replace
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
Translate given words to russian.
The words will be sent in lines, each containing the word that needs to be translated and a sentence in which it was used.
The word and the sentence are separeted by ::.
Answer in valid JSON.
In your response you should include the original word in a normalized form (name the field "word").
In your response you should include the translation of the original word (name the field "translation").
In your response you should include the phonetical transcription **without** syllables of the original word (name the field "transcription").
In your response you should include the original sentence in which the word was used (name the field "usage").
Do not translate the usage sentence into russian.
For the usage field you should highlight the original word in bold using html tag <b>.
Only highlight the original word in the sentence (the original word is located on the left side of :: symbol of the given input).
All lines contain separate examples, so do not mix the meanings across different lines.
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

    def to_history_record(self):
        return f"{self.translation} :: {self.usage}"


@dataclass
class Processed:
    confirmed: List[Translation]
    rejected: List[Translation]
    max_timestamp: int

    @cached_property
    def combined(self):
        return list(itertools.chain(self.confirmed, self.rejected))


class History:
    def __init__(self, words: Dict[str, List[str]]):
        self.words = words
        self.new_words: List[Translation] = []
        self.ps = PorterStemmer()

    def __contains__(self, word: Translation):
        return self.words.get(self.ps.stem(word.word))
    
    def __getitem__(self, word: Translation):
        stem = self.ps.stem(word.word)
        if stem in self.words:
            return self.words[stem]
        else: 
            raise KeyError(stem)
    
    def add(self, word: Translation):
        self.words.setdefault(self.ps.stem(word.word), []).append(word.to_history_record())
        self.new_words.append(word)

    def undo_add(self):
        word = self.new_words[-1]
        self.new_words = self.new_words[:-1]
        self.words[self.ps.stem(word.word)].remove(word.to_history_record())

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
        history = History.load()
        timestamp = load_timestamp()
        print_stats(timestamp, conn)
        processed = process_words(query_db(conn, timestamp, limit), history)
        filename = dump(processed.confirmed, PREFIX)
        click.echo(f"Saved new words to {filename}")
        if no_checkpoint:
            click.echo("Skipped updating checkpoints")
        else:
            click.echo("Updated checkpoints")
            save_timestamp(processed.max_timestamp)
            history.save()
            print_stats(processed.max_timestamp, conn)


@vocaby.command()
@click.option("-t", "--timestamp", default=-1, type=int)
@click.argument("filename", default="vocab.db")
def stats(timestamp, filename):
    """Show accumulated statistics."""
    with db_resource(filename) as conn:
        if timestamp == -1:
            timestamp = load_timestamp()
        print_stats(timestamp, conn)


@vocaby.command()
@click.option("--delete", is_flag=True, default=False)
def merge(delete):
    """Merge Anki flashcards into a single batch."""
    if os.path.exists(f"{PREFIX}.txt"):
        if not click.confirm(f"{PREFIX}.txt already exists, would you like to proceed?", default=False):
            return click.echo("Aborted")
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
    def inner(words: List[Lookup], actor: UserPropmtActor):
        translations = query_gpt(words)
        max_timestamp = get_max_timestamp(translations)
        filtered = filter_words(translations, history)
        actor.process(filtered)
        click.clear()
        if up_for_retries := actor.pop_retries():
            click.echo(f"Retrying words: {[tr.word for tr in up_for_retries]}")
            return inner(up_for_retries, actor)
        return Processed(actor._confirmed, actor._rejected, max_timestamp)
    
    actor = UserPropmtActor(history)
    return inner(words, actor)


def filter_words(words: List[Translation], history: History) -> List[Translation]:
    result = []
    for word in words:
        if word in history and any([h.startswith(word.translation) for h in history[word]]):
            continue
        result.append(word)
    click.echo(f"Filtered {len(words) - len(result)} words, as already translated")
    return result



class UserPropmtActor:
    """
    Interact with a User regarding translation's quality.
    """
    def __init__(self, history: History):
        self._confirmed = []
        self._rejected = []
        self._up_for_retry = []
        self._actions = []
        self.history = history

    def process(self, words: List[Translation], idx: int = 0):
        length = len(words)
        while idx < length:
            word = words[idx]
            action = self.prompt_user(idx, length, word)
            match action:
                case "y":
                    self._actions.append(action)
                    self.add_for_learning(word)
                    idx += 1
                case "h":
                    self._actions.append(action)
                    self.add_for_history(word)
                    idx += 1
                case "n":
                    self._actions.append(action)
                    self.reject(word)
                    idx += 1
                case "r":
                    self._actions.append(action)
                    self.add_for_retry(word)
                    idx += 1
                case "e":
                    self._actions.append(action)
                    self.edit_translation(word)
                    idx += 1
                case "u":
                    self.undo()
                    idx -= 1
    
    def prompt_user(self, idx: int, length: int, word: Translation):
        """Ask user if the translation is good enough. Either save the translation, reject it or retry with LLM."""
        click.clear()
        click.echo(f"{idx + 1}/{length}\n")
        click.echo(f"{word.word} [{word.transcription}] = {word.translation}\n")
        click.echo(textwrap.fill(word.usage, width=80))
        click.echo("")
        if word in self.history:
            click.echo("#"*80)
            click.echo("Similar word was already processed!\n\n".upper())
            numbered = map(lambda i: f'{i[0]+1}. {i[1]}', enumerate(self.history[word]))
            wrapped = map(lambda x: textwrap.fill(x, width=80), numbered)
            click.echo("\n---\n".join(wrapped))
            click.echo("#"*80 + "\n")
        click.echo(f"{word.book}")
        click.echo("---\n")
        text = "Save this translation?\n(y)es, (n)o, (r)etry, (h)istory, (e)dit, (u)ndo"
        return click.prompt(
            text=text, 
            default="y", 
            type=click.Choice(["y", "n", "r", "h", "e", "u"], case_sensitive=False), 
            show_choices=False
        )

    def undo(self):
        match self._actions[-1]:
            case "y":
                self.undo_add_for_learning()
            case "h":
                self.undo_add_for_history()
            case "n":
                self.undo_reject()
            case "r":
                self.undo_add_for_retry()
            case "e":
                self.undo_edit_translation()
        self._actions = self._actions[:-1]

    def add_for_learning(self, word: Translation):
        self._confirmed.append(word)
        self.history.add(word)

    def undo_add_for_learning(self):
        self._confirmed = self._confirmed[:-1]
        self.history.undo_add()
        self._actions = self._actions[:-1]

    def add_for_history(self, word: Translation):
        self.history.add(word)

    def undo_add_for_history(self):
        self.history.undo_add()
        self._actions = self._actions[:-1]

    def add_for_retry(self, word: Translation):
        self._up_for_retry.append(word)
    
    def undo_add_for_retry(self):
        self._up_for_retry = self._up_for_retry[:-1]
        self._actions = self._actions[:-1]

    def reject(self, word: Translation):
        self._rejected.append(word)
    
    def undo_reject(self):
        self._rejected = self._rejected[:-1]
        self._actions = self._actions[:-1]

    def edit_translation(self, word: Translation):
        result = click.prompt("What's the best translation?")
        new_word = replace(word, translation=result)
        self.add_for_learning(new_word)

    def undo_edit_translation(self):
        self.undo_add_for_learning()
        self._actions = self._actions[:-1]
    
    def pop_retries(self):
        up_for_retry = self._up_for_retry
        self._up_for_retry = []
        return up_for_retry
    

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
    if not os.path.exists(os.path.join(BASE_PATH, "checkpoint.json")):
        return 0
    with open(os.path.join(BASE_PATH, "checkpoint.json")) as file:
        return int(json.load(file).get("timestamp", 0))


def save_timestamp(timestamp: int):
    """Save latest processed timestamp."""
    os.makedirs(BASE_PATH, exist_ok=True)
    with open(os.path.join(BASE_PATH, "checkpoint.json"), "w") as file:
        json.dump({"timestamp": timestamp}, file)


def get_max_timestamp(words: List[Translation]) -> Optional[int]:
    if len(words):
        return max(map(lambda x: x.timestamp, words))
    

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
            return query_gpt(lookups, retry-1)
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
            tr["transcription"].replace(".", ""),
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
            file.write(f"{word.word} [{word.transcription}] <br><br>{word.usage}<br><br>{word.book}|{word.translation}\n")
    return filename


if __name__ == '__main__':
    vocaby()
