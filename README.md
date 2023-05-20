# Vocaby

Convert words from WordsDictionary from Kindle into Anki flashcards using ChatGPT.

## Prerequisites

1. Install local dependencies.
    ```sh
    pip install -r requirements.txt
    ```
1. [Obtain](https://platform.openai.com/docs/api-reference/authentication) OpenAI token and insert it into `vocab.py` file in `sk-...` placeholder.

## Usage

1. Download vocab.db from your Kindle.
1. Run CLI.
    ```sh
    python vocaby.py vocab.db
    ```
1. When prompted, confirm all the words that you would like to save for Anki.
1. Import anki_< timestamp >.txt file into your Anki library.
