import os
import abc
import random

import telegram
import uuid
import yaml
import openai
import sqlite3
import logging
import subprocess

from datetime import datetime

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes, ConversationHandler, MessageHandler, filters,
)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

with open('settings.yml') as s:
    settings = yaml.load(s, Loader=yaml.SafeLoader)

TELEGRAM_TOKEN = settings['telegram_token']
CHAT_DIR = settings['chat_dir']
SCRIPT_RUN = f"./{settings['script_name']}"
DATABASE_PATH = settings['database_path']
openai.api_key = settings['openai_token']

# HANDLERS
(
    REC_TYPE_CHOICE_HANDLER,
    HISTORY_HANDLER,
    DISLIKED_HANDLER,
    INFERENCE,
    REALITY,
    PRECISION,
    USEFULNESS,
    FEEDBACK
) = range(8)

MESSAGES_DICTIONARY = {
    'en': {
        'ask_content_type': 'What type of content you want GPT to recommend?',
        'ask_content_keyboard': [['Books'], ['Movies'], ['Music']],
        'ask_history': "What {content_type} have you {verb} in the past?\n"
                       "Enter titles separating by comma",
        'ask_disliked': "What {content_type} did you dislike?\n"
                        "Enter titles separating by comma",
        'inf_got_input': 'Got your input, please wait for ~5 minutes',
        'inf_waiting': 'Waiting GPT to init',
        'inf_rec_gen': 'Generating your recommendation',
        'reality': 'Please rate how, in your opinion, the model predicted real-life objects '
                   'and gave them an accurate description?',
        'precision': 'Please rate how the recommendation fits your interests',
        'usefulness': 'Please rate how useful such recommendations are for you '
                      'and would you like to use such a service in the future?',
        'feedback': 'Please give us feedback',
        'thank': 'Thank you for feedback!',
        'save': 'Saving results',
        'finish': 'Thank you! Question ID - {question_id}. '
                  'Press /start if you want to get recommendations again'
    }
}


def get_translation(context, msg_type):
    return MESSAGES_DICTIONARY[context.user_data['language']][msg_type]

# SYSTEM METHODS


def init_db():
    logger.info('Initializing database')

    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()

    cur.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            testing_group_id INTEGER,
            question_id TEXT,
            content_type TEXT,
            history TEXT,
            disliked TEXT,
            prompt TEXT,
            output TEXT,
            reality INTEGER,
            precision INTEGER,
            usefulness INTEGER,
            feedback TEXT,
            time_start INTEGER,
            time_end INTEGER
        )
        ''')

    cur.execute('''
        CREATE TABLE IF NOT EXISTS experiment_groups (
            user_id INTEGER PRIMARY KEY,
            testing_group_id INTEGER,
            model_group_id INTEGER
        )
        ''')

    conn.commit()
    conn.close()

    logger.info('Database initialized successfully')


def log_to_db(user_data):
    logger.info('Connecting to database')
    conn = sqlite3.connect(DATABASE_PATH)
    logger.info('Creating cursor')
    cur = conn.cursor()

    data = [
        (
            user_data['user_id'],
            user_data['testing_group_id'],
            user_data['question_id'],
            user_data['content_type'],
            user_data['history'],
            user_data.get('disliked', ''),
            user_data['prompt'],
            user_data['output'],
            user_data['reality'],
            user_data['precision'],
            user_data['usefulness'],
            user_data['feedback'],
            user_data['time_start'],
            user_data['time_end']
        )
    ]

    logger.info('Inserting data')
    cur.executemany(f'''
        INSERT INTO logs VALUES (NULL, {"?, " * 13}?)
        ''', data)

    logger.info('Committing changes')
    conn.commit()

    logger.info('Closing connection to database')
    conn.close()


def can_execute(user_data):
    user_id = user_data['user_id']

    query = f"""SELECT COUNT(*) FROM logs 
WHERE user_id = {user_id}
AND datetime(time_start, 'unixepoch') >= datetime('now', '-1 hour');"""

    logger.info('Connecting to database')
    conn = sqlite3.connect(DATABASE_PATH)
    logger.info('Creating cursor')
    cur = conn.cursor()

    res = cur.execute(query)
    queries_within_hour = res.fetchone()[0]

    conn.close()

    logger.info(f'{user_id}: queries within hour: {queries_within_hour}')
    return queries_within_hour < 10


def get_user_groups(user_data):
    user_id = user_data['user_id']
    query = f"""SELECT testing_group_id, model_group_id FROM experiment_groups
WHERE user_id = {user_id};"""

    logger.info('Connecting to database')
    conn = sqlite3.connect(DATABASE_PATH)
    logger.info('Creating cursor')
    cur = conn.cursor()

    res = cur.execute(query).fetchone()

    if not res:
        logger.info('Inserting groups to database')
        query = f"""INSERT INTO experiment_groups VALUES (?, ?, ?);"""
        testing_group_id = random.randint(0, 2)
        model_group_id = 0
        cur.execute(query, (user_id, testing_group_id, model_group_id))

        conn.commit()
    else:
        testing_group_id, model_group_id = res

    conn.close()
    return testing_group_id, model_group_id


def generate_prompt(user_data):
    logger.info('Generating prompt')

    testing_group_id = user_data["testing_group_id"]
    content_type = user_data["content_type"].lower()
    verb = user_data["content_type_verb"]

    if testing_group_id == 0:
        prompt = f"""\
Imagine that you are a recommender system that encourages people \
to collect {content_type} that they like by analyzing the interests of users. \
You must give an advice to the user. \
You know the user {verb} {content_type}: {user_data["history"]}. \
Please make a list with 5 {content_type} recommendations that users have not yet watched, \
the user's perceived interests."""
    elif testing_group_id == 1:
        prompt = f"""\
Imagine that you are a recommender system that encourages people \
to collect {content_type} that they like by analyzing the interests of users. \
You must give an advice to the user. \
You know the user {verb} {content_type}: {user_data["history"]}. \
And you also know the user disliked {content_type}: {user_data["disliked"]} \
Please make a list with 5 {content_type} recommendations that users have not yet watched, \
the user's perceived interests."""
    else:
        prompt = f"""\
Imagine that you are a recommender system that encourages people \
to collect {content_type} thatF they like by analyzing the interests of users. \
You must give an advice to the user. \
You know the user {verb} {content_type}: {user_data["history"]}. \
And you also know the user disliked {content_type}: {user_data["disliked"]} \
Please make a list with 5 {content_type} recommendations that users have not yet watched, \
the user's perceived interests. Explain your recommendation."""

    user_data["prompt"] = prompt

    return prompt


# BOT METHODS

async def help_(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
):
    await update.message.reply_text(
        "You can use the following commands:\n\
        /start - Start the bot\n\
        /help - Show this message\n"
    )


async def choice_rec_type(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
):
    user_id = update.message.from_user.id

    context.user_data["user_id"] = int(user_id)
    context.user_data["question_id"] = str(uuid.uuid4())

    testing_group_id, model_group_id = get_user_groups(context.user_data)

    context.user_data["testing_group_id"] = testing_group_id
    context.user_data["model_group_id"] = model_group_id
    context.user_data["time_start"] = int(datetime.now().timestamp())

    if not can_execute(context.user_data):
        return await query_limit(update, context)

    await update.message.reply_text(
        get_translation(context, 'ask_content_type'),
        reply_markup=telegram.ReplyKeyboardMarkup(
            keyboard=get_translation(context, 'ask_content_keyboard'),
            one_time_keyboard=True
        )
    )

    return REC_TYPE_CHOICE_HANDLER


async def query_limit(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
):
    logger.info('Query limit is reached')
    await update.message.reply_text(
        "You may make no more than 4 queries within an hour",
    )

    return ConversationHandler.END


async def choice_rec_type_handler(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
):
    content_type = update.message.text
    context.user_data["content_type"] = content_type
    context.user_data["content_type_msg"] = content_type

    if content_type == 'Movies':
        verb = 'watched'
    elif content_type == 'Books':
        verb = 'read'
    else:
        verb = 'listened to'

    verb_msg = verb

    context.user_data["content_type_verb"] = verb
    context.user_data["content_type_verb_msg"] = verb_msg

    return await ask_history(update, context)


async def ask_history(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
):
    content_type = context.user_data["content_type"]

    logger.info('Asking history')
    await update.message.reply_text(
        get_translation(context, 'ask_history').format(
            content_type=content_type.lower(),
            verb=context.user_data["content_type_verb_msg"]
        )
    )

    return HISTORY_HANDLER


async def ask_history_handler(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
):
    history = update.message.text
    context.user_data["history"] = history

    if context.user_data["testing_group_id"] > 0:
        return await ask_disliked(update, context)
    else:
        return await inference(update, context)


async def ask_disliked(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
):

    content_type = context.user_data["content_type"]

    logger.info('Asking disliked')

    await update.message.reply_text(
        get_translation(context, 'ask_disliked').format(
            content_type=content_type.lower()
        )
    )

    return DISLIKED_HANDLER


async def ask_disliked_handler(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
):
    disliked = update.message.text
    context.user_data["disliked"] = disliked

    return await inference(update, context)


class ChatModel(abc.ABC):

    @abc.abstractmethod
    async def ask_model(self, prompt) -> str:
        pass


class Gpt4All(ChatModel):

    @staticmethod
    def wait_for_init():
        logger.info('Staring GPT process')
        process = subprocess.Popen(
            [SCRIPT_RUN],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        logger.info('Finished staring GPT process')

        # Loop for waiting the request
        logger.info('Running GPT init loop')
        while True:
            output = process.stdout.readline()
            print(output)

            if output == b'\x1b[33m\n':
                break
        logger.info('Finished GPT init loop')

        return process

    async def write_to_ai(self, prompt):
        logger.info('Sending input to GPT')
        self.process.stdin.write(bytes(prompt, encoding='utf8'))
        self.process.stdin.flush()
        logger.info('Finished sending input to GPT')

    def __init__(self):
        self.process = self.wait_for_init()

    async def get_ai_output(self) -> str:
        logger.info('Starting getting GPT output')
        outputs = []
        while True:
            output = self.process.stdout.readline()

            outputs.append(output.decode())

            if output.endswith(b'\x1b[0m\n'):
                break

        logger.info('Finished starting getting GPT output')

        return " ".join(outputs)

    async def ask_model(self, prompt) -> str:
        await self.write_to_ai(prompt)

        return await self.get_ai_output()


class OpenAi(ChatModel):
    def __init__(self, model='gpt-3.5-turbo'):
        self.model = model

    async def ask_model(self, prompt) -> str:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
            ]
        )

        return completion.choices[0].message.content


def get_model_by_group(user_data):
    model_group_id = user_data["model_group_id"]

    return OpenAi(model='gpt-3.5-turbo')


async def inference(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
):

    # get the command from the user's message
    await update.message.reply_text(get_translation(context, 'inf_got_input'))

    # send the command to the AI
    await update.message.reply_text(get_translation(context, 'inf_waiting'))
    chat_model = get_model_by_group(context.user_data)

    # read the output from the AI
    await update.message.reply_text(get_translation(context, 'inf_rec_gen'))
    output = await chat_model.ask_model(generate_prompt(context.user_data))

    context.user_data["output"] = output

    # send the output to the user
    await update.message.reply_text(output)

    return await ask_reality(update, context)


async def ask_reality(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
):
    await update.message.reply_text(
        get_translation(context, 'reality'),
        reply_markup=telegram.ReplyKeyboardMarkup(
            keyboard=[['1', '2', '3', '4', '5']],
            one_time_keyboard=True
        )
    )
    return REALITY


async def ask_reality_handler(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
):
    reality = update.message.text
    context.user_data["reality"] = reality

    return await ask_precision(update, context)


async def ask_precision(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
):
    await update.message.reply_text(
        get_translation(context, 'precision'),
        reply_markup=telegram.ReplyKeyboardMarkup(
            keyboard=[['1', '2', '3', '4', '5']],
            one_time_keyboard=True
        )
    )
    return PRECISION


async def ask_precision_handler(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
):
    precision = update.message.text
    context.user_data["precision"] = precision

    return await ask_usefulness(update, context)


async def ask_usefulness(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
):
    await update.message.reply_text(
        get_translation(context, 'usefulness'),
        reply_markup=telegram.ReplyKeyboardMarkup(
            keyboard=[['1', '2', '3', '4', '5']],
            one_time_keyboard=True
        )
    )
    return USEFULNESS


async def ask_usefulness_handler(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
):
    usefulness = update.message.text
    context.user_data["usefulness"] = usefulness

    await update.message.reply_text(
        get_translation(context, 'feedback')
    )

    return FEEDBACK


async def ask_feedback(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
):
    feedback = update.message.text
    context.user_data["feedback"] = feedback

    logger.info('Asking for feedback')

    await update.message.reply_text(
        get_translation(context, 'thank')
    )

    return await save_user_date(update, context)


async def save_user_date(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
):
    context.user_data["time_end"] = int(datetime.now().timestamp())
    await update.message.reply_text(
        get_translation(context, 'save')
    )

    log_to_db(context.user_data)

    await update.message.reply_text(
        get_translation(context, 'finish').format(
            question_id=context.user_data["question_id"]
        )
    )
    return ConversationHandler.END


app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

conv_handler = ConversationHandler(
    entry_points=[CommandHandler("start", choice_rec_type)],
    states={
        REC_TYPE_CHOICE_HANDLER: [
            MessageHandler(
                filters.TEXT, choice_rec_type_handler
            )
        ],
        HISTORY_HANDLER: [
            MessageHandler(
                filters.TEXT, ask_history_handler
            )
        ],
        DISLIKED_HANDLER: [
            MessageHandler(
                filters.TEXT, ask_disliked_handler
            )
        ],
        INFERENCE: [
            MessageHandler(
                filters.TEXT, inference
            )
        ],
        PRECISION: [
            MessageHandler(
                filters.Regex(r'^[012345]$'), ask_precision_handler
            )
        ],
        REALITY: [
            MessageHandler(
                filters.Regex(r'^[012345]$'), ask_reality_handler
            )
        ],
        USEFULNESS: [
            MessageHandler(
                filters.Regex(r'^[012345]$'), ask_usefulness_handler
            )
        ],
        FEEDBACK: [
            MessageHandler(
                filters.TEXT, ask_feedback
            )
        ]
    },
    fallbacks=[],
)

app.add_handler(CommandHandler("help", help_))
app.add_handler(conv_handler)

init_db()

os.chdir(CHAT_DIR)
app.run_polling()
