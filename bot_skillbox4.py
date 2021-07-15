import random
import nltk
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split
import logging
from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from datetime import datetime
import settings

# с большим словарем из json
# и плюс машинное обучение
# и подключаем к telegram

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO, filename='bot_sb.log'
)

logger = logging.getLogger(__name__)

with open('BOT_CONFIG_13072021_2.json', 'r', encoding='utf-8') as f:
    BOT_CONFIG = json.load(f)


def clean(text):
    text = text.lower()
    cleaned_text = ''
    for ch in text:
        if ch in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя':
            cleaned_text = cleaned_text + ch
    return cleaned_text


def get_intent(text):
    for intent in BOT_CONFIG['intents'].keys():
        for example in BOT_CONFIG['intents'][intent]['examples']:
            cleaned_example = clean(example)
            cleaned_text = clean(text)
            if nltk.edit_distance(cleaned_example, cleaned_text) / \
                    max(len(cleaned_example), len(cleaned_text)) * 100 < 40:
                return intent
    return 'unknown_intent'


def get_intent_by_model(text):
    vectorized_text = vectorizer.transform([text])
    return clf.predict(vectorized_text)[0]


def bot(text):
    # intent = get_intent(text)
    intent = get_intent_by_model(text)
    if intent == 'unknown_intent':
        return random.choice(BOT_CONFIG['default'])
    else:
        return random.choice(BOT_CONFIG['intents'][intent]['responses'])


# Обучение модели
print('Начинаем обучение')
X = []
y = []
for intent in BOT_CONFIG['intents']:
    for example in BOT_CONFIG['intents'][intent]['examples']:
        X.append(example)
        y.append(intent)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Векторизация ', datetime.now())
vectorizer = CountVectorizer(analyzer='char', ngram_range=(1,3), preprocessor=clean)
# vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,3))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print('Классификация ', datetime.now())
clf = LogisticRegression()
# clf = RidgeClassifier()
clf.fit(X_train_vectorized, y_train)

print(clf.score(X_train_vectorized, y_train), clf.score(X_test_vectorized, y_test))
print('Классификация закончена ', datetime.now())

# Тестирование бота
# while True:
#     input_text = input()
#     if input_text == '':
#         break
#     response = bot(input_text)
#     print(response)
#    if response in BOT_CONFIG['intents']['bye']['responses']:
#        break


# Define a few command handlers. These usually take the two arguments update and
# context.
def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    logging.info('User {} press /start'.format(update.message.chat.username))
    update.message.reply_markdown_v2(
        fr'Hi {user.mention_markdown_v2()}\!',
        reply_markup=ForceReply(selective=True),
    )


def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def chat(update: Update, context: CallbackContext) -> None:
    """Echo the user message."""
    input_text = update.message.text
    logging.info(input_text)
    response = bot(clean(input_text))
    update.message.reply_text(response)
    logging.info(response)


def main() -> None:
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater(settings.TOKEN_TELEGRAMM)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, chat))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    logging.info('Bot started!')
    main()
