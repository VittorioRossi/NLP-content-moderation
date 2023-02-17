import re 
from sklearn.preprocessing import LabelBinarizer

def clean_text(x:str):
  x = re.sub('[^A-Za-z0-9@#]+', ' ', x)
  return x.lower().strip()

def replace_numbers(x:str):
  return re.sub('\d', "<number>", x)

def is_in_glove(word:str, vocab):
  return word in vocab

def replace_url(string, replace_with = "<url>"):
  regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
  return re.sub(regex, replace_with, string).strip().lower()

def replace_not_in_glove(phrase:str):
  repl = "<unknown>"
  phrase = phrase
  for word in phrase.split(" "):
    if not is_in_glove(word):
      phrase = phrase.replace(word, repl)
  return phrase

def replace_users(phrase:str):
  return re.sub("\s@[A-Za-z0-9]+", " <user>", phrase)

def replace_hashtags(phrase:str):
  repl = " <hashtag>"
  return re.sub("\s#[A-Za-z0-9]+", repl, phrase)


def preprocess_text(df):
  df.text = df.text.apply(replace_users)
  df.text = df.text.apply(replace_hashtags)
  df.text = df.text.apply(replace_url)
  df.text = df.text.apply(clean_text)
  df.text = df.text.apply(replace_numbers)
  df.text = df.text.apply(replace_not_in_glove)

  df = df[df.text.apply(len) < 500+1]
  X = df.text
  y = LabelBinarizer().fit_transform(df.label)

  return df.text, df.label