import re
def process_text(text):
    text = text.lower()
    text = re.sub("dr\.",'dr', text)
    text = re.sub('m\.d\.', 'md', text)
    text = re.sub('a\.m\.','am', text)
    text = re.sub('p\.m\.','pm', text)
    text = re.sub("\d+\.\d+", 'floattoken', text)
    text = re.sub("\.{2,}", '.', text)
    text = re.sub('[^\w_|\.|\?|!]+', ' ', text)
    text = re.sub('\.', ' . ', text)
    text = re.sub('\?', ' ? ', text)
    text = re.sub('!', ' ! ', text)
    text = re.sub('\d{3,}', '', text)
    return text
