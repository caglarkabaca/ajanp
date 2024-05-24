import re
from turkish.deasciifier import Deasciifier

def deasciifier(text):
    deasciifier = Deasciifier(text)
    return deasciifier.convert_to_turkish()

def remove_circumflex(text):
    circumflex_map = {
        'â': 'a',
        'î': 'i',
        'û': 'u',
        'ô': 'o',
        'Â': 'A',
        'Î': 'I',
        'Û': 'U',
        'Ô': 'O'
    }

    return ''.join(circumflex_map.get(c, c) for c in text)    
def turkish_lower(text):
    turkish_map = {
        'I': 'ı',
        'İ': 'i',
        'Ç': 'ç',
        'Ş': 'ş',
        'Ğ': 'ğ',
        'Ü': 'ü',
        'Ö': 'ö'
    }
    return ''.join(turkish_map.get(c, c).lower() for c in text)

def clean_text(text):
    # Metindeki şapkalı harfleri kaldırma
    # text = remove_circumflex(text)
    # Metni küçük harfe dönüştürme
    text = turkish_lower(text)
    # deasciifier
    # text = deasciifier(text)
    # Kullanıcı adlarını kaldırma
    text = re.sub(r"@\S*", " ", text)
    # Hashtag'leri kaldırma
    text = re.sub(r'#\S+', ' ', text)
    # URL'leri kaldırma
    text = re.sub(r"http\S+|www\S+|https\S+", ' ', text, flags=re.MULTILINE)
    # Noktalama işaretlerini ve metin tabanlı emojileri kaldırma
    # text = re.sub(r'[^\w\s]|(:\)|:\(|:D|:P|:o|:O|;\))', ' ', text)
    # Emojileri kaldırma
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r' ', text)

    # Birden fazla boşluğu tek boşlukla değiştirme
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == '__main__':

    import pandas as pd

    # df_test = pd.read_csv('turkish-offensive-language-detection/test.csv')
    # df_train = pd.read_csv('turkish-offensive-language-detection/train.csv')
    # df_val = pd.read_csv('turkish-offensive-language-detection/valid.csv')
    # df = pd.concat([df_train, df_test, df_val], ignore_index=True)


    df = pd.read_csv('./hate-speech-target/troff-v1.0.csv')
    df.drop('_', axis=1, inplace=True)
    df.drop('id', axis=1, inplace=True)
    df.drop('timestamp', axis=1, inplace=True)

    print(df.head())

    df['text'] = df['text'].apply(clean_text)
    df['label'] = df['label'].apply(lambda x: 0 if x == 'non' else 1)

    label_counts = df['label'].value_counts()
    print(label_counts)
    print(df['text'].count)

    print(df.head())

    df.dropna()
    df.to_csv('troff-v1.0_preprocessed.csv', index=False)