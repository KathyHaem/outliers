langs_tatoeba = ['ara', 'heb', 'vie', 'ind', 'jav', 'tgl', 'eus', 'mal',
                 'tel', 'afr', 'nld', 'deu', 'ell', 'ben', 'hin', 'mar',
                 'urd', 'tam', 'fra', 'ita', 'por', 'spa', 'bul', 'rus',
                 'jpn', 'kat', 'kor', 'tha', 'swh', 'cmn', 'kaz', 'tur',
                 'est', 'fin', 'hun', 'pes']

langs_tatoeba_2 = ['ar', 'he', 'vi', 'id', 'jv', 'tl', 'eu', 'ml', 'te', 'af',
                   'nl', 'de', 'el', 'bn', 'hi', 'mr', 'ur', 'ta', 'fr', 'it',
                   'pt', 'es', 'bg', 'ru', 'ja', 'ka', 'ko', 'th', 'sw', 'zh',
                   'kk', 'tr', 'et', 'fi', 'hu', 'fa']

langs_wiki = ['ara', 'eng', 'spa', 'sun', 'swh', 'tur']

lang_dict = {'ar': 'ara', 'he': 'heb', 'vi': 'vie', 'id': 'ind',
             'jv': 'jav', 'tl': 'tgl', 'eu': 'eus', 'ml': 'mal',
             'te': 'tel', 'af': 'afr', 'nl': 'nld', 'de': 'deu', 'en': 'eng',
             'el': 'ell', 'bn': 'ben', 'hi': 'hin', 'mr': 'mar', 'ur': 'urd',
             'ta': 'tam', 'fr': 'fra', 'it': 'ita', 'pt': 'por', 'es': 'spa',
             'bg': 'bul', 'ru': 'rus', 'ja': 'jpn', 'ka': 'kat', 'ko': 'kor',
             'th': 'tha', 'sw': 'swh', 'zh': 'cmn', 'kk': 'kaz', 'tr': 'tur',
             'et': 'est', 'fi': 'fin', 'hu': 'hun', 'fa': 'pes', 'su': 'sun'}

lang_dict_3_2 = dict((v, k) for k, v in lang_dict.items())

sts_tracks = [
    'track2-ar-en', 'track4a-es-en', 'track4b-es-en', 'track6-tr-en']

sts_gold_files = {
    'track2-ar-en':  '../data/sts/STS.gs.track2.ar-en.txt',
    'track4a-es-en': '../data/sts/STS.gs.track4a.es-en.txt',
    'track4b-es-en': '../data/sts/STS.gs.track4b.es-en.txt',
    'track6-tr-en':  '../data/sts/STS.gs.track6.tr-en.txt'
}
