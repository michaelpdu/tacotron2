# coding:utf-8 
from xpinyin import Pinyin
import argparse

class PinyinHelper:
    def __init__(self):
        self.p = Pinyin()
        self.punctuation_dict = {
            u'，': ',',
            u'。': '.',
            u'！': '!',
            u'？': '?',
            u'：': ':',
        }

    def convert_punctuation(self, chinese_text):
        converted_text = ''
        for c in chinese_text:
            # print(c)
            if c in self.punctuation_dict.keys():
                converted_text += self.punctuation_dict[c]
            else:
                converted_text += c
        return converted_text

    def get_pinyin(self, chinese_text):
        return self.p.get_pinyin(self.convert_punctuation(chinese_text),
                splitter=' ', tone_marks='numbers')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command Usages of Pinyin Convertor')
    parser.add_argument("-s", "--sentence", type=str, help="chinese sentense")
    # parser.add_argument("-o", "--output", type=str, default='wav_files', help="WAV file output")
    args = parser.parse_args()

    helper = PinyinHelper()
    print(helper.get_pinyin(args.sentence))
