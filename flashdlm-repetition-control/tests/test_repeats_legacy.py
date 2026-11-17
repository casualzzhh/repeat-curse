import unittest
from repeats import spans,requested

class CharacterTokenizer:
    def decode(self,ids,**kw):return ''.join(map(chr,ids))

class RepetitionTests(unittest.TestCase):
    def setUp(self):self.tok=CharacterTokenizer()
    def events(self,s):return spans(list(map(ord,s)),self.tok)
    def test_numeric_is_not_abnormal(self):self.assertFalse(self.events('100 24000 2.666666666667'))
    def test_token(self):self.assertTrue(any(s['kind']==0 for s in self.events('挖挖')))
    def test_phrase(self):self.assertTrue(any(s['kind']==1 for s in self.events('主轴、主轴、主轴、')))
    def test_sentence(self):self.assertTrue(any(s['kind']==2 for s in self.events('这是一句足够长的测试句子。这是一句足够长的测试句子。')))
    def test_template(self):self.assertTrue(any(s['kind']==2 for s in self.events('The wind was so strong that the old tree shook. The wind was so strong that the tall tree shook.')))
    def test_intent(self):
        self.assertTrue(requested('Repeat hello five times.'))
        self.assertFalse(requested('How much is exactly 100 dollars?'))
        self.assertFalse(spans(list(map(ord,'好好好好好')),self.tok,True))
    def test_distinct_sentences(self):self.assertFalse(any(s['kind']==2 for s in self.events('图书馆提供借阅服务。博物馆展示历史文物。')))

if __name__=='__main__':unittest.main()
