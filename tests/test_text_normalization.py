import pytest

from daseg.data import create_text_normalizer

normalize = create_text_normalizer()


@pytest.mark.parametrize(
    ["text", "result"],
    [
        ('((such fail))', ''),
        ('((such fail)) dummy', 'dummy'),
        ('(such fail)', ''),
        ('(such fail) dummy', 'dummy'),
        ('<<such fail>>', ''),
        ('<<such fail>> dummy', 'dummy'),
        ('<such fail>', '<such fail>'),
        ('<such fail> dummy', '<such fail> dummy'),
        ('# <such fail> #', '<such fail>'),
        ('# <such fail> dummy #', '<such fail> dummy'),
        ('this is text * this is comment?', 'this is text'),
        ('# <such fail> # * comment', '<such fail>'),
        ('# <such fail> dummy # * comment', '<such fail> dummy'),
        ('punctuation ?', 'punctuation?'),
        ('. argh', 'argh'),
        (', . argh', 'argh'),
        ('?!,. ! ? !, . argh', 'argh'),
        ('<<this is a pretty >> ((complicated)) (example) . Correct . <normalization> # (will resolve) # * this',
         'Correct. <normalization>'),
        ('This is a regular turn.', 'This is a regular turn.'),
        ('This is a regular turn. Quite? Some! -- punctuation, which is ok!',
         'This is a regular turn. Quite? Some! punctuation, which is ok!')
    ]
)
def test_text_normalization(text, result):
    assert normalize(text) == result
