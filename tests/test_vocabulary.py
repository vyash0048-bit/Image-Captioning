from captioner.vocabulary import Vocabulary


def test_vocabulary_roundtrip() -> None:
    vocabulary = Vocabulary()
    vocabulary.build(["A dog runs", "A dog sleeps"], min_frequency=1)
    assert vocabulary.decode(vocabulary.encode("A dog runs")) == "a dog runs"
