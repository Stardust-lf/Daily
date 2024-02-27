from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 创建一个空的BPE模型
tokenizer = Tokenizer(BPE())

# 创建一个训练器，设置词汇表大小为16
trainer = BpeTrainer(vocab_size=16, min_frequency=2, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# 设置预分词器为空格分词
tokenizer.pre_tokenizer = Whitespace()

# 训练模型
tokenizer.train_from_iterator(["old", "older", "oldest", "hug", "pug", "hugs"], trainer)

# 使用训练好的分词器对单词进行分词
words = ["hold", "oldest", "older", "pug", "mug", "huggingf ace"]
for word in words:
    print(tokenizer.encode(word).tokens)
