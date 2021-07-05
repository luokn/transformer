from nn.transformer import Transformer
import torch
from torch import nn, optim
sentences = [
    ['i love you <space>', '<bos> ich liebe dich', 'ich liebe dich <eos>'],
    ['you love me <space>', '<bos> du liebst mich', 'du liebst mich <eos>']
]
source_vocab = ['<space>', 'i', 'love', 'you', 'me']
target_vocab = ['<space>', '<bos>', '<eos>', 'ich', 'liebe', 'dich', 'du', 'liebst', 'mich']
source_vocab_dict = {word: i for i, word in enumerate(source_vocab)}
target_vocab_dict = {word: i for i, word in enumerate(target_vocab)}
encoder_inputs = torch.tensor([[source_vocab_dict[word] for word in strs[0].split(' ')] for strs in sentences])
decoder_inputs = torch.tensor([[target_vocab_dict[word] for word in strs[1].split(' ')] for strs in sentences])
decoder_outputs = torch.tensor([[target_vocab_dict[word] for word in strs[2].split(' ')] for strs in sentences])

print(source_vocab_dict)
print(target_vocab_dict)
print(encoder_inputs)
print(decoder_inputs)
print(decoder_outputs)

if __name__ == "__main__":
    net = Transformer(source_vocab_size=len(source_vocab), target_vocab_size=len(target_vocab), max_len=4)
    output = net(encoder_inputs, decoder_inputs)
    print(output.shape)
