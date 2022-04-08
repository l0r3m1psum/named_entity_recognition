'''This module implents the trainig of the Neural Network for Named Entity
Recognition (NER). To run use the following command:

$ caffeinate -d python3 -O -m hw1.stud.my_model

-OO cannot be used because it breaks pytorch, I'm speachless...

The pre-trained word2vec model can be downloaded from this page:
https://code.google.com/archive/p/word2vec/

Info about this homework can be found here:
https://github.com/SapienzaNLP/nlp2022-hw1'''

import multiprocessing, functools, re, unicodedata
from typing import List, Tuple, Dict

import torch

# TYPES ########################################################################

recognized_sentence_t = List[Tuple[str, str]]
index_token_converter_t = List[str]
token_index_converter_t = Dict[str, int]

# FUNCTIONS ####################################################################

def read_vocab(path: str) -> Tuple[index_token_converter_t, token_index_converter_t]:
	index2word: index_token_converter_t = []
	word2index: token_index_converter_t = {}
	with open(path) as f:
		for index, word in enumerate(f):
			file_line_no = index+1
			word = word.rstrip()
			if not word:
				print(f'skipping empty line in {path}:{file_line_no}')
				continue
			if ' ' in word:
				raise Exception(f'space in word found {path}:{file_line_no}')
			index2word.append(word)
			word2index[word] = index
	assert OOV_TOKEN in word2index and OOV_TOKEN in index2word
	assert PAD_TOKEN in word2index and PAD_TOKEN in index2word
	assert len(index2word) == len(word2index)
	return index2word, word2index

def prepare_batch(
	word2index: token_index_converter_t,
	batch: List[recognized_sentence_t]
	) -> Tuple[torch.LongTensor, torch.LongTensor]:

	# TODO: transforming sentences in numbers can be done in the Dataset.

	sentences: List[List[str]] = [[tup[0] for tup in recog_sent] for recog_sent in batch]
	sentences_labels: List[List[str]] = [[tup[1] for tup in recog_sent] for recog_sent in batch]
	assert len(sentences) == len(sentences_labels)
	assert all(len(a) == len(b) for a, b in zip(sentences, sentences_labels))

	x: List[List[int]] = [[word2index.get(word, word2index[OOV_TOKEN]) for word in sentence] for sentence in sentences]
	y: List[List[int]] = [[entity2index[label] for label in labels] for labels in sentences_labels]

	# At this point each list inside x and y can have different length, so we
	# fix this issue.

	# X and Y will be tensor of shape (B,T), where B is the batch size and T is
	# the length of the longest sequence
	X = torch.nn.utils.rnn.pad_sequence(
		[torch.as_tensor(sample) for sample in x],
		batch_first=True,
		padding_value=word2index[PAD_TOKEN]
	)
	Y = torch.nn.utils.rnn.pad_sequence(
		[torch.as_tensor(sample) for sample in y],
		batch_first=True,
		padding_value=entity2index[PAD_ENTITY]
	)
	assert X.shape == Y.shape
	assert X.shape == (len(batch), max(map(len, sentences)))
	return X, Y

def clean_word(word: str):
	# regs = [
	# 	# This matches all numbers composed of just digits that can be optionally
	# 	# followed by some letters, like in the case of units of measure e.g. 10mm.
	# 	'^[0-9½¾]+[a-z]*$',
	# 	# This matches all numbes that have a dot or a coma in the middle optionally
	# 	# followed by some letters, again in the case that it is a unit of measure.
	# 	'^[0-9½¾]*[\.,][0-9½¾]+[a-z]*$',
	# 	# This matches all numbers alternated by dots or dashes in the middle, stuff
	# 	# like version numbers 1.2.3 and dates 1-2-3 they can optionally be followeb
	# 	# by letters.
	# 	'^([0-9½¾]+[\.,-/x–])+[0-9½¾]+[a-z]*$',
	# 	# This matches all numbers that ends with a dot.
	# 	'^[0-9½¾]+\.$',
	# 	# This matches dates like 2k20
	# 	'^[0-9½¾]k[0-9½¾]{2}$',
	# ]
	# # All lines that matches this regex shall be substituted with the '<NUM>'
	# # token.
	# number_re = re.compile('|'.join(regs))

	# if len(word) == 1:
	# 	cat = unicodedata.category(word)
	# 	if cat[0] == 'P':
	# 		word = '<PUN>'
	# 	elif cat[0] == 'S':
	# 		word = '<SYM>'
	# word = re.sub('[’’]', "'", word)
	# word = re.sub('[“”]', '"', word)
	# word = re.sub("""^[-_.(¿"']|[-_:,;?!)'"]$|\.\.+$|\'s$""", '', word)
	# # UGLY UGLY hack to clean words like "xxx),", that end with two
	# # consecutive bad characters.
	# word = re.sub("""^[-_.(¿"']|[-_:,;?!)'"]$|\.\.+$|\'s$""", '', word)
	# word = number_re.sub('<NUM>', word)
	# if len(word) > 1:
	# 	word = re.sub('[,:;]$', '', word)
	assert word != ''
	return word

# CLASSES ######################################################################

class NERDataset(torch.utils.data.Dataset):
	def __init__(self, input_file_name: str) -> None:
		# Each sentence in the corpus is a list of 'word entity-name' pairs
		corpus: List[recognized_sentence_t] = []
		sentence: recognized_sentence_t = []
		# input_file: io.TextIOBase
		with open(input_file_name) as input_file:
			for num, line in enumerate(input_file, 1):
				if line == '\n':
					# We are at the end of a sentence
					if sentence == []:
						raise Exception(f'error @ {input_file.name}:{num}')
					corpus.append(sentence)
					sentence = []
					continue
				if line[0] == '#':
					# We are at the beginning of a sentence
					if sentence != []:
						raise Exception(f'error @ {input_file.name}:{num}')
					continue
				# We are looking at a 'word entity-name' pair
				word, entity = line.split('\t')
				entity = entity.rstrip()
				sentence.append((word, entity))
			if sentence != []:
				# In case the file doesn't finish with a newline.
				corpus.append(sentence)
				sentence = []
		assert sentence == []

		self.data = corpus

	def __len__(self) -> int:
		return len(self.data)

	def __getitem__(self, index: int) -> recognized_sentence_t:
		return self.data[index]

class NERModule(torch.nn.Module):
	def __init__(
			self,
			num_embeddings: int,
			embedding_dim: int,
			dropout_rate: float,
			lstm_hidden_dim: int,
			lstm_layers_num: int,
			out_features: int
			) -> None:
		super().__init__()
		self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
		self.dropout = torch.nn.Dropout(dropout_rate)

		self.recurrent = torch.nn.LSTM(
			input_size=embedding_dim,
			hidden_size=lstm_hidden_dim,
			num_layers=lstm_layers_num,
			batch_first=True,
			bidirectional=True
			# , dropout=dropout_rate
		)
		lstm_output_dim = 2*lstm_hidden_dim
		self.classifier = torch.nn.Linear(lstm_output_dim, out_features)

	def forward(self, x: torch.LongTensor) -> torch.Tensor:
		assert x.shape[0] <= BATCH_SIZE # the second element should be the sequence length
		if __debug__: seq_len = x.shape[1]
		assert x.dim() == 2
		embeddings = self.embedding(x)         # (*) -> (*,H)
		assert embeddings.shape[0] <= BATCH_SIZE \
			and embeddings.shape[1:] == (seq_len, EMBED_DIM)
		embeddings = self.dropout(embeddings)  # (*) -> (*)
		# N = batch size
		# L = sequence length
		# H_in = input size
		# H_out = hidden size
		o, (h, c) = self.recurrent(embeddings) # (N,L,H_in) -> (N,L,H_out)
		assert o.dim() == 3
		assert o.shape[0] <= BATCH_SIZE and o.shape[1] == seq_len
		o = self.dropout(o)
		output = self.classifier(o) # (*,H_in') -> (*,H_out')
		assert output.dim() == 3
		assert output.shape == (o.shape[0], seq_len, NUM_ENTITIES)
		return output

# CONSTANTS ####################################################################

SEED:            int   = 42
OOV_TOKEN:       str   = '<UNK>'
PAD_TOKEN:       str   = '<PAD>'
PAD_ENTITY:      str   = 'PAD'
EPOCHS:          int   = 17 # 100
DROPOUT_RATE:    float = 0.5
LSTM_HIDDEN_DIM: int   = 128
LSTM_LAYERS:     int   = 1
BATCH_SIZE:      int   = 100
EMBED_DIM:       int   = 300 # like google news
TRAIN_FNAME:     str   = 'data/train.tsv'
DEV_FNAME:       str   = 'data/dev.tsv'
MODEL_FNAME:     str   = 'model/model.pt'
VOCAB_FNAME:     str   = 'model/lexicon2.txt'
DATALOADER_WORKERS: int = 0

index2entity: index_token_converter_t = [
	'O',
	'B-PER', 'B-LOC', 'B-GRP', 'B-CORP', 'B-PROD', 'B-CW',
	'I-PER', 'I-LOC', 'I-GRP', 'I-CORP', 'I-PROD', 'I-CW',
	PAD_ENTITY,
]
entity2index: token_index_converter_t = {
	entity: index for index, entity in enumerate(index2entity)
}

NUM_ENTITIES:    int   = len(index2entity)

catcode2catname = {
	'Cc': 'Other, Control',
	'Cf': 'Other, Format',
	'Cn': 'Other, Not Assigned (no characters in the file have this property)',
	'Co': 'Other, Private Use',
	'Cs': 'Other, Surrogate',
	'LC': 'Letter, Cased',
	'Ll': 'Letter, Lowercase',
	'Lm': 'Letter, Modifier',
	'Lo': 'Letter, Other',
	'Lt': 'Letter, Titlecase',
	'Lu': 'Letter, Uppercase',
	'Mc': 'Mark, Spacing Combining',
	'Me': 'Mark, Enclosing',
	'Mn': 'Mark, Nonspacing',
	'Nd': 'Number, Decimal Digit',
	'Nl': 'Number, Letter',
	'No': 'Number, Other',
	'Pc': 'Punctuation, Connector',
	'Pd': 'Punctuation, Dash',
	'Pe': 'Punctuation, Close',
	'Pf': 'Punctuation, Final quote (may behave like Ps or Pe depending on usage)',
	'Pi': 'Punctuation, Initial quote (may behave like Ps or Pe depending on usage)',
	'Po': 'Punctuation, Other',
	'Ps': 'Punctuation, Open',
	'Sc': 'Symbol, Currency',
	'Sk': 'Symbol, Modifier',
	'Sm': 'Symbol, Math',
	'So': 'Symbol, Other',
	'Zl': 'Separator, Line',
	'Zp': 'Separator, Paragraph',
	'Zs': 'Separator, Space',
}

# MAIN #########################################################################

def main() -> int:
	torch.set_num_threads(multiprocessing.cpu_count())
	# Seeding stuff
	# os.environ['PYTHONHASHSEED'] = SEED
	# random.seed(SEED)
	# numpy.random.seed(SEED)
	torch.manual_seed(SEED)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.use_deterministic_algorithms(True)

	# Generating the lexicon from the training data
	print('generating the lexicon')
	words = {OOV_TOKEN, PAD_TOKEN}
	with open(VOCAB_FNAME, 'w') as lexicon_file, open(TRAIN_FNAME) as train_data_file:
		for line in train_data_file:
			if line == '\n' or line[0] == '#':
				continue
			word, _ = line.split('\t')
			word = clean_word(word)
			words.add(word)
		for word in sorted(words):
			print(word, file=lexicon_file)
	del words

	# Vocabulary
	index2word: index_token_converter_t
	word2index: token_index_converter_t
	index2word, word2index = read_vocab(VOCAB_FNAME)

	# Model stuff
	my_collate_fn = functools.partial(prepare_batch, word2index)
	train_dataset = NERDataset(TRAIN_FNAME)
	print(f'{len(train_dataset)=}')
	train_dataloader = torch.utils.data.DataLoader(
		train_dataset,
		collate_fn=my_collate_fn,
		batch_size=BATCH_SIZE,
		num_workers=DATALOADER_WORKERS,
		shuffle=True
	)
	validation_dataset = NERDataset(DEV_FNAME)
	print(f'validation {len(validation_dataset)=}')
	validation_dataloader = torch.utils.data.DataLoader(
		validation_dataset,
		collate_fn=my_collate_fn,
		batch_size=BATCH_SIZE,
		num_workers=DATALOADER_WORKERS,
		shuffle=False
	)
	model = NERModule(
		num_embeddings=len(index2word),
		embedding_dim=EMBED_DIM,
		dropout_rate=DROPOUT_RATE,
		lstm_hidden_dim=LSTM_HIDDEN_DIM,
		lstm_layers_num=LSTM_LAYERS,
		out_features=NUM_ENTITIES
	)
	criterion = torch.nn.CrossEntropyLoss(ignore_index=entity2index[PAD_ENTITY])
	optimizer = torch.optim.Adam(model.parameters())

	print(model)

	log_level = 10
	log_steps = 10
	train_loss: float = 0.0
	for epoch in range(EPOCHS):
		if log_level > 0:
			print(f' Epoch {epoch + 1:03d}')
		epoch_loss: float = 0.0

		model.train()
		for step, batch in enumerate(train_dataloader):
			X, Y = batch
			assert X.dim() == 2
			assert X.shape[0] <= BATCH_SIZE
			if __debug__: seq_len = X.shape[1]
			assert X.shape == Y.shape
			optimizer.zero_grad()

			predictions = model(X)
			assert predictions.dim() == 3
			assert predictions.shape[0] <= BATCH_SIZE
			assert predictions.shape[1:] == (seq_len, NUM_ENTITIES)
			predictions = predictions.view(-1, predictions.shape[-1])
			Y = Y.view(-1)
			# COME FUNZIONA LOSS IN QUESTO CASO???
			# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
			# TODO: put shape assertions here.
			loss = criterion(predictions, Y)
			loss.backward()
			optimizer.step()

			epoch_loss += loss.tolist()

			if log_level > 1 and step % log_steps == log_steps - 1:
				print(f'\t[E: {epoch:2d} @ step {step:3d}] current avg loss = '
					f'{epoch_loss / (step + 1):0.4f}')

		avg_epoch_loss = epoch_loss / len(train_dataloader)
		train_loss += avg_epoch_loss
		if log_level > 0:
			print(f'\t[E: {epoch:2d}] train loss = {avg_epoch_loss:0.4f}')

		valid_loss = 0.0
		model.eval()
		with torch.no_grad():
			for batch in validation_dataloader:
				X, Y = batch
				assert X.shape == Y.shape

				predictions = model(X)
				predictions = predictions.view(-1, predictions.shape[-1])
				Y = Y.view(-1)
				sample_loss = criterion(predictions, Y)
				valid_loss += sample_loss.tolist()

		valid_loss /= len(validation_dataloader)

		if log_level > 0:
			print(f'  [E: {epoch:2d}] valid loss = {valid_loss:0.4f}')

	if log_level > 0:
		print('... Done!')

	avg_epoch_loss = train_loss/EPOCHS

	torch.save(model.state_dict(), MODEL_FNAME)

	return 0

if __name__ == '__main__':
	raise SystemExit(main())

# UNUSED FUNCTINOS #############################################################

def recognized_sentence_to_tensors_pairs(
	sentence: recognized_sentence_t,
	window_size: int,
	window_shift: int,
	word2index: token_index_converter_t,
	entity2index: token_index_converter_t
	) -> List[Tuple[torch.LongTensor, torch.LongTensor]]:
	'''This function splits a sentence with recognized entities in pieces of
	length window_size paddind the final piece if it is too short. Then each of
	this splits is transformed in a pair of tensors.'''
	res: List[Tuple[torch.LongTensor, torch.LongTensor]] = []
	for i in range(0, len(sentence), window_shift):
		# Normalizing length.
		window: recognized_sentence_t = sentence[i:i+window_size]
		if len(window) < window_size:
			assert i + window_shift >= len(sentence), 'this is not the last iteration'
			window.extend([(PAD_TOKEN, PAD_ENTITY) for _ in range(window_size - len(window))])
		assert len(window) == window_size, 'not all pieces of the function have the same lenght'
		# Transforming in tensors.
		words: List[str] = []
		entities: List[str] = []
		for word, entity in window:
			words.append(word2index[word])
			entities.append(entity2index[entity])
		x = torch.tensor(words)
		y = torch.tensor(entities)
		assert x.shape[0] == window_size and y.shape[0]  == window_size
		res.append((x, y))
	return res

def unicode_normalization(s: str) -> str:
	'''This functions transform pesky unicode characters in more "normal" one,
	doing stuff like removing accents from letters.'''
	# Removing accent from letters
	import unicodedata, string
	res = ''.join(c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
		and c in string.ascii_letters + ".,;'")
	return res

# The Google News pre-trained embeddings use '#' characters instead of numbers,
# to save space. So numbers in the input must be changed in '#'. It also
# contains the '</s>' token that may be used to my advantage. '_' is used
# instead of spaces in some "words" e.g. "Andromeda_Galaxy_M##"
def export_from_gensim() -> None:
	'''This functions exports a Gensim's pre-trained word2vec plus its
	vocabulary, adding to them them special tokens.'''
	import gensim, numpy
	import os
	if False:
		# If the file does not exists.
		os.environ['GENSIM_DATA_DIR'] = 'model/'
		gensim.downloader.load('word2vec-google-news-300')
	kv = gensim.models.KeyedVectors.load_word2vec_format(VEC_FNAME, binary=True)
	# 	#limit=10000)#limit=None)
	# NOTE: using 0 and 1 vectors is probably a bad idea... Should I use
	# rendomvectors instead?
	kv.add_vectors([OOV_TOKEN, PAD_TOKEN],
		[numpy.zeros(kv.vector_size), numpy.ones(kv.vector_size)])
	assert all(kv.vectors[kv.key_to_index['king']] == kv['king'])

	word_embeddings = torch.Tensor(kv.vectors)
	index2word: index_token_converter_t = kv.index_to_key
	word2index: token_index_converter_t = kv.key_to_index

	# torch.nn.Embedding.from_pretrained(word_embeddings, frozen=True)

	torch.save(word_embeddings, path1)
	with open(path2, 'w') as f:
		for word in index2word:
			f.write(word)
			f.write('\n')
