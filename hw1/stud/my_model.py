'''This module implents the trainig of the Neural Network for Named Entity
Recognition (NER). To run use the following command:

$ caffeinate -d python3 -O -m hw1.stud.my_model

-OO cannot be used because it breaks pytorch, I'm speachless...

The pre-trained word2vec model can be downloaded from this page:
https://code.google.com/archive/p/word2vec/'''

import io # Diciamo che non mi serve davvero importarlo.
# import unicodedata, string
import multiprocessing

import torch

# https://github.com/SapienzaNLP/nlp2022-hw1

class NERDataset(torch.utils.data.Dataset):
	def __init__(
			self,
			input_file: io.TextIOBase,
			window_size: int,
			window_shift: int,
			word2index: dict[str, int],
			entity2index: dict[str, int]) -> None:
		# Each sentence in the corpus is a list of 'word entity-name' pairs
		corpus: list[list[tuple[str, str]]] = []
		sentence: list[tuple[str, str]] = []
		oov_token_counter: int = 0
		token_counter: int = 0
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
			# word = word.lower()
			# Removing accent from letters
			# NOTE: looks like this makes thing worse: oov_token_counter=3871 vs oov_token_counter=3850
			# word = ''.join(c for c in unicodedata.normalize('NFD', word)
			# 	if unicodedata.category(c) != 'Mn'
			# 	and c in string.ascii_letters + ".,;'")
			# Supstituting unknown words.
			if word not in word2index:
				word = OOV_TOKEN
				oov_token_counter += 1
			token_counter += 1
			sentence.append((word, entity))
		if sentence != []:
			# In case the file doesn't finish with a newline.
			corpus.append(sentence)
			sentence = []

		print(f'{token_counter=}')
		print(f'{oov_token_counter=}')

		# Now each sentence is splitted in pieces of length window_size. After
		# this fixed_len_sentences will contain a list of sentences each of
		# length window_size.
		fixed_len_sentences: list[list[tuple[str, str]]] = []
		for sentence in corpus:
			for i in range(0, len(sentence), window_shift):
				window: list[tuple[str, str]] = sentence[i:i+window_size]
				if len(window) < window_size:
					window.extend([(PAD_TOKEN, PAD_ENTITY) for _ in range(window_size - len(window))])
				assert len(window) == window_size
				fixed_len_sentences.append(window)
		assert all(len(sentence) == window_size for sentence in fixed_len_sentences)

		self.tensors: list[tuple[torch.LongTensor, torch.LongTensor]] = []
		for sentence in fixed_len_sentences:
			X = torch.tensor([word2index[word] for word in map(lambda tup: tup[0], sentence)])
			assert X.shape[0] == window_size
			y = torch.tensor([entity2index[entity] for entity in map(lambda tup: tup[1], sentence)])
			assert y.shape[0] == window_size
			self.tensors.append((X, y))

	def __len__(self) -> int:
		return len(self.tensors)

	def __getitem__(self, index: int) -> tuple[torch.LongTensor, torch.LongTensor]:
		return self.tensors[index]

class NERModule(torch.nn.Module):
	def __init__(
			self,
			word_embeddings: torch.Tensor,
			dropout_rate: float,
			lstm_hidden_dim: int,
			lstm_layers_num: int,
			num_entities: int):
		super().__init__()
		embedding_dim = word_embeddings.shape[1]
		self.embedding = (torch.nn.Embedding
			.from_pretrained(word_embeddings, freeze=True))
		assert self.embedding.num_embeddings == word_embeddings.shape[0]
		assert self.embedding.embedding_dim == embedding_dim
		self.dropout = torch.nn.Dropout(dropout_rate)
		self.recurrent = torch.nn.LSTM(
			input_size=embedding_dim,
			hidden_size=lstm_hidden_dim,
			num_layers=lstm_layers_num,
			dropout=dropout_rate
		)
		lstm_output_dim = lstm_hidden_dim
		self.classifier = torch.nn.Linear(lstm_output_dim, num_entities)

	def forward(self, x: torch.LongTensor):
		embeddings = self.embedding(x)
		embeddings = self.dropout(embeddings)
		o, (h, c) = self.recurrent(embeddings)
		o = self.dropout(o)
		output = self.classifier(o)
		return output

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
	# NOTE: using 0 and 1 vectors is probably a bad idea... Should I use rendomvectors instead?
	kv.add_vectors([OOV_TOKEN, PAD_TOKEN],
		[numpy.zeros(kv.vector_size), numpy.ones(kv.vector_size)])
	assert all(kv.vectors[kv.key_to_index['king']] == kv['king'])

	word_embeddings = torch.Tensor(kv.vectors)
	index2word: list[str] = kv.index_to_key
	word2index: dict[str, int] = kv.key_to_index

	torch.save(word_embeddings, path1)
	with open(path2, 'w') as f:
		for word in index2word:
			f.write(word)
			f.write('\n')

SEED:            int   = 42
OOV_TOKEN:       str   = '<UNK>'
PAD_TOKEN:       str   = '<PAD>'
PAD_ENTITY:      str   = 'PAD'
EPOCHS:          int   = 100
DROPOUT_RATE:    float = 0.5
LSTM_HIDDEN_DIM: int   = 128
LSTM_LAYERS:     int   = 1
WINDOW_SIZE:     int   = 10
BATCH_SIZE:      int   = 100
TRAIN_FNAME:     str   = 'data/train.tsv'
DEV_FNAME:       str   = 'data/dev.tsv'
VEC_FNAME:       str   = 'model/word2vec-google-news-300.gz'
MODEL_FNAME:     str   = 'model/model.pt'
EMBED_FNAME:     str   = 'model/embeggings.pt'
VOCAB_FNAME:     str   = 'model/vocabulary.txt'

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

	# Vocabulary and word embreddings
	word_embeddings: torch.Tensor = torch.load(EMBED_FNAME)
	embeddings_dimension: int = word_embeddings.shape[1]
	print(f'{embeddings_dimension=}')
	index2word: list[str] = []
	word2index: dict[str, int] = {}
	with open(VOCAB_FNAME) as f:
		for index, word in enumerate(f):
			word = word.rstrip()
			index2word.append(word)
			word2index[word] = index
	assert OOV_TOKEN in word2index
	index2entity: list[str] = [
		'O',
		'B-PER', 'B-LOC', 'B-GRP', 'B-CORP', 'B-PROD', 'B-CW',
		'I-PER', 'I-LOC', 'I-GRP', 'I-CORP', 'I-PROD', 'I-CW',
		PAD_ENTITY,
	]
	entity2index: dict[str, int] = {entity: index for index, entity in enumerate(index2entity)}

	# Model stuff
	with open(TRAIN_FNAME) as f:
		dataset = NERDataset(f, WINDOW_SIZE, WINDOW_SIZE, word2index, entity2index); del f
	print(f'train {len(dataset)=}')
	train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
	print(f'validation {len(dataset)=}')
	with open(DEV_FNAME) as f:
		dataset = NERDataset(f, WINDOW_SIZE, WINDOW_SIZE, word2index, entity2index); del f
	validation_dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
	model = NERModule(
		word_embeddings=word_embeddings,
		dropout_rate=DROPOUT_RATE,
		lstm_hidden_dim=LSTM_HIDDEN_DIM,
		lstm_layers_num=LSTM_LAYERS,
		num_entities=len(index2entity)
	)
	# To ditch the gensim dependency...
	# torch.save(model.embedding.weight, 'path/googlenews.pt')
	# word_embeddings = torch.load('path/googlenews.pt')
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
			inputs, labels = batch
			optimizer.zero_grad()

			predictions = model(inputs)
			predictions = predictions.view(-1, predictions.shape[-1])
			labels = labels.view(-1)

			loss = criterion(predictions, labels)
			loss.backward()
			optimizer.step()

			epoch_loss += loss.tolist()

			if log_level > 1 and step % log_steps == log_steps - 1:
				print(f'\t[E: {epoch:2d} @ step {step:3d}] current avg loss = {epoch_loss / (step + 1):0.4f}')

		avg_epoch_loss = epoch_loss / len(train_dataloader)
		train_loss += avg_epoch_loss
		if log_level > 0:
			print(f'\t[E: {epoch:2d}] train loss = {avg_epoch_loss:0.4f}')

		valid_loss = 0.0
		model.eval()
		with torch.no_grad():
			for batch in validation_dataloader:
				inputs, labels = batch
				assert inputs.shape == labels.shape
				assert inputs.shape[0] <= BATCH_SIZE and inputs.shape[1] ==  WINDOW_SIZE, f'{inputs.shape}'

				predictions = model(inputs)
				assert (lambda p: p[0] <= BATCH_SIZE and p[1] == WINDOW_SIZE and p[2] == len(index2entity))(predictions.shape)
				predictions = predictions.view(-1, predictions.shape[-1])
				assert predictions.shape[0] <= BATCH_SIZE*WINDOW_SIZE and predictions.shape[1] == len(index2entity)
				labels = labels.view(-1)
				assert labels.shape[0] <= BATCH_SIZE*WINDOW_SIZE
				sample_loss = criterion(predictions, labels)
				valid_loss += sample_loss.tolist()

		valid_loss /= len(validation_dataloader)

		if log_level > 0:
			print(f'  [E: {epoch:2d}] valid loss = {valid_loss:0.4f}')

	if log_level > 0:
		print('... Done!')

	avg_epoch_loss = train_loss / EPOCHS

	torch.save(model.state_dict(), MODEL_FNAME)

	return 0

if __name__ == '__main__':
	raise SystemExit(main())
