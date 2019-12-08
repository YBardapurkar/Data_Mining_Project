import numpy as np
import pickle
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

class ImageCaption():

	def __init__(self):
		print('Image Caption Created')


	def init(self):
		with open('image_caption_model/tokenizer.gru.pkl', 'rb') as handle:
			self.tokenizer = pickle.load(handle)

		with open('image_caption_model/meta.gru.pkl', 'rb') as handle:
			meta_dict = pickle.load(handle)

		self.max_length = meta_dict['max_length']
		self.attention_features_shape = meta_dict['attention_features_shape']
		self.embedding_dim = meta_dict['embedding_dim']
		self.units = meta_dict['units']
		self.vocab_size = meta_dict['vocab_size']

		self.encoder_new = CNN_Encoder(self.embedding_dim)
		self.decoder_new = RNN_Decoder(self.embedding_dim, self.units, self.vocab_size)

		image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

		new_input = image_model.input
		hidden_layer = image_model.layers[-1].output
		self.image_features_extract_model_new = tf.keras.Model(new_input, hidden_layer)

		self.encoder_new.load_weights('image_caption_model/encoder.gru')  # path to the saved weight files 
		self.decoder_new.load_weights('image_caption_model/decoder.gru')
		self.decoder_new.attention.load_weights('image_caption_model/attention.gru')


	def load_image(self, image_path):
		img = tf.compat.v1.read_file(image_path)
		img = tf.image.decode_jpeg(img, channels=3)
		img = tf.image.resize_images(img, (299, 299))
		img = tf.keras.applications.inception_v3.preprocess_input(img)
		return img, image_path

	def evaluate(self, image):
		attention_plot = np.zeros((self.max_length, self.attention_features_shape))

		hidden = self.decoder_new.reset_state(batch_size=1)

		temp_input = tf.expand_dims(self.load_image(image)[0], 0)
		img_tensor_val = self.image_features_extract_model_new(temp_input)
		img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

		features = self.encoder_new(img_tensor_val)

		dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
		result = []

		for i in range(self.max_length):
			predictions, hidden, attention_weights = self.decoder_new(dec_input, features, hidden)

			attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

			predicted_id = tf.argmax(predictions[0]).numpy()
			result.append(self.tokenizer.index_word[predicted_id])

			if self.tokenizer.index_word[predicted_id] == '<end>':
				return result, attention_plot

			dec_input = tf.expand_dims([predicted_id], 0)

		attention_plot = attention_plot[:len(result), :]
		return result, attention_plot

	def generate_caption(self, image_url):
		image_extension = image_url[-4:]
		image_name = image_url.split('/')[-1]
		image_path = tf.keras.utils.get_file(image_name, origin=image_url)

		result, attention_plot = self.evaluate(image_path)
		caption = ' '.join(result)
		print(caption)
		return caption

	# # captions on the validation set
	# rid = np.random.randint(0, len(img_name_val))
	# image = img_name_val[rid]
	# real_caption = ' '.join([self.tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
	# result, attention_plot = evaluate(image)

	# print ('Real Caption:', real_caption)
	# print ('Prediction Caption:', ' '.join(result))
	# plot_attention(image, result, attention_plot)
	# # opening the image
	# Image.open(img_name_val[rid])

def gru(units):
	return tf.keras.layers.GRU(units, return_sequences=True, return_state=True, recurrent_activation='sigmoid', recurrent_initializer='glorot_uniform')


class BahdanauAttention(tf.keras.Model):
	def __init__(self, units):
		super(BahdanauAttention, self).__init__()
		self.W1 = tf.keras.layers.Dense(units)
		self.W2 = tf.keras.layers.Dense(units)
		self.V = tf.keras.layers.Dense(1)
  
	def call(self, features, hidden):
		# features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

		# hidden shape == (batch_size, hidden_size)
		# hidden_with_time_axis shape == (batch_size, 1, hidden_size)
		hidden_with_time_axis = tf.expand_dims(hidden, 1)

		# score shape == (batch_size, 64, hidden_size)
		score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

		# attention_weights shape == (batch_size, 64, 1)
		# we get 1 at the last axis because we are applying score to self.V
		attention_weights = tf.nn.softmax(self.V(score), axis=1)

		# context_vector shape after sum == (batch_size, hidden_size)
		context_vector = attention_weights * features
		context_vector = tf.reduce_sum(context_vector, axis=1)

		return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
	# Since we have already extracted the features and dumped it using pickle
	# This encoder passes those features through a Fully connected layer
	def __init__(self, embedding_dim):
		super(CNN_Encoder, self).__init__()
		# shape after fc == (batch_size, 64, embedding_dim)
		self.fc = tf.keras.layers.Dense(embedding_dim)
        
	def call(self, x):
		x = self.fc(x)
		x = tf.nn.relu(x)
		return x


class RNN_Decoder(tf.keras.Model):
	def __init__(self, embedding_dim, units, vocab_size):
		super(RNN_Decoder, self).__init__()
		self.units = units

		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = gru(self.units)
		self.fc1 = tf.keras.layers.Dense(self.units)
		self.fc2 = tf.keras.layers.Dense(vocab_size)

		self.attention = BahdanauAttention(self.units)
        
	def call(self, x, features, hidden):
		# defining attention as a separate model
		context_vector, attention_weights = self.attention(features, hidden)

		# x shape after passing through embedding == (batch_size, 1, embedding_dim)
		x = self.embedding(x)

		# x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
		x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

		# passing the concatenated vector to the GRU
		output, state = self.gru(x)

		# shape == (batch_size, max_length, hidden_size)
		x = self.fc1(output)

		# x shape == (batch_size * max_length, hidden_size)
		x = tf.reshape(x, (-1, x.shape[2]))

		# output shape == (batch_size * max_length, vocab)
		x = self.fc2(x)

		return x, state, attention_weights

	def reset_state(self, batch_size):
		return tf.zeros((batch_size, self.units))

# captions on the validation set
# rid = np.random.randint(0, len(img_name_val))
# image = img_name_val[rid]
# real_caption = ' '.join([self.tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
# result, attention_plot = evaluate(image)

# print ('Real Caption:', real_caption)
# print ('Prediction Caption:', ' '.join(result))
# plot_attention(image, result, attention_plot)
# # opening the image
# Image.open(img_name_val[rid])

# image_caption = ImageCaption()
# image_caption.init()

# i = 380
# image_url = 'https://raw.githubusercontent.com/YBardapurkar/ImageCaption/master/' + str(i) + '.jpg'
# image_extension = image_url[-4:]
# image_path = tf.keras.utils.get_file('new' + str(i) + image_extension, origin=image_url)

# result, attention_plot = image_caption.evaluate(image_path)
# caption = ' '.join(result)
# print(caption)