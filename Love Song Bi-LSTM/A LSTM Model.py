###load Files

a1 = open(r' PATH \Lyrics\Love\AFranklin_SayaLittlePrayer.txt', encoding='utf8').read()
a2 ...

###join everything together into one large string
lyrics = ''.join([a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18 , a19 , a20 , a21  , a23 , a24 , a25 , a26 , a27 , a28 , a29 , a30 , a31 , a32 , a33 , a34 , a35 , a36 , a37 , a38 , a40 , a41 , a42 , a43 , a44 , a45 , a46 , a47 , a48 , a49 , a50 , a51 , a52, a53 , a54 , a55 ,a56 ,a57 , a58 , a59 , a60 , a61 , a62 , a63 , a64 , a65, a66 , a67 , a68 , a69 , a70 , a71 , a72 , a73 , a74 , a75 , a76 , a77 , a78 , a79])


###Wordcloud set-up
wordcloud = WordCloud(max_font_size=100,
                      max_words=250,
                      background_color="black").generate(lyrics)
  

###Plotting the WordCloud
plt.figure(figsize=(12, 4))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig("WordCloud.png")
plt.show()


###Generating the corpus by splitting the text into lines
corpus = lyrics.lower().split("\n")
print(corpus[:10])

#Random repeating word causing issues 
unwanted_word = "handle"  # word  to remove

# Remove the unwanted word from the corpus
filtered_corpus = []
for line in corpus:
    filtered_line = " ".join([word for word in line.split() if word != unwanted_word])
    filtered_corpus.append(filtered_line)

###Fitting a Tokenizer on the Corpus
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
  
  
###Count of the corpus
total_words = len(tokenizer.word_index)
  
print("Total Words:", total_words)


###Converting the text into embeddings for numeric identification 
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
  
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
  
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences,
                                         maxlen=max_sequence_len,
                                         padding='pre'))
predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
label = ku.to_categorical(label, num_classes=total_words+1)


###Building a Bi-Directional LSTM Model
model = Sequential()
model.add(Embedding(total_words+1, 250,
					input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(175, return_sequences=True)))
model.add(Dropout(0.2))
model.add(LSTM(120))
model.add(Dense(total_words+1/2, activation='relu',
				kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(total_words+1, activation='softmax'))
model.compile(loss='categorical_crossentropy',
			optimizer='adam', metrics=['accuracy'])
print(model.summary())


###train model
history = model.fit(predictors, label, epochs=100, verbose=1)

###See output of loss and accuracy
print(history.history['loss'])
print(history.history['accuracy'])

###Plot model performance 
from matplotlib import pyplot
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['accuracy'])
pyplot.title('model loss vs accuracy')
pyplot.xlabel('epoch')
pyplot.legend(['loss', 'accuracy'], loc='upper right')
pyplot.show() 




#####provide opening words to start the process
seed_text = "I am"       ###use whatever words you want to start the 'song' with
next_words = 100         ###Numer of words to output afterwards
ouptut_text = ""
  
for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences(
        [token_list], maxlen=max_sequence_len-1,
      padding='pre')
    predicted = np.argmax(model.predict(token_list, 
                                        verbose=0), axis=-1)
    output_word = ""
      
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
              
    seed_text += " " + output_word
      
print(seed_text)   ###See your lyrics
