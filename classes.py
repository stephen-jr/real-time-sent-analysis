from _utils import *
from config import access_token, access_token_secret, consumer_key, consumer_secret


class Model:
    def __init__(self):
        self.train_dataset = pd.DataFrame()
        self.model = Sequential()
        self.tokenizer = Tokenizer(num_words=10000)
        self.X = self.Y = self.X_Test = self.X_Train = self.Y_Train = self.Y_Test = ''
        self.vocab_size, self.maxlen = 0,  126

    def input_train_data(self, dataset):
        if dataset.split('.')[-1] == 'csv':
            try:
                self.train_dataset = pd.read_csv(dataset, error_bad_lines=False, encoding='ISO-8859-1')
            except Exception as e:
                return "Error on loading CSV Dataset", e
        elif dataset.split('.')[-1] == 'json':
            try:
                self.train_dataset = pd.read_json(dataset, lines=True)
            except Exception as e:
                return "Error on loading JSON Dataset", e
        else:
            exit("Specify a csv or json dataset")
        try:
            # self.train_dataset.sample(frac=1)
            self.train_dataset['SentimentText'].astype('str').apply(self.preprocess)
            self.X = self.train_dataset['SentimentText']
            self.Y = self.train_dataset['Sentiment']
        except KeyError:
            exit("Text data have no text field. Ensure your dataset has a text 'column'")
        except Exception as e:
            exit(("Error", e))

    def preprocess(self, text):
        text = re.sub(r'http\S+', ' ', text)
        text = re.sub(r'@\w+', ' ', text)
        text = re.sub(r'pic.\S+', ' ', text)
        text = re.compile(r'<[^>]+>').sub(' ', text)
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
        text = re.sub(r'RT', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def tokenize(self):
        self.X_Train, self.X_Test, self.Y_Train, self.Y_Test = train_test_split(self.X,
                                                                                self.Y,
                                                                                test_size=0.25,
                                                                                random_state=42)
        self.tokenizer.fit_on_texts(self.X_Train)
        self.X_Train = self.tokenizer.texts_to_sequences(self.X_Train)
        self.X_Test = self.tokenizer.texts_to_sequences(self.X_Test)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        print("Found {} unique words".format(self.vocab_size))
        # self.maxlen = max(len(x) for x in self.X_Train)
        self.X_Train = pad_sequences(self.X_Train, padding='post', maxlen=self.maxlen)
        self.X_Test = pad_sequences(self.X_Test, padding='post', maxlen=self.maxlen)

    def create_model(self):
        print("Creating Sentiment Analytic model")
        self.model.add(Embedding(self.vocab_size, 300, input_length=self.maxlen))
        self.model.add(Conv1D(64, 2, activation='relu'))
        self.model.add(MaxPooling1D())
        self.model.add(Conv1D(128, 2, activation='relu'))
        self.model.add(MaxPooling1D())
        self.model.add(Conv1D(256, 3, activation='relu'))
        self.model.add(MaxPooling1D())
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', prcsn, rcll, f1_m])
        self.model.summary()

    def ld_model(self, pth):
        try:
            self.model = load_model(pth, custom_objects={'prcsn': prcsn, 'rcll': rcll, 'f1_m': f1_m})
            dirt = 'models/tokenizer.pkl'
            if os.path.exists(dirt):
                with open(dirt, 'rb') as f:
                    self.tokenizer = dill.load(f)
            else:
                exit("Tokenizer file doesn't exist")
            return True
        except Exception as e:
            return e

    def train(self):
        print("Model Training Process in Session")

        es = EarlyStopping(monitor='val_loss', mode='min', patience=1, verbose=1)
        history = self.model.fit(self.X_Train,
                                 self.Y_Train,
                                 epochs=10,
                                 verbose=True,
                                 validation_data=(self.X_Test, self.Y_Test),
                                 batch_size=128,
                                 callbacks=[es])
        _, train_accuracy, train_precision, train_recall, train_fscore = self.model.evaluate(self.X_Train,
                                                                                             self.Y_Train,
                                                                                             verbose=True)
        print("Training metrics : \n")
        print(f'Accuracy : {train_accuracy:{5}} \n'
              f'Precision : {train_precision:{5}} \n'
              f'Recall : {train_recall:{5}} \n'
              f'F-Score : {train_fscore:{5}} \n')
        print()
        _, test_accuracy, test_precision, test_recall, test_fscore = self.model.evaluate(self.X_Test,
                                                                                         self.Y_Test,
                                                                                         verbose=True)

        print("Testing metrics\n")
        print(f'Accuract : {test_accuracy:{5}} \n'
              f'Precision : {test_precision:{5}} \n'
              f'Recall : {test_recall:{5}} \n'
              f'F-Score : {test_fscore:{5}}\n')
        print()
        ckpt = time.gmtime(time.time())
        ckpt = 'model-' + str(ckpt[0]) + '-' + str(ckpt[1]) + '-' + str(ckpt[2])

        if os.path.exists('models/' + ckpt + '.h5'):
            os.remove('models/' + ckpt + '.h5')

        self.model.save('models/' + ckpt + '.h5')

        if os.path.exists('models/tokenizer.pkl'):
            os.remove('models/tokenizer.pkl')

        with open('models/tokenizer.pkl', 'wb') as f:
            dill.dump(self.tokenizer, f)
        return history

    def classify(self, text):
        text = self.preprocess(text)
        text = [text]
        text = self.tokenizer.texts_to_sequences(text)
        text = pad_sequences(text, padding='post', maxlen=self.maxlen)
        pred = self.model.predict(text)
        rt = {}
        if len(pred) == 1:
            if pred[0] > 0.5:
                rt['classification'] = 'Positive'
            else:
                rt['classification'] = 'Negative'
            rt['score'] = pred[0]
        else:
            exit('Error obtaining prediction classification')
        return rt

    def plot(self, hist):
        plt.style.use('ggplot')
        acc = hist.history['accuracy']
        val_acc = hist.history['val_accuracy']
        loss = hist.history['loss']
        val_loss = hist.history['val_loss']
        x = range(1, len(acc) + 1)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
        plt.savefig('')


class StdOutListener(StreamListener):
    def __init__(self, model_dir):
        super().__init__()
        self.auth = OAuthHandler(consumer_key, consumer_secret)
        self.auth.set_access_token(access_token, access_token_secret)
        t = time.gmtime(time.time())
        self.outfile = "StreamData-" + str(t[0]) + '-' + str(t[1]) + '-' + str(t[2]) + '.csv'
        with open(self.outfile, 'w', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Created_at', 'Text', 'SentimentClassification', 'PolarityScore'])
            # f.write('created_at, text, classification, score')
            # f.write("\n")

        if model_dir:
            try:
                self.model = Model()
                self.model.ld_model(model_dir)
            except Exception as e:
                exit(("StdOutListenerError : ", e))
        else:
            exit("Please Specify model's directory")

    def on_status(self, status):
        if hasattr(status, 'retweeted_status'):
            try:
                tweet = status.retweeted_status.extended_tweet["full_text"]
            except:
                tweet = status.retweeted_status.text
        else:
            try:
                tweet = status.extended_tweet["full_text"]
            except AttributeError:
                tweet = status.text
        prediction = self.model.classify(tweet)
        try:
            print(f'Text: {tweet}')
            print(f'Sentiment Classification: {prediction["classification"]}')
            print(f"Model's Prediction Score: {prediction['score']}")
            with open(self.outfile, 'a', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([status.created_at, tweet, prediction['classification'], prediction['score']])
                # file_data = str(status.created_at) + ', ' + str(tweet) + ', ' + str(prediction["classification"]) + ', ' + str(prediction["score"])
                # f.write(file_data)
                # f.write("\n")

        except KeyError:
            pass

    # def on_data(self, data):
    #     data = json.loads(data)
    #     try:
    #         try:
    #             text = data['extended_tweet']['full_text']
    #         except KeyError:
    #             text = data['full_text']
    #         except:
    #             text = data['text']
    #         prediction = self.model.classify(text)
    #         print(f'Text {text:{140}}')
    #         print(f'Sentiment Classification: {prediction}')
    #     except BaseException as err:
    #         print("Error on prediction : {}".format(err))

    def on_error(self, status):
        print("Error : ", status)
