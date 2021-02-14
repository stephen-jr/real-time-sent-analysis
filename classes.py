from _utils import *
from config import access_token, access_token_secret, consumer_key, consumer_secret


class Model:
    def __init__(self):
        self.train_dataset = None
        self.model = Sequential()
        self.tokenizer = Tokenizer(num_words=10000)
        self.X = self.Y = self.X_Test = self.X_Train = self.Y_Train = self.Y_Test = ''
        self.vocab_size, self.maxlen = 0,  140

    def input_train_data(self, dataset):
        print("======Inputting Train Dataset==========\n")
        dataset = root_dir(dataset)
        if dataset.split('.')[-1] == 'csv':
            try:
                self.train_dataset = pd.read_csv(dataset, error_bad_lines=False, encoding='ISO-8859-1')
            except Exception as e:
                exit("Error on loading CSV Dataset: " + str(e))
        elif dataset.split('.')[-1] == 'json':
            try:
                self.train_dataset = pd.read_json(dataset, lines=True)
            except Exception as e:
                print("Error on loading JSON Dataset")
                exit(e)
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
            exit(e)
        print("========= Dataset Input Complete ============\n")

    @staticmethod
    def preprocess(text):
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
        print("=========== Tokenizing Training and Validation Texts ===============\n")
        try:
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
            print("============Tokenization Complete=======================\n")
        except BaseException as e:
            exit(e)

    def create_model(self):
        print("=========== Creating Sentiment Analytic model =============\n")
        self.model.add(Embedding(self.vocab_size, 100, input_length=self.maxlen))
        self.model.add(LSTM(128, return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])
        self.model.summary()
        print("\n============ Model Build Complete ==============")

    def ld_model(self, pth):
        pth = root_dir(pth)
        self.model = load_model(pth)  # custom_objects={'precision_1': Precision(), 'recall_1': Recall()})
        f_path = root_dir('models/tokenizer.pkl')
        if os.path.exists(f_path):
            with open(f_path, 'rb') as f:
                self.tokenizer = dill.load(f)
        else:
            raise FileNotFoundError("Tokenizer file doesn't exist")

    def train(self):
        print("\n================= Model Training Process in Session===================== \n")
        es = EarlyStopping(monitor='val_loss', mode='min', patience=1, verbose=1)
        history = self.model.fit(self.X_Train,
                                 self.Y_Train,
                                 epochs=10,
                                 verbose=True,
                                 validation_data=(self.X_Test, self.Y_Test),
                                 batch_size=128,
                                 callbacks=[es])
        _, train_accuracy, train_precision, train_recall = self.model.evaluate(self.X_Train,
                                                                               self.Y_Train,
                                                                               verbose=True)
        train_fscore = f1_m(train_recall, train_precision)
        print("Training metrics : ")
        print(f'Accuracy : {train_accuracy:{.5}} \n'
              f'Precision : {train_precision:{.5}} \n'
              f'Recall : {train_recall:{5}} \n'
              f'F-Score : {train_fscore:{5}} \n')
        print("\n===================== Training Complete=========================== \n")
        print("=============== =Validation in progress ========================")
        _, test_accuracy, test_precision, test_recall = self.model.evaluate(self.X_Test,
                                                                            self.Y_Test,
                                                                            verbose=True)

        test_fscore = f1_m(test_precision, test_precision)
        print("\nTesting metrics")
        print(f'Accuracy : {test_accuracy:{5}} \n'
              f'Precision : {test_precision:{5}} \n'
              f'Recall : {test_recall:{5}} \n'
              f'F-Score : {test_fscore:{5}}\n')
        print("\n=================== Saving Model ===================")
        try:
            model_name = 'RNN-model@latest'
            if os.path.exists(root_dir('models/' + model_name)):
                os.remove(root_dir('models/' + model_name))
            self.model.save(root_dir(None) + '/models/' + model_name)
            if os.path.exists(root_dir('models/tokenizer.pkl')):
                os.remove(root_dir('models/tokenizer.pkl'))
            with open(root_dir('models/tokenizer.pkl'), 'wb') as f:
                dill.dump(self.tokenizer, f)
            print("\n=================== Save Complete ===================")
            return history
        except BaseException as e:
            print("Error Saving the model : ")
            exit(e)

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

    @staticmethod
    def plot(hist):
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


class StdOutListener(StreamListener):
    def __init__(self, filename):
        super().__init__()
        self.auth = OAuthHandler(consumer_key, consumer_secret)
        self.auth.set_access_token(access_token, access_token_secret)
        t = time.gmtime(time.time())
        self.outfile = "stream_data/" + filename + "-" + str(t[2]) + '-' + str(t[1]) + '-' + str(t[0]) + '.csv'
        with open(root_dir(self.outfile), 'w', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Created_at', 'Text', 'SentimentClassification', 'PolarityScore'])

        # if model_dir:
            try:
                self.model = Model()
                self.model.ld_model('models/RNN-model@latest')
            except Exception as e:
                print("StdOutListenerError : ")
                exit(e)
        # else:
        #     exit("Please Specify model's directory")

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
            print("\n===============================================")
            print(f'Text: {tweet}')
            print(f'Sentiment Classification: {prediction["classification"]}')
            print(f"Model's Prediction Score: {prediction['score']}")
            print('================================================')
            with open(self.outfile, 'a', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([status.created_at, tweet, prediction['classification'], prediction['score']])
        except BaseException as e:
            exit(e)

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
