from classes import *


# try:
def main(arg):
    _dir = os.path.abspath(os.path.dirname(__name__))
    if arg[1] == '--train' and arg[2]:
        model = Model()
        model.input_train_data(os.path.join(_dir, 'train_data\\' + arg[2]))
        model.tokenize()
        model.create_model()
        plt_data = model.train()
        model.plot(plt_data)
        exit()
    # else:
    #     exit('Select File from models folder')
    elif arg[1] == '--stream':
        # if arg[2].startswith('models/'):
        if arg[2]:
            listener = StdOutListener(arg[2])
            print("API Connected ............")
            print("Creating Stream.....")
            stream = Stream(listener.auth, listener, tweet_mode='extended')
            stream.filter(track=[arg[2]])
        else:
            exit("Please Specify model's")
        # else:
        #     exit("Specify model from the model directory")
    else:
        exit("Specify an action flag")


# except Exception as e:
#     exit(("Error in Main Function", e))

if __name__ == '__main__':
    args = sys.argv
    main(args)
