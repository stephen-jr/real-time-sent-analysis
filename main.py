from classes import *

args = sys.argv
# try:
if args[1] == '--train':
    if args[2].startswith('models/'):
        model = Model()
        model.input_train_data(args[2])
        model.tokenize()
        model.create_model()
        plt_data = model.train()
        model.plot(plt_data)
        exit("Training Finished")
    else:
        exit('Select File from models folder')
elif args[1] == '--stream':
    if args[2].startswith('models/'):
        listener = StdOutListener(args[2])
        if args[3]:
            print("API Connected ............")
            print("Creating Stream.....")
            stream = Stream(listener.auth, listener, tweet_mode='extended')
            stream.filter(track=[args[3]])
        else:
            exit("Please Specify model's directory")
    else:
        exit("Specify model from the model directory")
else:
    exit("Specify a flag")


# except Exception as e:
#     exit(("Error in Main Function", e))
