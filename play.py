from asl_data import AslDb
from my_model_selectors import (
    SelectorConstant, SelectorBIC, SelectorDIC, SelectorCV,
)
from my_recognizer import recognize
from asl_utils import show_errors

asl = AslDb() # initializes the database

df_means = asl.df.groupby('speaker').mean()
df_stds = asl.df.groupby('speaker').std()

# Features
asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])
asl.df['left-y-mean']= asl.df['speaker'].map(df_means['left-y'])
asl.df['right-x-mean']= asl.df['speaker'].map(df_means['right-x'])
asl.df['right-y-mean']= asl.df['speaker'].map(df_means['right-y'])

asl.df['left-x-std']= asl.df['speaker'].map(df_stds['left-x'])
asl.df['left-y-std']= asl.df['speaker'].map(df_stds['left-y'])
asl.df['right-x-std']= asl.df['speaker'].map(df_stds['right-x'])
asl.df['right-y-std']= asl.df['speaker'].map(df_stds['right-y'])

features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']

asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

df_means = asl.df.groupby('speaker').mean()
df_stds = asl.df.groupby('speaker').std()

asl.df['grnd-rx-mean'] = asl.df['speaker'].map(df_means['grnd-rx'])
asl.df['grnd-lx-mean'] = asl.df['speaker'].map(df_means['grnd-lx'])
asl.df['grnd-ry-mean'] = asl.df['speaker'].map(df_means['grnd-ry'])
asl.df['grnd-ly-mean'] = asl.df['speaker'].map(df_means['grnd-ly'])

asl.df['grnd-rx-std']= asl.df['speaker'].map(df_stds['grnd-rx'])
asl.df['grnd-lx-std']= asl.df['speaker'].map(df_stds['grnd-lx'])
asl.df['grnd-ry-std']= asl.df['speaker'].map(df_stds['grnd-ry'])
asl.df['grnd-ly-std']= asl.df['speaker'].map(df_stds['grnd-ly'])

features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']

asl.df['norm-rx'] = (asl.df['right-x'] - asl.df['right-x-mean']) / asl.df['right-x-std']
asl.df['norm-ry'] = (asl.df['right-y'] - asl.df['right-y-mean']) / asl.df['right-y-std']
asl.df['norm-lx'] = (asl.df['left-x'] - asl.df['left-x-mean']) / asl.df['left-x-std']
asl.df['norm-ly'] = (asl.df['left-y'] - asl.df['left-y-mean']) / asl.df['left-y-std']

features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']

asl.df['delta-rx'] = asl.df['right-x'].diff().fillna(method='bfill')
asl.df['delta-ry'] = asl.df['right-y'].diff().fillna(method='bfill')
asl.df['delta-lx'] = asl.df['left-x'].diff().fillna(method='bfill')
asl.df['delta-ly'] = asl.df['left-y'].diff().fillna(method='bfill')

features_custom = ['delta-grnd-rx', 'delta-grnd-ry', 'delta-grnd-lx', 'delta-grnd-ly']

# asl.df['delta-grnd-rx'] = asl.df['grnd-rx'].diff().fillna(method='bfill')
# asl.df['delta-grnd-ry'] = asl.df['grnd-ry'].diff().fillna(method='bfill')
# asl.df['delta-grnd-lx'] = asl.df['grnd-lx'].diff().fillna(method='bfill')
# asl.df['delta-grnd-ly'] = asl.df['grnd-ly'].diff().fillna(method='bfill')

asl.df['delta-grnd-rx'] = (asl.df['grnd-rx'] - asl.df['grnd-rx-mean']) / asl.df['grnd-rx-std']
asl.df['delta-grnd-ry'] = (asl.df['grnd-ry'] - asl.df['grnd-ry-mean']) / asl.df['grnd-ry-std']
asl.df['delta-grnd-lx'] = (asl.df['grnd-lx'] - asl.df['grnd-lx-mean']) / asl.df['grnd-lx-std']
asl.df['delta-grnd-ly'] = (asl.df['grnd-ry'] - asl.df['grnd-ly-mean']) / asl.df['grnd-ly-std']

# Choose a feature set and model selector
features = features_custom # change as needed
model_selector = SelectorCV # change as needed

def train_all_words(features, model_selector):
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word, 
                        n_constant=3).select()
        model_dict[word]=model
    return model_dict

# Recognize the test set and display the result with the show_errors method
models = train_all_words(features, model_selector)
test_set = asl.build_test(features)
probabilities, guesses = recognize(models, test_set)
show_errors(guesses, test_set)
