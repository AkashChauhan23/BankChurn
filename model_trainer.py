import pickle
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
y_test = pd.read_csv('sample_submission.csv')
y_test = y_test.drop('id', axis = 1)

def del_col(col, data):
    data = data.drop(col, axis=1)
    return data

train_data = del_col(['id', 'CustomerId', 'Surname'], train_data)
test_data = del_col(['id', 'CustomerId', 'Surname'], test_data)

# print(train_data_clean.head(4))
# print(test_data_clean.head(4))

def pickle_saver(col, encoder):
    with open(col + '_encoder.pkl', 'wb') as file:
        pickle.dump(encoder, file)


def encoder(col, data, flag = False):
    if flag == True:
        col_encoder = LabelEncoder()
        data[col] = col_encoder.fit_transform(data[[col]])
    else:
        col_encoder = OneHotEncoder()
        col_name_encoder = col_encoder.fit_transform(data[[col]]).toarray()
        col_name_df = pd.DataFrame(col_name_encoder, columns=col_encoder.get_feature_names_out([col]))
        data = pd.concat([data.drop(col, axis=1), col_name_df], axis=1)

    pickle_saver(col, col_encoder)
    return data

train_data = encoder('Geography', train_data, flag=False)
train_data = encoder('Gender', train_data, flag=True)


x_test = encoder('Geography', test_data, flag=False)
x_test = encoder('Gender', x_test, flag=True)

x_train = train_data.drop('Exited', axis=1)
y_train = train_data['Exited']


scaller = StandardScaler()
x_train = scaller.fit_transform(x_train)
x_test = scaller.transform(x_test)

pickle_saver('Scaller', scaller)

model = Sequential([
    Input(shape=[x_train.shape[1],]),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid'),
])

opt = tf.keras.optimizer.Adam(learning_rate=0.001)

model.compile(optimizer = opt, loss = 'binary_crossentropy', metrices = ['accuracy'])

early_stop = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)

history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epocs=80, callbacks = [early_stop])

model.save('bankchurnprediction.keras')
