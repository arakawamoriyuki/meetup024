import os, pickle, io, csv, json

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from access_points import get_scanner


# sampling functions

def sampling(**kwargs):
    X, y = fetch_datasets(label=kwargs['label'], size=kwargs['size'])
    save_datasets(X, y, path=kwargs['datasets_path'])
    training(model_path=kwargs['model_path'], datasets_path=kwargs['datasets_path'])

def training(**kwargs):
    X, y = load_datasets(path=kwargs['datasets_path'])
    model = get_model(path=kwargs['model_path'])
    model.fit(X, y)
    save_model(model, path=kwargs['model_path'])

def predict(**kwargs):
    X, _ = fetch_datasets(size=1)
    model = get_model(path=kwargs['model_path'])
    result = model.predict(X[0])[0]
    print(result)
    return result

def percentage(**kwargs):
    X, _ = fetch_datasets(size=1)
    model = get_model(path=kwargs['model_path'])
    result = {x: y for x, y in zip(model.classes_, model.predict_proba(X[0])[0])}
    print(result)
    return result

def labels(**kwargs):
    model = get_model(path=kwargs['model_path'])
    result = model.classes_
    print(result)
    return result


# random forest model functions

def get_model(path='models/model.pkl'):
    if not os.path.exists(path):
        classifier = RandomForestClassifier(n_estimators=500, class_weight='balanced')
        model = make_pipeline(DictVectorizer(sparse=False), classifier)
        save_model(model, path=path)
    else:
        model = load_model(path=path)
    return model

def save_model(model, path='models/model.pkl'):
    with open(path, 'wb') as model_file:
        pickle.dump(model, model_file)

def load_model(path='models/model.pkl'):
    model_file = open(path, 'rb')
    model = pickle.load(model_file)
    return model


# datasets functions

def fetch_datasets(label='default place', size=1):
    X = []
    y = []
    wifi_scanner = get_scanner()
    for index in range(int(size)):
        print('scan ' + str(index) + ' access points')
        access_points = wifi_scanner.get_access_points()
        X.append({' '.join([ap.ssid, ap.bssid]): ap.quality for ap in access_points })
        y.append(label)
    return X, y

def save_datasets(X, y, path='datasets/datasets.csv'):
    csv_writer = csv.writer(
        io.open(path, 'a', encoding='utf-8'),
        delimiter='\t'
    )
    for Xi, yi in zip(X, y):
        csv_writer.writerow([yi, json.dumps(Xi)])

def load_datasets(path='datasets/datasets.csv'):
    csv_reader = csv.reader(
        io.open(path, 'r', encoding='utf-8'),
        delimiter='\t'
    )
    X = []
    y = []
    for line in csv_reader:
        y.append(line[0])
        X.append(json.loads(line[1]))
    return X, y


