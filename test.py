import pickle


def load_var(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

query = load_var('Query_fea_ORB')
gallery = load_var('Gallery_fea_ORB')
print(len(query))
print(len(gallery))

