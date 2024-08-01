"this function split the data in train and test set with a given percentage"


def train_test_split(data, ratio):
    n = len(data)
    train_size = int(n * ratio)
    train = data[:train_size]
    test = data[train_size:]
    return train, test

if __name__ == "__main__":
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ratio = 0.8
    train, test = train_test_split(data, ratio)
    print(train)
    print(test)