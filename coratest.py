import math
import numpy
import torch
import torch.linalg

from torch import nn

from datetime import datetime

from sklearn.metrics import roc_auc_score


def print_now():
    now = datetime.now()
    print(now.strftime("%b-%d-%Y %H:%M:%S"))


# Standard accuracy calculation. Nothing fancy like ROCAUC or F1 just percent accurate.
# Doesn't return 0 to 100 but rather 0 to 1.
def accuracy(labels, outputs):
    argmax = torch.argmax(outputs, dim=1)
    correct = 0.0
    count = float(len(outputs))
    for x in range(0, len(outputs)):
        if labels[x][argmax[x]] == 1:
            correct += 1
    return correct / count


# First we create the feature vectors that will be used for training and testing.
print("Creating feature vectors")
print_now()
start_time = datetime.now()
# Change to alter the number of times the training is done. The precalculation of features only needs done once.
# This repeats the training after the precalculation to ensure things are consistent.
runs = 10

content = open("cora/cora.content")
cites = open("cora/cora.cites")

content_lines = content.readlines()
cites_lines = cites.readlines()

content_indices = {}
content_features = []
content_labels = []

adjacency = None


# This creates the label vectors. These are all one hot based on the provided text class.
for x in range(0, len(content_lines)):
    current_content = content_lines[x].split()
    paper_id = int(current_content[0])
    content_indices[paper_id] = x
    classification = current_content[-1]
    current_labels = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    features = []

    for y in range(1, len(current_content) - 1):
        features.append(float(current_content[y]))

    if classification == "Neural_Networks":
        current_labels[0] = 1.0
    elif classification == "Rule_Learning":
        current_labels[1] = 1.0
    elif classification == "Reinforcement_Learning":
        current_labels[2] = 1.0
    elif classification == "Probabilistic_Methods":
        current_labels[3] = 1.0
    elif classification == "Theory":
        current_labels[4] = 1.0
    elif classification == "Genetic_Algorithms":
        current_labels[5] = 1.0
    elif classification == "Case_Based":
        current_labels[6] = 1.0
    else:
        print(classification)

    content_features.append(features)
    content_labels.append(current_labels)

adjacency = numpy.zeros((len(content_features), len(content_features)))
degree = numpy.zeros((len(content_features), len(content_features)))
degree_inverse_square = numpy.zeros((len(content_features), len(content_features)))


# Calculate the adjacency matrices. These are undirected. Also calculate the degree matrices as that will
# matter for the normalized adjacency.
for x in range(0, len(cites_lines)):
    current_cite = cites_lines[x].split()
    paper_0 = int(current_cite[0])
    paper_1 = int(current_cite[1])
    adjacency[content_indices[paper_0], content_indices[paper_1]] = 1
    adjacency[content_indices[paper_1], content_indices[paper_0]] = 1
    degree[content_indices[paper_0], content_indices[paper_0]] += 1
    degree[content_indices[paper_1], content_indices[paper_1]] += 1


# Add self-references and then calculate the inverse square root degrees.
for x in range(0, len(content_lines)):
    adjacency[x, x] = 1
    degree[x, x] += 1
    degree_inverse_square[x, x] = 1 / math.sqrt(degree[x, x])


# Calculate normalized adjacency matrices. The more you do this the more hops you get through the graph.
# Reduce or increase the number of times this is done to change the number of hops this will do through the graph.
# Uncomment related lines below to make it actually do things. Yes I know there are better ways to do this
# but this is ugly test code to make sure the algorithm works.
# Uncomment the lines appropriate to each hop here and below to make it go further. Comment them out to
# make it not hop as far.
normalized_adjacency_matrix_1 = numpy.dot(numpy.dot(degree_inverse_square, adjacency), degree_inverse_square)
normalized_adjacency_matrix_2 = numpy.dot(normalized_adjacency_matrix_1, normalized_adjacency_matrix_1)
#normalized_adjacency_matrix_3 = numpy.dot(normalized_adjacency_matrix_2, normalized_adjacency_matrix_1)
#normalized_adjacency_matrix_4 = numpy.dot(normalized_adjacency_matrix_3, normalized_adjacency_matrix_1)
#normalized_adjacency_matrix_5 = numpy.dot(normalized_adjacency_matrix_4, normalized_adjacency_matrix_1)
#normalized_adjacency_matrix_6 = numpy.dot(normalized_adjacency_matrix_5, normalized_adjacency_matrix_1)
#normalized_adjacency_matrix_7 = numpy.dot(normalized_adjacency_matrix_6, normalized_adjacency_matrix_1)

training_features = []
training_labels = []

test_features = []
test_labels = []

accuracies = []
training_times = []


# This is where the magic happens. Calculate new concatenated feature vectors for each node.
# The end result is the unmodified node with information gathered from the nodes connected through the graph
# concatenated to the unmodified data.
for x in range(0, len(content_lines)):
    new_feature_vector = content_features[x].copy()
    # Each piece is a deeper gathering of information.
    convolved_piece_1 = numpy.zeros((len(new_feature_vector)))
    convolved_piece_2 = numpy.zeros((len(new_feature_vector)))
    #convolved_piece_3 = numpy.zeros((len(new_feature_vector)))
    #convolved_piece_4 = numpy.zeros((len(new_feature_vector)))
    #convolved_piece_5 = numpy.zeros((len(new_feature_vector)))
    #convolved_piece_6 = numpy.zeros((len(new_feature_vector)))
    #convolved_piece_7 = numpy.zeros((len(new_feature_vector)))

    # This ends up creating a new set of features that will be concatenated. They are based on weighted averages
    # gathered from connected nodes. Each piece is up to one hop deeper. Note that this does not add the node we
    # are gathering information for.
    for y in range(0, len(content_lines)):
        if x == y:
            continue
        convolved_piece_1 = numpy.add(convolved_piece_1, numpy.multiply(normalized_adjacency_matrix_1[x, y], content_features[y]))
        convolved_piece_2 = numpy.add(convolved_piece_2, numpy.multiply(normalized_adjacency_matrix_2[x, y], content_features[y]))
        #convolved_piece_3 = numpy.add(convolved_piece_3, numpy.multiply(normalized_adjacency_matrix_3[x, y], content_features[y]))
        #convolved_piece_4 = numpy.add(convolved_piece_4, numpy.multiply(normalized_adjacency_matrix_4[x, y], content_features[y]))
        #convolved_piece_5 = numpy.add(convolved_piece_5, numpy.multiply(normalized_adjacency_matrix_5[x, y], content_features[y]))
        #convolved_piece_6 = numpy.add(convolved_piece_6, numpy.multiply(normalized_adjacency_matrix_6[x, y], content_features[y]))
        #convolved_piece_7 = numpy.add(convolved_piece_7, numpy.multiply(normalized_adjacency_matrix_7[x, y], content_features[y]))

    # Create the new vector. Concatenate all the pieces that were calculated to get all the hops actually in the data.
    # While commenting out or uncommenting pieces they also need to be added here.
    new_feature_vector = numpy.concatenate((new_feature_vector, convolved_piece_1, convolved_piece_2))

    # Split things into a training set and a test set. This is a 60/40 split.
    if x % 5 == 0 or x % 5 == 3:
        test_features.append(new_feature_vector.tolist())
        test_labels.append(content_labels[x])
    else:
        training_features.append(new_feature_vector.tolist())
        training_labels.append(content_labels[x])

features_length = len(training_features[0])

training_features = torch.as_tensor(training_features)
training_labels = torch.as_tensor(training_labels)

test_features = torch.as_tensor(test_features)
test_labels = torch.as_tensor(test_labels)

print("Time to create features:", (datetime.now() - start_time).total_seconds())


# Basic one layer classifier. Nothing terribly exciting.
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.05)
        self.classifier = nn.Linear(features_length, 7)
        self.activation = nn.ReLU()

    def forward(self, input):
        x = self.dropout(input)
        x = self.activation(self.classifier(x))
        return torch.softmax(x, dim=1)


# Now do the actual training. Pretty simple. Train the single layer linear model.
for run in range(0, runs):
    start_time = datetime.now()
    model = Classifier()
    model.train()
    print(model)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.2)

    for epoch in range(0, 150):
        prediction = model(training_features)
        loss = loss_function(prediction, training_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 125:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

        if epoch % 25 == 0:
            print_now()
            print("Epoch:", epoch)
            print("Loss:", loss)
            print("Train ROCAUC:", roc_auc_score(training_labels, prediction.detach().numpy()))
            print("Train Accuracy:", accuracy(training_labels, prediction))

            test_results = model(test_features)

            print("Test ROCAUC:", roc_auc_score(test_labels, test_results.detach().numpy()))
            print("Test Accuracy:", accuracy(test_labels, test_results))

    model.eval()

    train_results = model(training_features)
    test_results = model(test_features)

    print()

    print("Final Train ROCAUC:", roc_auc_score(training_labels, train_results.detach().numpy()))
    print("Final Test ROCAUC:", roc_auc_score(test_labels, test_results.detach().numpy()))

    print()

    this_accuracy = accuracy(test_labels, test_results)

    print("Final Train Accuracy:", accuracy(training_labels, train_results))
    print("Final Test Accuracy:", this_accuracy)

    accuracies.append(this_accuracy)

    training_time = (datetime.now() - start_time).total_seconds()

    print()
    print("Training Time:", training_time)
    print()

    training_times.append(training_time)


print("Accuracy:", accuracies)
print("Training Times:", training_times)
print("Training Size:", len(training_labels))
print("Test Size:", len(test_labels))
# This should be manually changed to match what was done.
print("K = 2, Learn Rate = 0.2, 0.02 for epochs >= 125, Epochs = 150")
