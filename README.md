submit50 slug
CS50
submit John-DAcquisto-3416/projects

python
def player(board):
    """Returns the player who has the next turn (X or O)."""
    X_count = sum(row.count("X") for row in board)
    O_count = sum(row.count("O") for row in board)
    return "X" if X_count == O_count else "O"
2. Actions Function
Find all available moves.

python
def actions(board):
    """Returns a set of possible actions (i, j) for the given board."""
    return {(i, j) for i in range(3) for j in range(3) if board[i][j] is None}
3. Result Function
Return the new board after making a move.

python
import copy

def result(board, action):
    """Returns a new board state after applying action, without modifying the original board."""
    if action not in actions(board):
        raise Exception("Invalid move")

  4. Winner Function
Check for a winner.

python
def winner(board):
    """Returns the winner (X or O) if there is one, else None."""
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] is not None:
            return board[i][0]  # Horizontal win
        if board[0][i] == board[1][i] == board[2][i] and board[0][i] is not None:
            return board[0][i]  # Vertical win

5. Terminal Function
Check if the game is over.

python
def terminal(board):
    """Returns True if the game is over, else False."""
    return winner(board) is not None or all(cell is not None for row in board for cell in row)
6. Utility Function
Assign values to terminal states.

python
def utility(board):
    """Returns the utility value: 1 if X wins, -1 if O wins, 0 if tie."""
    game_winner = winner(board)
    return 1 if game_winner == "X" else -1 if game_winner == "O" else 0
7. Minimax Function
Find the optimal move.

python
def minimax(board):
    """Returns the optimal move (i, j) using Minimax algorithm."""
    if terminal(board):
        return None

   deque

def shortest_path(source, target):
    """Return the shortest path from source to target using BFS."""
    # Initialize queue with starting node (source)
    queue = deque([[(None, source)]])  # Path format: [(movie_id, person_id)]
    explored = set()

 rom kanren import var, Relation, facts, run

knight = Relation()
kn Relation()

factsave ("A",)) # Since the statement is false, A must be knave.

facts(kn, ("A",)) A is a kn, making his statement false.
facts(knight, ("B",)) # "we are both knaves" is false, B must be a knight.

facts(knave, ("A",)) # Since A is lying, they be different types.
facts(knight, ("B",)) # B tells the truth that they are different kinds.

facts(knight, ("A",)) # Since C truthfully states A is a knight.
facts(knave, ("B",)) # B lies about A saying, "I am a kn."
facts(knight, ("C",)) # C tells the truth.

x = var()
print("ights:", run(0, x, knight(xprint("Kn:", run(0, x, knave(x)))

from kanren import var, Relation, facts, run

knight = Relation()
knave = Relation()

facts(knave, ("A",))
facts(knave, ("A",))
facts(knight, ("B",))
facts(knave, ("A",))
facts(knight, ("B",))
facts(knight, ("A",))
facts(knave, ("B",))
facts(knight, ("C",))

x = var()
print("Knights:", run(0, x, knight(x)))
print("Knaves:", run(0, x, knave(x)))

PROBS = {
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },
    "trait": {
        2: {True: 0.65, False: 0.35},
        1: {True: 0.56, False: 0.44},
        0: {True: 0.01, False: 0.99}
    },
    "mutation": 0.01
}

def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute the joint probability of a given configuration of gene copies and exhibited traits.
    """
    probability = 1.0

   
2. update
This function updates the probability distributions based on the computed joint probability.

python
def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add the new joint probability to existing probability distributions.
    """
    for person in probabilities:
        genes = (2 if person in two_genes else 1 if person in one_gene else 0)
        probabilities[person]["gene"][genes] += p
        probabilities[person]["trait"][person in have_trait] += p
3. normalize
This function normalizes probabilities so they sum to 1.

python
def normalize(probabilities):
    """
    Normalize probability distributions so they sum to 1.
    """
    for person in probabilities:
        for field in ["gene", "trait"]:
            total = sum(probabilities[person][field].values())
            for key in probabilities[person][field]:
                probabilities[person][field][key] /= total

  
   
2. sample_pagerank
This function estimates PageRank using random sampling.

python
def sample_pagerank(corpus, damping_factor, n):
    """
    Uses a random surfer model to estimate PageRank values.
    """
    all_pages = list(corpus.keys())
    page_rank = {page: 0 for page in all_pages}

    
3. iterate_pagerankities = transition_model(corpus, current_page, damping_factor)
        current_page = random.choices(list(probabilities.keys()), weights=probabilities.values())[0]

    # Normalize PageRank values
    total_samples = sum(page_rank.values())
This function implements the iterative PageRank formula until convergence.

python
def iterate_pagerank(corpus, damping_factor, tolerance=0.001):
    """
    Computes PageRank values iteratively until no value changes by more than `tolerance`.
    """
    num_pages = len(corpus)
    ranks = {page: 1 / num_pages for page in corpus}  # Initial rank distribution
    
   class CrosswordCreator:

    def enforce_node_consistency(self):
        """Ensure each variable's domain contains only values of correct length."""
        for var in self.domains:
            self.domains[var] = {word for word in self.domains[var] if len(word) == var.length}

    def revise(self, x, y):
        """Make variable x arc consistent with y by removing inconsistent values."""
        revised = False
        overlap = self.crossword.overlaps.get((x, y))
        if overlap is None:
            return revised
        
        i, j = overlap
        to_remove = {word_x for word_x in self.domains[x] if not any(
            word_y for word_y in self.domains[y] if word_x[i] == word_y[j]
        )}
        
        if to_remove:
            self.domains[x] -= to_remove
            revised = True

        return revised

    def ac3(self, arcs=None):
        """Enforce arc consistency using AC3 algorithm."""
        queue = arcs if arcs is not None else [(x, y) for x in self.domains for y in self.crossword.neighbors(x)]
        while queue:
            x, y = queue.pop(0)
            if self.revise(x, y):
                if not self.domains[x]:
                    return False
                queue.extend([(z, x) for z in self.crossword.neighbors(x) if z != y])
        return True

    def assignment_complete(self, assignment):
        """Return True if all variables have assigned values."""
        return len(assignment) == len(self.domains)

    def consistent(self, assignment):
        """Return True if assignment satisfies constraints."""
        words = set(assignment.values())
        if len(words) != len(assignment):  # Ensure uniqueness
            return False
        
        for var, word in assignment.items():
            if len(word) != var.length:  # Check word length
                return False
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    overlap = self.crossword.overlaps[(var, neighbor)]
                    if overlap and assignment[var][overlap[0]] != assignment[neighbor][overlap[1]]:
                        return False
        return True

    def order_domain_values(self, var, assignment):
        """Return domain values ordered by least-constraining heuristic."""
        return sorted(self.domains[var], key=lambda word: sum(
            1 for neighbor in self.crossword.neighbors(var) if neighbor not in assignment
            for neighbor_word in self.domains[neighbor] if word[self.crossword.overlaps[(var, neighbor)][0]] != neighbor_word[self.crossword.overlaps[(var, neighbor)][1]]
        ))

    def select_unassigned_variable(self, assignment):
        """Select the next variable using MRV and degree heuristics."""
        unassigned = [var for var in self.domains if var not in assignment]
        return min(unassigned, key=lambda var: (len(self.domains[var]), -len(self.crossword.neighbors(var))))

    def backtrack(self, assignment):
        """Perform backtracking search to find a valid crossword solution."""
        if self.assignment_complete(assignment):
            return assignment
        
        var = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(var, assignment):
            new_assignment = assignment.copy()
            new_assignment[var] = value
            if self.consistent(new_assignment):
                result = self.backtrack(new_assignment)
                if result:
                    return result
        
        return None
        
        import csv
from sklearn.neighbors import KNeighborsClassifier

def load_data(filename):
    """Load shopping data from CSV file and return evidence and labels."""
    months = {
        "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5,
        "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
    }
    
    evidence, labels = [], []

    with open(filename, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            evidence.append([
                int(row[0]),  # Administrative
                float(row[1]),  # Administrative_Duration
                int(row[2]),  # Informational
                float(row[3]),  # Informational_Duration
                int(row[4]),  # ProductRelated
                float(row[5]),  # ProductRelated_Duration
                float(row[6]),  # BounceRates
                float(row[7]),  # ExitRates
                float(row[8]),  # PageValues
                float(row[9]),  # SpecialDay
                months[row[10]],  # Month (converted to int)
                int(row[11]),  # OperatingSystems
                int(row[12]),  # Browser
                int(row[13]),  # Region
                int(row[14]),  # TrafficType
                1 if row[15] == "Returning_Visitor" else 0,  # VisitorType
                1 if row[16] == "TRUE" else 0  # Weekend
            ])
            labels.append(1 if row[17] == "TRUE" else 0)

    return evidence, labels

def train_model(evidence, labels):
    """Train k-nearest-neighbor classifier using given evidence and labels."""
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model

def evaluate(labels, predictions):
    """Evaluate performance by computing sensitivity and specificity."""
    true_positive = sum(1 for actual, predicted in zip(labels, predictions) if actual == 1 and predicted == 1)
    false_negative = sum(1 for actual, predicted in zip(labels, predictions) if actual == 1 and predicted == 0)
    true_negative = sum(1 for actual, predicted in zip(labels, predictions) if actual == 0 and predicted == 0)
    false_positive = sum(1 for actual, predicted in zip(labels, predictions) if actual == 0 and predicted == 1)

    import random

class NimAI:
    def get_q_value(self, state, action):
        """Return the Q-value for a given state-action pair."""
        return self.q.get((tuple(state), action), 0)

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """Update Q-value using the Q-learning formula."""
        new_value = reward + future_rewards
        self.q[(tuple(state), action)] = old_q + self.alpha * (new_value - old_q)

    def best_future_reward(self, state):
        """Return the best possible reward for any available action in the given state."""
        possible_actions = [(i, j) for i, pile in enumerate(state) for j in range(1, pile + 1)]
        if not possible_actions:
            return 0
        return max(self.get_q_value(state, action) for action in possible_actions)

    def choose_action(self, state, epsilon=False):
        """Choose action using an epsilon-greedy approach or best action."""
        possible_actions = [(i, j) for i, pile in enumerate(state) for j in range(1, pile + 1)]
        if not possible_actions:
            return None

        if epsilon and random.random() < self.epsilon:
            return random.choice(possible_actions)

        return max(possible_actions, key=lambda action: self.get_q_value(state, action), default=None)
        
        import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def load_data(data_dir):
    """Load image data and labels from the given directory."""
    images, labels = [], []
    
    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))
        for filename in os.listdir(category_path):
            image_path = os.path.join(category_path, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))  # Resize image
            images.append(image)
            labels.append(category)
    
    return np.array(images), np.array(labels)

def get_model():
    """Return a compiled neural network model."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
    
    1. get_mask_token_index
Retrieve the index of the mask token in a tokenized input sequence.

def get_mask_token_index(mask_token_id, inputs):
    """Return the index of the mask token in the input sequence."""
    try:
        return inputs.input_ids.index(mask_token_id)
    except ValueError:
        return None
        
        2. get_color_for_attention_score
Map an attention score (0 to 1) to an RGB grayscale color.

def get_color_for_attention_score(attention_score):
    """Convert attention score into grayscale RGB values."""
    shade = int(round(255 * attention_score))
    return (shade, shade, shade)
    
    3. visualize_attentions
Generate diagrams for all attention heads across layers.

def visualize_attentions(tokens, attentions):
    """Generate diagrams for all attention heads across layers."""
    num_layers = len(attentions)
    num_heads = attentions[0].shape[2]

    for layer_index in range(num_layers):
        for head_index in range(num_heads):
            generate_diagram(layer_index + 1, head_index + 1, tokens, attentions[layer_index][0][head_index])
            
            Analysis (analysis.md)
Layer 2, Head 5: Seems to focus on noun-adjective relationships—adjectives consistently attend to the nouns they describe.

import nltk
from nltk.tokenize import word_tokenize

def preprocess(sentence):
    """Tokenize and preprocess sentence."""
    words = word_tokenize(sentence.lower())  # Convert to lowercase and tokenize
    return [word for word in words if any(c.isalpha() for c in word)]
    
    Content Fix:

NONTERMINALS = """
S -> NP VP | S PP
NP -> Det N | Det Adj N | NP PP
VP -> V NP | V PP | VP Adv
PP -> P NP
Det -> 'the' | 'a' | 'an'
N -> 'dog' | 'cat' | 'home' | 'park' | 'boy' | 'girl'
Adj -> 'big' | 'small' | 'happy' | 'sad'
V -> 'chased' | 'saw' | 'loved' | 'walked'
Adv -> 'quickly' | 'slowly'
P -> 'in' | 'on' | 'with' | 'by'
"""
NP Chunk Function
from nltk.tree import Tree

def np_chunk(tree):
    """Extract noun phrase (NP) chunks that do not contain nested NPs."""
    return [subtree for subtree in tree.subtrees(lambda t: t.label() == "NP" and not any(child.label() == "NP" for child in subtree))]
    
    nltk
Holmes sat in the red armchair

        S
   _____|______
  NP         VP
  |          |______
  N          V      PP
  |          |      |
holmes     sat     P NP
                   |  |_____
                   in Det  N
                       |    |
                      the red armchair
