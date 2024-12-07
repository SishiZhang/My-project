# load raw data
import pandas as pd
import numpy as np
import xgboost as xgb 
from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



abalone = fetch_ucirepo(id=1) 
  
X = abalone.data.features 
y = abalone.data.targets 
  


# data preprocessed
import pandas as pd
def classify_count(val):
    if val <= 7:
        return 'class 1'
    elif 7 < val <= 10:
        return 'class 2'
    elif 10 < val <= 15:
        return 'class 3'
    else:
        return 'class 4'
def classification(y):
    y = y.squeeze()
    y_classification = y.apply(classify_count)
    return pd.Series(y_classification, index = y.index, name = 'Rings')

y_classification = classification(y)
class_counts = y_classification.value_counts()
# print(y_classification)
# print(class_counts)

X['Sex'] = X['Sex'].replace({'M': 1, 'F': -1, 'I': 0})



# distribution of class

plt.figure(figsize=(8, 8)) 
plt.pie(
    class_counts, 
    labels=class_counts.index,                      
    autopct='%1.1f%%',                              
    colors=['skyblue', 'lightgreen', 'lightcoral', 'peachpuff'],  
    startangle=60                                 
)
plt.title("Distribution of class") 
# plt.savefig("/Users/sishizhang/Desktop/abalone-age-prediction/pie_chart.png", format="png", dpi=300)                
plt.show()                                         


# Distribution of features

for column in X.columns:
    plt.figure(figsize=(10, 8))
    sns.histplot(X[column], kde = True, color = 'blue')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    # plt.savefig(f"/Users/sishizhang/Desktop/abalone-age-prediction/histogram_{column}.png", format="png", dpi=300) 
    plt.show()


data = pd.concat([X, y_classification], axis = 1)
# print(data.info())

# Factor object col
data_numeric = data.copy()
for col in data.select_dtypes(include=['object']).columns:
    data_numeric[col] = pd.factorize(data[col])[0]

corr_matrix = data_numeric.corr()

# mapping heatmap 
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap between Features and Ring age classified')
# plt.savefig("/Users/sishizhang/Desktop/abalone-age-prediction/Heatmap.png", format="png", dpi=300) 
plt.show()


def calculate_chi_square(feature):
    contingency_table = pd.crosstab(feature, y_classification)
    chi_square_value = chi2_contingency(contingency_table)[0]
    return chi_square_value

chi_square_values = X.apply(calculate_chi_square)

chi_square_df = pd.DataFrame({'Feature': X.columns, 'Chi-square Value': chi_square_values})

chi_square_df= chi_square_df.sort_values(by='Chi-square Value', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Feature', y='Chi-square Value', data=chi_square_df, palette='coolwarm')
plt.title('Association between Features and Ring age classified')
for index, value in enumerate(chi_square_df['Chi-square Value']):
    plt.text(index, value + 200, f'{value:.0f}', ha='center')
# plt.savefig("/Users/sishizhang/Desktop/abalone-age-prediction/chi_square bar.png", format="png", dpi=300) 
plt.show()






# decision tree

best_model_info = {"accuracy": 0, "model": None, "run": None}

for run in range(7):
    X_train, X_test, y_train, y_test = train_test_split(X, y_classification, test_size=0.3, random_state = run)

    model = DecisionTreeClassifier(max_depth = run + 2, random_state = run)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print(f"Run {run + 1}: Train Accuracy = {train_accuracy:.3f}, Test Accuracy = {test_accuracy:.3f}")

    if test_accuracy > best_model_info["accuracy"]:
        best_model_info.update({"accuracy": test_accuracy, "model": model, "run": run + 1})


print(f"\nBest Run: {best_model_info['run']}, Test Accuracy: {best_model_info['accuracy']:.3f}")


best_model = best_model_info["model"]

plt.figure(figsize=(20, 10)) 
plot_tree(
    best_model,
    feature_names=X.columns,  
    class_names=["Class 1", "Class 2", "Class 3", "Class 4"], 
    filled=True,  
    rounded=True,  
    fontsize=10  
)
plt.title(f"Best Decision Tree (Run {best_model_info['run']})", fontsize=16)
# plt.savefig("/Users/sishizhang/Desktop/abalone-age-prediction/best_decision_tree.png", dpi=300, bbox_inches="tight")  
plt.show()


# IF THEN RULES
output_file = "IF_THEN_rules.txt"

with open(output_file, "w") as file:
    tree = best_model.tree_
    total_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold

    current_path = []
    for node_id in range(total_nodes):
        if children_left[node_id] != children_right[node_id]:  
            condition = f"{X.columns[feature[node_id]]} <= {threshold[node_id]:.3f}"
            current_path.append(condition)
        else:  
            if current_path:
                class_label = tree.value[node_id].argmax() + 1
                rule = f"IF {' AND '.join(current_path)} THEN Class {class_label}"
                print(rule)  # 
                file.write(rule + "\n")  
            if current_path:
                current_path.pop()  




path = best_model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
postpruned_models, postpruned_accuracies = [], []

for ccp_alpha in ccp_alphas:
    postpruned_model = DecisionTreeClassifier(random_state = 0, ccp_alpha = ccp_alpha)
    postpruned_model.fit(X_train, y_train)
    pruned_accuracy = postpruned_model.score(X_test, y_test)
    postpruned_models.append(postpruned_model)
    postpruned_accuracies.append(pruned_accuracy)


fig, ax = plt.subplots()
ax.plot(ccp_alphas, postpruned_accuracies, marker = 'o', drawstyle = 'steps-post')
ax.set_xlabel('Effective Alpha')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vs Alpha for postPruned Decision Trees')
# plt.savefig("/Users/sishizhang/Desktop/abalone-age-prediction/Accuracy After postprunning.png", format="png", dpi=300) 
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y_classification, test_size = 0.3, random_state = 42)

n_estimators = list(range(10,770, 50))
def accuracy_model(model_class, **kwargs):
    accuracies = []
    for n_estimator in n_estimators:
        model = model_class(n_estimators = n_estimator, **kwargs)
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred_test))
    return accuracies
rf_results = accuracy_model(RandomForestClassifier, random_state=42)

plt.figure(figsize=(10, 8))
plt.plot(n_estimators, rf_results, marker = 's', linestyle = '-', color = 'b')
plt.title("Random Forest Performance vs Number of Trees")
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
# plt.savefig("/Users/sishizhang/Desktop/abalone-age-prediction/Random Forests.png", format="png", dpi=300) 
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y_classification, test_size = 0.3, random_state = 42)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
n_estimators = list(range(10,770, 50))

rf_results = accuracy_model(RandomForestClassifier, random_state=42)
gb_results = accuracy_model(GradientBoostingClassifier, random_state=42)
xgb_results = accuracy_model(xgb.XGBClassifier, max_depth=6, learning_rate=0.01, random_state=42)


plt.figure(figsize = (12, 8))
plt.plot(n_estimators, rf_results, marker = 'D', linestyle = '-', color = 'y', label = 'Random Forest')
plt.plot(n_estimators, gb_results, marker = 's', linestyle = '-', color = 'b', label = 'Gradient Boosting')
plt.plot(n_estimators, xgb_results, marker = 'o', linestyle = '--', color = 'r', label = 'XGBoost')
plt.title("Model Comparison: Random Forest, Gradient Boosting and GBoost")
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.legend()
# plt.savefig("/Users/sishizhang/Desktop/abalone-age-prediction/comparison of RF, GB and XGB.png", format="png", dpi=300) 
plt.show()


mlp_accuracies = []
for solver in ['adam', 'sgd']:
    model = MLPClassifier(solver=solver, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlp_accuracies.append(accuracy)
plt.figure(figsize=(12, 8))
plt.plot(n_estimators, rf_results, marker = 'D', linestyle = '-', color = 'y', label = 'Random Forest')
plt.plot(n_estimators, gb_results, marker = 's', linestyle = '-', color = 'b', label = 'Gradient Boosting')
plt.plot(n_estimators, xgb_results, marker = 'o', linestyle = '--', color = 'r', label = 'XGBoost')
plt.axhline(y=mlp_accuracies[0], color='g', linestyle='--', label='MLP (Adam)')
plt.axhline(y=mlp_accuracies[-1], color='c', linestyle='--', label='MLP (SGD)')

plt.xlabel('Number of Trees in the Ensemble')
plt.ylabel('Accuracy Score')
plt.title("Model Comparison: Random Forest, Gradient Boosting, GBoost and MLP")
plt.legend()
plt.tight_layout()
# plt.savefig("/Users/sishizhang/Desktop/abalone-age-prediction/Comparision-with-Adam-and-SGD.png", format="png", dpi=300) 
plt.show()


le = LabelEncoder()
y_encoded = le.fit_transform(y_classification)
X_train, X_value, y_train, y_value = train_test_split(
    X.values, y_encoded, test_size=0.3, random_state=42
)
X_train_tensor = torch.tensor(X.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_encoded, dtype=torch.long) 
X_value_tensor = torch.tensor(X_value, dtype=torch.float32)
y_value_tensor = torch.tensor(y_value, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
value_dataset = TensorDataset(X_value_tensor, y_value_tensor)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  
value_loader = DataLoader(value_dataset, batch_size=4, shuffle=False)

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 4)  
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



def train_and_evaluate(model, optimizer, criterion, train_loader, epochs=1000):
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in value_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)  
                val_correct += (predicted == y_batch).sum().item()  
                val_total += y_batch.size(0) 
        val_accuracy = val_correct / val_total
        # print(f"Epoch {epoch+1}/{epochs}: Val Accuracy: {val_accuracy:.2f}%")
        return  val_accuracy

hyperparameter_combinations = [{"dropout_rate": 0.1, "weight_decay": 0.01}, {"dropout_rate": 0.2, "weight_decay": 0.001}, {"dropout_rate": 0.3, "weight_decay": 0.0001}]

results = {}
input_size = X.shape[1]  
hidden_size = 40         

for i, params in enumerate(hyperparameter_combinations):
    # print(f"Training with dropout_rate={params['dropout_rate']} and weight_decay={params['weight_decay']}...")
    model = SimpleNN(input_size, hidden_size, params["dropout_rate"])
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=params["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    val_accuracy = train_and_evaluate(model, optimizer, criterion, train_loader, epochs=1000)

    results[f"HyperparameterCombination {i+1}"] = {
        "dropout_rate": params["dropout_rate"],
        "weight_decay": params["weight_decay"],
        "accuracy": val_accuracy,
    }


print("\nResults Summary:")
for key, value in results.items():
    print(f"{key}: {value}")
