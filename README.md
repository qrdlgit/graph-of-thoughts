# graph-of-thoughts

I've been noodling this in my head for awhile, finding a way to 'autonomously' improve an ML algorithm with GPT4.  I was also seperately thinking about using alternative paths of COT (I even suggested it here, maybe not the best place, https://github.com/openai/evals/issues/968)

But the idea of combining the two tasks didn't come to me until reading this paper that just recently hit arxiv - "Tree of Thoughts" https://arxiv.org/abs/2305.10601

It's a simple idea (depth/breadth first search on a tree of chain of thoughts), but sometimes the simple ideas are the best ones, even if they are fairly high up in the stack.

To that end, I started with a basic sklearn dataset and code and asked GTP4 to improve its r2_score.  The starting point was this, base.py:

```
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd

# Fetch the data                                                                                                                 
data = pd.read_pickle("data.pkl")

# Split into features (X) and target (y)                                                                                         
X, y = data.data, data.target

# Split into training and testing sets                                                                                           
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the model                                                                                                          
model = LinearRegression()

# Train the model                                                                                                                
model.fit(X_train, y_train)

# Make predictions                                                                                                               
predictions = model.predict(X_test)

# Compute and display r^2 score                                                                                                  
print('r2_score:', r2_score(y_test, predictions))

```

Not wanting to do a lot of copy/pasting I asked GPT4 to write a program for me to do this 'graph of thoughts' approach.  (I say graph rather than tree, because cycles can exist).

Here's the prompt: 

"Assume a function exists called get_response(prompt) which will make a request to GPT4 and return a response.  You don't have to create that code.  
Now create a python program which when given a base source filename of a Python program that prints out r2_score: <score> will send to getResponse(""Please improve the r2_score metric of the following code. 
Do not provide an explanation, just the code and no additional text.\n\n"+contents of source).  
It will than take the response and remove any boundary characters like "```", save it to a filename which represents its position in the tree (like 'n0.py') and then execute it.  
It will then record the r2_score that is outputed.  It will do this three times, and then pick the run with the best accuracy, which will be the best parent to run from.  
Make this program run recursively and ensure that the filenames of the source tracks the position of the source in the tree.  Output the filenames and r2_score after each run."
    
It created get_best_model.py, which I added a bit of retry logic to and then ran.
    
Here are the results:

| Insight | Initial File | New File | Initial Score | New Score |
|---------|--------------|----------|---------------|-----------|
| Changing the model from LinearRegression to Ridge with alpha=1.0 and adding StandardScaler | base.py | base_n0.py | 0.575 | 0.576 |
| Changing the model from LinearRegression to Ridge with alpha=1.0, adding StandardScaler, and applying PolynomialFeatures with degree=2 | base.py | base_n1.py | 0.575 | 0.647 |
| Changing the model from LinearRegression to Ridge with alpha=10.0, adding StandardScaler, and applying PolynomialFeatures with degree=3 | base.py | base_n2.py | 0.575 | -14.131 |
| Changing the model from Ridge with alpha=1.0 to Lasso with alpha=0.1 | base_n1.py | base_n1_n0.py | 0.647 | 0.482 |
| Changing the model from Ridge with alpha=1.0 to ElasticNet with alpha=0.1 and l1_ratio=0.5 | base_n1.py | base_n1_n1.py | 0.647 | 0.515 |
| Changing the model from Ridge with alpha=1.0 to RidgeCV with automatic alpha selection | base_n1.py | base_n1_n2.py | 0.647 | 0.656 |
| Changing the model from Ridge with alpha=1.0 to RidgeCV with automatic alpha selection and using a pipeline for preprocessing | base_n1.py, base_n1_n2.py | base_n1_n2_n0.py | 0.656 | 0.656 |
| Changing the model from Ridge with alpha=1.0 to RidgeCV with automatic alpha selection, using a pipeline for preprocessing, and increasing the degree of PolynomialFeatures to 3 | base_n1.py, base_n1_n2_n0.py | base_n1_n2_n1.py | 0.656 | -15.415 |
| Changing the degree of PolynomialFeatures from 2 to 3 and using a pipeline for preprocessing | base_n1_n2.py | base_n1_n2_n2.py | 0.656 | -15.415 |
| Changing the model from RidgeCV with automatic alpha selection to LassoCV with automatic alpha selection | base_n1_n2_n0.py | base_n1_n2_n0_n0.py | 0.656 | 0.482 |
| Changing the model from RidgeCV with automatic alpha selection to RandomForestRegressor with 100 estimators | base_n1_n2_n0.py | base_n1_n2_n0_n1.py | 0.656 | 0.799 |
| Changing the model from RidgeCV with automatic alpha selection to GradientBoostingRegressor with n_estimators=200, learning_rate=0.1, and max_depth=2 | base_n1_n2_n0.py | base_n1_n2_n0_n2.py | 0.656 | 0.775 |
| Changing the model from RidgeCV with automatic alpha selection to RandomForestRegressor with GridSearchCV for hyperparameter tuning | base_n1_n2_n0.py | base_n1_n2_n0_n1_n0.py | 0.799 | 0.802 |
| Changing the model from RandomForestRegressor with 100 estimators to GradientBoostingRegressor with n_estimators=300, learning_rate=0.1, and max_depth=3 | base_n1_n2_n0_n1.py | base_n1_n2_n0_n1_n1.py | 0.799 | 0.817


You can find the source for these in the repo. 
    
--
    
There are a lot of optimisations that you can do here, limited only by your imagination (and the 8k/32k context window).  Some ideas are in the paper linked to above, some you'll find on various places where this concept is discussed. 

Among the various obvious ones like dupe checks, pruning, backtracking and monte carlo - the particular optimisation I'm interested in trying next is recording insights gained on each run and refeeding them so that GPT4 can learn from previous changes its made.  

Another idea I'd like to try is appending a set of selected techniques to suggest to GPT4 that it might try.  This last idea might seem a bit like cheating, but it's worth realizing that the library of techniques doesn't need to be in a format exactly like the problem, just enough to hint to GPT4 to try them.  Impediance mismatch is not a problem, so these techniques can be reused for any arbitrary ML problem.

--
    
FAQ
    
1. Wouldn't it be cheaper and easier to just do X?   
    
    Sure, but then why not just make X your baseline.  If automl or optuna is your choice, you can start there.  Or feed them in as a library of selected techniques.
    
    
2. Why did it take so long for GPT4 to try something other than linear models?
    
    I noticed that as well.  Some prompt engineering might help, but I think it's more of an indication as to the limits of GPT4 reasoning.  Better use of the context window and prompt can help here.
    
--    

You might encounter some folks lower down in the stack that will call this '[prompt hacking](https://twitter.com/karpathy/status/1659653943754891279)', but for their benefit I'd like to share this XKCD comic.
   
![image](https://github.com/qrdlgit/graph-of-thoughts/assets/129564070/09b3f3ca-a6c4-4e30-9f8d-e6748aacfe79)


One can even argue there is a formal/propositional logic person even to the right of that.  The point is - everyone has a role to play in the tree of knowledge, why worry about it?