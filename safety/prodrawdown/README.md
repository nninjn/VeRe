Different from previous work, we slightly modified the definition of drawdown to more appropriately evaluate the degree to which the original performance of the model is affected after repair.
Evaluating drawdown only within the state space of the property to be repaired is one-sided. For example, an overfitting repair may repair the model to satisfy the property specification throughout the
entire state space. In this case, the original drawdown is 0, which is biased. Therefore, we select 3 properties (including the property to be repaired) for each model, and sample 5K non-violating instances
from the state space of each properties. Specifically, we select the property (2, 7, 8) for 𝑁1,9 , (2, 3, 8) for 𝑁2,9 and (2, 3, 7) for the others.
Finally, we generate a drawdown set of size 15,000 for each model. 
