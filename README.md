# Demo

1. Train to predict 5


2. Calculate softmax


3. Calculate loss:


![Alt text](loss.png?raw=true "Title")

4. Since y_i is 0 for all i!=5, and y_5 is 1, the loss becomes:
** -log(y_prediction_5)


5. Calculate gradients by backpropagation (chain rule)


6. Update parameters based on optimizer (learning rate, learning rate decay etc.) 


