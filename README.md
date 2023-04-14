# CSC413-RNN

## Introduction
This repository contains the implementation of a RNN model with attention mechanism for sentiment analysis on Yelp reviews (text classogocaton). The task being solved is to classify the reviews which are inputs, into positive (4-5 stars) or negative (1-2 stars). The input is a sequence of words represented as word embeddings, and the output is a probability distribution over the categories. In the model, we used an LSTM to learn sequential dependencies in the input text with a self-attention mechanism to weigh the importance of each word in the input sequence. The output is generated by passing the final hidden state of the LSTM through two fully connected layers with ReLU activation. Then, a final softmax layer is used to obtain the category probabilities.

## Model Figure
![image](https://user-images.githubusercontent.com/77242297/230280568-371ffc05-9019-4147-b126-791cbe1b22ef.png)

Preprocessing:
1. Data Augmentation (See data transformation)
2. Tokenizing the text and padding the sequences to ensure that they are of the same length
3. Update vocabulary
4. Convert input tokens into embedding vectors.

RNN:
1. A Self-Attention layer that computes attention weights and modifies the input embeddings accordingly. (https://atcold.github.io/pytorch-Deep-Learning/en/week12/12-3/)
2. An LSTM layer to process the input sequence and capture temporal dependencies. It addresses gradient explosion or vanishing issue.
3. Fully Connected layers to produce the final output.

Output:
Use threshold to the determine whether the review is positive or negative.

## Model Parameter
* Embedding dimension: The dimensionality of the input embeddings.
* Hidden dimension: The dimensionality of the hidden states in the LSTM layer.
* Output dimension: The dimensionality of the output.
* Number of attention heads: The number of attention heads used in the multi-head attention mechanism.
* Learning rate: The step size of the parameter updates during the optimization process.
* Batch size: The number of training samples used in a single update step of the model's parameters during training.
* Number of epochs: An epoch represents a complete iteration over the entire training dataset during which the model updates its parameters.


## Model Examples
Here is a preview of the test set result. The test accuracy of out best model is 85.1%.
![image](https://user-images.githubusercontent.com/77242297/231547960-b6bf2dfd-3f3b-4b10-93f0-e52784358549.png)

### Successful exmaple:
Input: 'I really enjoyed this place. I wish I could give it 4.5 stars, but couldn\'t give it 5. Had steamed vegetable dumplings, which were very good. For my entree, it was the pad kra pao. I ordered it "Thai Hot", but I don\'t think they really believed me. Good thing they brought extra hot pepper flakes. I hate to leave a Thai restaurant when my mouth isn\'t on fire. The dish was good, but just a bit dry. I really did enjoy it, but it couldn\'t earn top marks.'

Prediction: Positive (1)
True Label: Positive (1)

### Unsccessful exmaple:
Input: "Great to have a new spot in Temple Terrace. Food is good. Service is good. However, I'm not sure if they are trying to be a night club or a restaurant. The Music is so dang loud it's hard to think let alone carry on a conversation w/ folks I'm hanging out with. Turn down the music & you will get more business!"

Prediction: Positive (1)
True Label: Negative (0)

## Data Source
The souce of our data set comes from a Yelp dataset(https://www.yelp.com/dataset). 
The complete dataset contains over 6 million reviews, along with other business and user information.

## Data Summary
Here are some summary statistics
1.	Average length of review:  550 (include any character)
2.	Average rating : 3.8 
3.	Number of reviews positive reviews (4-5 stars): 139314
4.	Number of reviews negative reviews (1-2 stars): 38038
5.	Total data : 200000


## Data Transformation
We read the original JSON file was read line by line, loads each line as a JSON object, extracts the text and star rating from the object. We saved the output into a text file in the format of "text"\t rating\n where text is the cleaned review text and rating is the star rating (1-5). We collected 200,000 reviews in total.

To make the model more robust and capable of handling a variety of inputs, we used data augmentation techniques on a randomly selected 40% of the training set. The first technique was synonym replacement, which involves replacing a word with its synonym to increase the diversity of the vocabulary in the dataset. The second technique was random word swap, which swaps the positions of two words to change the word order and syntax of the sentences. The third technique was word insertion, which involves inserting a word into the review text to add more context and complexity to the sentences. The final technique was word deletion, which deletes two words from the review text to simplify the sentences. These techniques were chosen to help the model generalize better and get a better understanding of text grammar and syntax.

## Data Split
The dataset was split into train (60%), validation (20%), and test sets (20%) using sklearn's train_test_split function. We used this split because we wanted a large dataset for training to identify details and patterns. The 20% validation is sufficient to fine tune and the 20% testing is sufficient to evaluate performance.

## Training Curve
![image](https://user-images.githubusercontent.com/71288561/231910554-5249c93d-2fa6-432e-9b4b-a979713fb796.png)


## Hyperparameter Tuning
Epoch = 10, Learning Rate = 0.01
| Batch Size  | Train Accuracy | Valid Accuracy |
| ----------- | -------------- | -------------- |
| 4           |     89.8%      |     83.3%      |
| 8           |     91.4%      |     83.5%      |
| 16          |     92.3%      |     85.1%      |
| 32          |     89.2%      |     85.0%      |
| 64          |     86.2%      |     83.9%      |

First, we fixed epoch and learning rate and varied batch sizes.
The training accuracy initially increases from batch size 4 to 16, and then decreases with further increase in batch size to 64. The initial increase may be due to the smoother gradient updates and more stable training with larger batch sizes, while the decrease at larger batch sizes could be attributed to slower convergence and less effective exploration of the solution space.

The validation accuracy initially increases from batch size 4 to 16, and then decreases from 16 to 64. This behavior indicates that there is a trade-off between the batch size and the model's ability to generalize to unseen data.

Epoch = 10, Batch Size = 16
|Learning Rate| Train Accuracy | Valid Accuracy |
| ----------- | -------------- | -------------- |
| 0.05        |     89.5%      |     83.6%      |
| 0.01        |     92.3%      |     85.1%      |
| 0.005       |     88.3%      |     83.7%      |
| 0.001       |     77.0%      |     76.0%      |
| 0.0005      |     70.4%      |     69.4%      |

Then, we fixed epoch and batchsize and varied learning rate. The training accuracy decreases with decreasing learning rates. This is because a smaller learning rate causes the model to learn more slowly, making fewer adjustments to its parameters during each update. With a fixed number of epochs (10), smaller learning rates may not provide enough updates for the model to effectively learn the underlying patterns in the data, resulting in lower training accuracy.

The validation accuracy follows a similar pattern. It initially increases from a learning rate of 0.05 to 0.01 and then decreases as the learning rate continues to decrease. The suggested optimal learning rate is 0.01 that allows the model to learn effectively from the training data and generalize well to the validation dataset.

Batch Size = 16, Learning Rate = 0.01
|    Epoch    | Train Accuracy | Valid Accuracy |
| ----------- | -------------- | -------------- |
| 5           |     89.1%      |     84.2%      |
| 10          |     92.3%      |     85.1%      |
| 20          |     94.5%      |     83.9%      |
| 30          |     97.2%      |     83.8%      |
| 40          |     97.8%      |     83.6%      |

Finally, we fixed batch size and learning rate and varied the number of epochs. The training accuracy consistently increases with more epochs. This is expected, as the model has more iterations over the entire dataset, allowing it to learn the underlying patterns more effectively and fine-tune its parameters.

The validation accuracy initially increases from 5 to 10 epochs and then starts to decrease as the number of epochs continues to increase. This suggests that beyond 10 epochs, the model begins to overfit the training data, learning the noise rather than the underlying patterns. As a result, the model's ability to generalize to unseen data (validation dataset) deteriorates, leading to a decrease in validation accuracy.

In conclusion, to yield the highest validation accuracy while minimizing overfitting, we select 10 epochs, a batch size of 16, and a learning rate of 0.01 as the final parameters to provide the best performance on the test set.


## Quantitative Measure
The model performance was evaluated by simply comparing the predicted class with the ground truth.

## Quantitative and Qualitative Results
The model achieved an accuracy of 85.1% on the test set.

## Justification of Results
Based on the result, our model has achieved a very high accuracy on the test set (85.1%), meaning it is able to well find the patterns in the training set and well generalize on the test set. This is an expected result from both model structure and hyperparameter tuning.

For model structure, LSTM is well-suited for processing sequential data, such as text. It is able to capture long-term dependencies in the input data, which is important for understanding the context of a review and how the sentiment of one sentence may be influenced by the sentiment of previous sentences. The LSTM model can also handle variable-length inputs and outputs, which is useful for processing text data that can have varying lengths.

Additionally ,we used the attention mechnism to allow the model to selectively focus on specific parts of the input data, giving more weight to the parts that are most relevant to the prediction. This is particularly important for text data, where different words and phrases can have different levels of importance in determining the sentiment of a review. The attention mechanism can help the model to identify and weigh the most important features of the input data, leading to more accurate predictions.

Combining LSTM and Attention can be especially effective for sentiment analysis tasks, as the LSTM can capture long-term dependencies in the input data, while the attention mechanism can help the model to focus on the most relevant parts of the input. By using both techniques together, the model can better understand the context of a review and make more accurate predictions about its sentiment.

For hyperparameter tuning. we have discussed above we chose the parameters that performed best on the valid set to optimize the bias-variance tradeoff.

We also produces a confusion matrix to further interpret the result from a business perspective. 
|          | Positive | Negative |
| -------- | -------- | -------- |
| Positive |    419   |    174   |
| Negative |    124   |   1283   |

We calculated the False Positive Rate, False Negatice Rate, True Positive Rate and True Negative Rate from the confusion matrix.
False Positive Rate: 11.9%
False Negative Rate: 22.8%
True Positive Rate: 77.2%
True Negative Rate: 88.1%

The fact that false positive rate is only half of the false negative rate suggest that our model is risk-efficient from a business perspective, in the following ways:
1. Accurate information for users: Our platform's main goal is to provide accurate information to users about businesses. If the model predicts false positive ratings, it could mislead users into visiting a business that doesn't meet their expectations, leading to negative experiences and potentially harming our platform's reputation. 
2. Trust with businesses: False positive ratings can also harm businesses. If a business receives an unwarranted positive rating, it could attract customers who expect a different experience, which could result in negative reviews and harm the business's reputation. This could also lead to mistrust between our platform and the businesses we serve. 
3. Avoid legal issues: Sometimes, businesses may take legal action against our platform if they feel they have been unfairly represented. By ensuring that reviews accurately reflect the quality of the business, a low FPR can help us avoid such legal issues. 
4. Positive user engagement: When our users trust the accuracy of the reviews, they are more likely to engage with our platform and contribute their own reviews. A low FPR can help build trust with users and encourage them to use our platform more frequently.


## Ethical Consideration
Our model is to provide restaurants with an automated system to analyze customer reviews to help them improve restaurant service and dining quality. However, there are potential ethical concerns regarding the misuse of our model. For example,if the training data is biased towards certain demographics or experiences, the model may not represent the experiences of all customers, leading to discriminatory or unfair outcomes when used to make decisions about restaurants. Moreover, there is a risk that the model may be misused by restaurant owners or others to manipulate ratings of their own businesses or competitors. This could harm consumers and erode trust in the Yelp platform.

## Authors
Aoqi long: LSTM Model + Attention Mechanism Implementation

Zechuan Liu: Data Summary + Data Augmentation + Training Implementation

Hanxin Yuan: Hyperparameter Tuning + Result Interpretation + Readme Write-up

Yi Fei Pang: 
Debugging + Result Interpretation + ReadMe Write-up
