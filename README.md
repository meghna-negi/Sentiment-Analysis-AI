# Sentiment-Analysis-AI
This project creates a simple web application where an input comment is categorized into positive, negative and neutral.
The comments were scrapped from the trailer of Jurassic World Dominion (released in 2022) to train the Neural Network.

**Data Pre-processing steps for Neural Network Model**

1. Emoji removal

2. Tokenization

3. PoS Tagging

4. Lematization

**Technology Stack for Web Development**

1. **Flask** framework for backend logic

2. **HTML** and **CSS** for frontend development

**Web application layout**

The homepage of this sentiment analysis web application has text box to enter a comment that you want to analyze the sentiment of.
The Analyze Sentiment button directs the request to call the predict function and returns the sentiment of the comment.
The homepage looks something like this:

![sentimentAnalysisHomePage](https://github.com/user-attachments/assets/3b4a2540-0f3a-48c2-8f1c-d07be9b880e1)

The webpage wih sentiment contains the comment to be analyzed and the sentiment of the comment predicted by the Neural Network model.
The example on how the sentiment of a given input is displayed on the webpage is shown below:

![sentimentAnalysisNegativeComment](https://github.com/user-attachments/assets/a40d4491-71b3-41be-8971-998d8ba4ad7f)

