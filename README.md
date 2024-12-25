This project contains the following files

Models.py: The ML models to detect language- including AdaBoost and Decision Tree.

best.model: The best trained model created to detect langauge as either English or Dutch - best.model is trained from the Stumped Decision Tree.

features.txt: The features used to train the best model created(best.model) the features are common distinct words between both languages English & Dutch.

myexamples.txt: Sentences models were trained on. 

examples.dat: Sentences models were evaluated/tested on. 

93.75 % accuracy in detecting language as English or Dutch(tested on 416 sentences). 

Acc: 0.9375 (390/416)
Score  is adjusted to accuracy range (min:0.5, max:0.9) 
STDOUT not reported due to length.
===========Log===========
Testing with python3.11 /autotester/submission/lab3.py predict /autotester/source/testing/data/test.dat /autotester/submission/features.txt /autotester/submission/best.model

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Read Writeup.pdf for more information :) 
