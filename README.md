The project contains the following files

Models.py: the ML models to detect language- including AdaBoost and Decision Tree

best.model: the best trained model created to detect langauge as either English or Dutch - best.model is trained from the Decision Tree,

features.txt: the features used to train the best model created(best.model) the features are common distinct words between both languages English & Dutch.

myexamples.txt: sentences models were trained on. 

examples.dat: sentences models were evaluated/tested on. 

93.75 % accuracy in detecing languages

Acc: 0.9375 (390/416)
Score  is adjusted to accuracy range (min:0.5, max:0.9) 
STDOUT not reported due to length.
===========Log===========
Testing with python3.11 /autograder/submission/lab3.py predict /autograder/source/testing/data/test.dat /autograder/submission/features.txt /autograder/submission/best.model

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Read Writeup.pdf for more information :) 
