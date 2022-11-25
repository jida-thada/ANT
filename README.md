# Knowledge Amalgamation for Multi-Label Classification (KA-MLC)
This is the implementation repository of <i>Adaptive kNowledge Transfer (ANT)</i>, the solution for KA-MLC.

## File listing

+ __main.py__ : Code for training ANT
+ __model.py__ : Supporting models
+ __utils.py__ : Supporting utility functions
+ __requirements.txt__ : Library requirements

## Instructions 

Prepared folders:

+ __data__ : directory to place training data
+ __teachers__ : directory to place teacher models
+ __output__ : directory for training logs and outputs 

The datasets and the teachers we use in our paper are available here: 
https://drive.google.com/drive/folders/1QlFDmR4D5fA6MQMnkqy1-fG-6leQI-Od?usp=sharing


Run script as:

    python main.py -t_numlabel 4 4 -t_labels 1 2 3 4 3 4 5 6 -stu_labels 1 2 3 4 5 6 \
    -t_path './teachers/t0.sav' './teachers/t1.sav' -dataname 'sample_data'   

<b>Parameters:</b>

+ __Required:__
  + __-t_path__ : path of each teacher model, e.g., './t0.sav' './t1.sav'
  + __-t_numlabel__ : #labels corresponding to each teacher in t_path, e.g., 4 4'
  + __-t_labels__ : concatenated specialized labels of each teacher corresponding to t_path, e.g., t0_label: 1 2 3 4 and t1_label: 3 4 5 6, then t_labels: 1 2 3 4 3 4 5 6
  + __-stu_labels__ : student labels, e.g., 1 2 3 4 5 6
  + __-dataname__ : unlabelled data for training the student

+ __Hyperparameters:__
  + __-lr__ : learning rate, default 0.001
  + __-ep__ : epochs, default 500
  + __-bs__ : batch size, default 8
  + __-layer__ : #layers, default 1
  + __-hidden__ : #hidden size, default 32

  
