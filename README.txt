
Project "A lightweight method for facial expression recognition based on improved MobileNetV3"

data set
FERPlus, download in "https://www.worldlink.com.cn/osdir/ferplus.html"
RAF-DB,  download in "http://www.whdeng.cn/RAF/model1.html"
After downloading, put the data set in the corresponding path.

models
This folder is used to store pre-trained weights.

new_newexp
This folder is used to store the trained models and logs.

src, including
     image_utils.py, used for data enhancement;
     Model_V3.py   , the main body of the network;
     train.py      , training;
     test          , validation.

Pay attention to the path in the code.

