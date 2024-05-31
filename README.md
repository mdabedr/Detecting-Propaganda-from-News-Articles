
This is a course project for Deep Learning for NLP course (CMPUT-651) from 2019 [ https://lili-mou.github.io/teaching/651_2019/651_2019.html ]
This project attempts to detect propaganda spans from News Articles which is the first Prblem of SemEval 2020 Task-11 [ https://alt.qcri.org/semeval2020/ ] 

Implements a BiLSTM for learning representations and uses a CRF for span generation. Models beat the performance from the original paper by Martino et. al. [ https://aclanthology.org/2020.semeval-1.186/ ]

**Details of the Dataset**
The following table gives the summary stats of the dataset: 
<p align="center">
<img src="https://github.com/mdabedr/Detecting-Propaganda-from-News-Articles/assets/35268893/76000fe5-ee57-4f32-b18f-fa4a689232a6.png" width=50% height=50%>
</p>

**Sentence Based Models**
The first approach is to follow the footsteps of (Martino et al., 2019) and do sentence-level classification. Second approach involves treating entire news articles as data samples for the models. The Results are as follows:

<p align="center">
<img src="https://github.com/mdabedr/Detecting-Propaganda-from-News-Articles/assets/35268893/f9a10849-ad2c-4d9b-b818-971d5f0c3caf.png" width=40% height=40%>
</p>

**Article Based Models**
The motivation behind the article-based approach is to utilize the context information of the articles which is lost in sentence-level tagging.
The Results are as follows:

<p align="center">
<img src="https://github.com/mdabedr/Detecting-Propaganda-from-News-Articles/assets/35268893/f68f2205-bc85-44bc-8e36-128d56104b4a.png" width=40% height=40%>
</p>






