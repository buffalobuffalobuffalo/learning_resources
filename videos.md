# Videos
- Collection of links to lectures, courses and tutorials, in rough chronological order of viewing
## Youtube
- [andrej karpathy - intro to llms](https://www.youtube.com/watch?v=zjkBMFhNj_g)
    - great high level overview of the topic, shareable with others, does not require much prior knowledge
- pydata
    - [vincent warmerdam - Keynote "Natural Intelligence is All You Need"](https://www.youtube.com/watch?v=C9p7suS-NGk)
        - use the hiplot library to visualize parallel coordinates
            - interesting way to easily visualize areas in binary classification feature space dominated by a single class, to develop simple rules as powerful features
    - [Forecasting Customer Lifetime Value (CLTV) for Marketing Campaigns under Uncertainty with PySTAN](https://www.youtube.com/watch?v=hcQST0RnN_o)
        - dataset used
            - https://www.kaggle.com/datasets/baetulo/lifetime-value?select=train.csv
        - website for talk with code
            - https://raphaeltamaki.github.io/raphaeltamaki/posts/Forecasting%20Customer%20Lifetime%20Value%20-%20Why%20Uncertainty%20Matters/
        - uses half cauchy as priors a bunch, worth considering
        - look up pymc's implemention of marketing mixture models 
    - [Okke van der Wal - Personalization at Uber scale via causal-driven machine learning | PDAMS 2023](https://www.youtube.com/watch?v=c_dOpCvkNc0)
        - also discusses CausalML (works at uber)
        - seems like a very strong framework for evaluating strategies for prioritizing contact among set of leads towards the most persuadable
    - [Ana Chaloska - To One-Hot or Not: A guide to feature encoding and when to use what | PDAMS 2023](https://www.youtube.com/watch?v=4Opsiqj6gcY)
        - try taking the log of the target encoding as well
    - [Cikla & Zhutovsky - Transfer Learning in Boosting Models | PyData Amsterdam 2023](https://www.youtube.com/watch?v=lmQw_B-JP9o)
        - lgbm has  `booster_.refit(data, label, decay_rate=0.9, **kwargs)`
            - where the data  and label is the new records
        - you can also add trees at the end of the process by makng another classifier with an init_model
            - lgb.LGBMClassifier().fit(X, y, init_model = original_model)
                - X is the new features
                - y is the new labels
        - possibly a interesting way of quickly updating a model
    - [Van den Bossche - What the PDEP? An overview of some upcoming pandas changes | PyData Amsterdam 2023](https://www.youtube.com/watch?v=z47QwqDUKTo)
        - in pandas 2.0 , there's a `pd.options.mode.copy_on_write = True`
            - this gets rid of the copy warning and means that all modification only apply to current df, which massively speeds up chained operations
            - there's a pyarrow string dtype in 2.1, 
- AI Engineer Summit
    - [Pydantic is all you need: Jason Liu](https://www.youtube.com/watch?v=yj-wSRJwrrc)
        - discussion of a bunch of patterns for trying to introuce broadly understood and testable OOP patterns into the construction of LLM apps. Lots of good examples and motivating ideas, no link to the code and at the end of the talk, the QR code didn't show up in his slide

- tutorial
	- [Polars: The Next Big Python Data Science Library... written in RUST?](https://www.youtube.com/watch?v=VHqn7ufiilE)
                - summary of how to do basic operations in polars
                        - this is a df analysis library written in rust

- [George Hotz | Programming | what is the Q* algorithm? OpenAI Q Star Algorithm | Mistral 7B | PRM800K](https://www.youtube.com/watch?v=2QO3vzwHXhg)
	- recording of 4h live stream of Hotz
		-  watched first hour 2023-11-27
			- going over Q*
			- usiing a fine tuned 7B model called open hermes 2.5 and integrating it with tinygrad

- [5 Awesome Linux Terminal Tools You Must Know](https://www.youtube.com/watch?v=ghWECXWi9kU)
    - btop
        - way better than `top`
    - tldr
        - example use cases for various common tools
    - neofetch
        - nicely configured sys info

- [Talk to Your Documents, Powered by Llama-Index](https://www.youtube.com/watch?v=WL7V9JUy2sE)
    - tutorial for using llamaindex to build a simple talk to your documents RAG

- [3Blue1Brown - But what is a convolution](https://www.youtube.com/watch?v=KuXjwB4LzSA)
- Applied Machine Learning Days 
    - [Developing Llama 2 - Angela Fan](https://www.youtube.com/watch?v=NvTSfdeAbnU)

    - [Building Red Panda - Ce Zhang](https://www.youtube.com/watch?v=zi75gM6ijWw)

- George Hotz
    - [ Mistral mixtral on a tinybox | AMD P2P multi-GPU mixtral-8x7b-32kseqlen](https://www.youtube.com/watch?v=H40QRJFzThQ)
        - talks about the [switch transformers](https://arxiv.org/abs/2101.03961) paper, bc apparently the new mistal MOE model has a similar structure to GPT4

- [Mamba - a replacement for Transformers?](https://www.youtube.com/watch?v=ouF-H35atOY)
    - overview of the recent literature on how this structure may address some of the shortcomings of transformers over long context scenarios.
    - paper recs
        - the annotated s4 by albert gu, karan goel and christopher re
        - mamba: linear-time sequence modeling with selective state spaces by albert gu
    - this approach does incredibly well on induction heads extrapolation (4000x the length of max training examples vs not better than 2x for the other methods tested)
    - also performs incredibly well on language modeling
    - weak on audio waveforms (so good on discrete modalities, bad on continuous)
    - not compared to things llama size

- [Vision Transformer Basics](https://www.youtube.com/watch?v=vsqKGZT8Qn8&t=5s)
    - another video from samuel albanie

- [Self-supervised vision](https://www.youtube.com/watch?v=UaJDdft6BdI)
    - another samuel albanie
    - there's no amount of labelled data that can get us to human level perception via supervised learning, to the point where it won't make dumb mistakes or be sensitive to advesarial examples
    - the motivation is that we should take hints from the development of human perception itself
    - one early approach
        - derive labels from the co-occuring input to other modalities and minimise disageement between class labels predicted by each
        - this starts getting quite close to unsupervised learning
    - for vision
        - pretext task
            - train a computer vision model on a game, where the primary goal is to get it to develop a coherent world model so it can handle generic CV tasks like classification, detection, and segmentation
                - training on image rotations (where the game is to identify how many degrees an image was rotated from 'rightway up') is a very effective method
                - another that works surprisingly well is
                    - image inputs
                    - convnet
                    - run k-means on the features from the convnet, with high k 
                    - use those as labels and use backprop and sgd to train the model
                    - loop over this process a bunch of times
                    - it eventually learns strong features for images without labels 
                - also mentioned
                    - contrastive learning
                        - mess around with an image, flip, convolve blurs, resize, crop, etc
                        - have models evaluate different samples from that original image
                        - train it to identify that they're the same, thus learning deep understandings of the concepts in the image
                    - masked autoencoders
                        - mask 75% of the image
                        - encode the remaining 25%
                        - the decoder has to reconstruct the full image
                        - ends up working pretty well but it's kinda blurry (the issue that often happens with L2 loss)

    - mentions that L2 loss is terrible for multi modal distributions 

    - [The Attention Mechanism in Large Language Models](https://www.youtube.com/watch?v=OxCpWwDCDFQ)
        - first in the series of 3 videos about transformers
        - very basic and high level geometric descriptoin of attention
        - intro to a longer course called llm university
    - [The math behind Attention: Keys, Queries, and Values matrices](https://www.youtube.com/watch?v=UPtG_38Oq8o)