## CLARA-PnP environment experiment
The project page is here [Project](https://clararobot.github.io/)

![example](picknplace.jpg)

The Demo code can be runed in
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeongeun980906/CLARA-PNP-DEMO/blob/master/DEMO_colab.ipynb)

### 0. Run Vision (Optional)
```
python collect_vision_only.py
```
### 1. RUN LLM
```
python test.py --method 2 --lm chat
```

### 2. Val on Language Domain
```
python val_language.py --method 2 --lm chat
```
Please edit success annotation for ood task!
### 3. Run CLIPORT
```
python val_policy.py --method 2 --lm chat
```
Please edit poicy success annotation for ood task!
### 4. Question Generation
```
python test_inter.py --method 2 --lm chat --first
```

### 5. Label
On json file
"answer" :["your answer here"]

### 6. Regeneration
```
python test_inter.py --method 2 --lm chat --last
```


