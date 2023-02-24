# Final Project Readme

## Environment
```
pip install torch-rechub

```

## Download
```
sh download.sh
```

## (Reproduce) Inference

### Seen Course
```
cd seen_course
sh test.sh ../input/hahow/test_seen.csv /path/to/submission.csv
```

### Unseen Course
```
cd unseen_course
sh test.sh ../input/hahow/test_unseen.csv /path/to/submission.csv
```

### Seen Topic
```
cd seen_topic
sh test.sh ../input/hahow/test_seen_group.csv /path/to/submission.csv
```


### Unseen Topic
```
cd unseen_topic
sh test.sh ../input/hahow/test_unseen_group.csv /path/to/submission.csv
```

## Training

```
sh train.sh
```