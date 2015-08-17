# mingpipe
#### Authors: Nanyun Peng <npeng1@jhu.edu> and Mark Dredze <mdredze@cs.jhu.edu>

mingpipe is a name matcher for Chinese names. It takes two names and predicts whether or not they can refer to the same entity (person, organization or location).

For example:
佛罗伦萨 翡冷翠 true

mingpipe implements several types of name matching algorithms. The best one is based on a trained supervised classifier. The other matchers can be accessed programmatically or via the command line interface.

mingpipe is written in Python and support Python 2.7.x.

mingpipe does not support cross-lingual matching. It requires two names in Chinese characters.

mingpipe ships with a pre-trained model, which corresponds to the best model based on experiments in the mingpipe paper.

The name mingpipe comes from Chinese pronunciation for name(名/ming/) + pipe from the word pipeline.


## How to use mingpipe
There are two ways to use the (classifier) name matcher in mingpipe:

### 1) The Easy Way

mingpipe can be easily integrated into existing code as follows:

> import mingpipe
> 
> mingpipe.score(name1, name2)

`score` returns a tuple of: (boolean, score). The boolean is the prediction, and score is the associated score.

If you have trained a new model, you can also specify a model file and feature file:

> mingpipe.score(s,s, model-file=model-file, feature-file=feature-file)

Every subsequent call to `score` will use this model and feature file. 

To change this, you can reset the name matcher:
> mingpipe.reset_model()

The next time you call reset_model() a new model will be called.

### 2) The Less Easy Way
The above methods wrap the main functionality of mingpipe. You can access this functionality directly. mingpipe implements a NameMatcher (`mingpipe.mingpipe.NameMatcher`). The NameMatcher has multiple prediction methods, and options for training. You can construct a NameMatcher object directly and use it in your application. If you plan on having multiple NameMatchers active at once, this is the approach you should take. The above approach only works with one NameMatcher at a time.

> from mingpipe.mingpipe import NameMatcher
> 
> matcher = NameMatcher(model\_filename=ARG, threshold=0.5, character\_converter=ARG, pinyin\_converter=ARG, pronun\_converter= ARG, feature\_extractor= ARG)

where ARG are suitable values. See mingpipe.mingpipe.py for details. 

## Scoring Names
mingpipe support scoring a file of names using a command line interface.

> python -m mingpipe.test 

Running the above command with the `--help` flag will provide details on the arguments.

Some example arguments:
> \# Levenshtein matcher
> 
> python -m mingpipe.test --data ${DEV} --matcher levenshtein --output-file PREDICTIONS --threshold THRESHOLD --data-type ${TYPE}

> \# Jaro-Winkler matcher
> 
> python -m mingpipe.test --data ${DEV} --matcher jarowinkler --output-file PREDICTIONS --threshold THRESHOLD --data-type ${TYPE}

> \# classifier
> 
> python mingpipe/test --data ${DEV} --matcher classifier --output-file PREDICTIONS --model-file MODEL-FILE --feature-map FEATURE-MAP-FILE --data-type ${TYPE}
 
The PREDICTIONS file will contain predictions for each pair in the DEV file, where each tab separated line contains:
> label  score

## Training Models
Given a set of training examples, mingpipe can train a new model. This is useful if you have a set of training examples that is more reflective of your data than the training data used in the distributed model. 

> python -m mingpipe.train
> 
The data used to train mingpipe is available here:

http://www.cs.jhu.edu/~npeng/data/mingpipe/

Here are some example commands for training mingpipe.

> python -m mingpipe.train  --train-data ${TRAIN} --matcher levenshtein  --data-type ${TYPE}
> python -m mingpipe.train  --train-data ${TRAIN} --matcher jarowinkler --data-type ${TYPE}

These training methods will select an optimal threshold based on these fixed similarity functions.

> python -m mingpipe.train  --train-data ${TRAIN} --dev-data ${TRAIN} --matcher classifier --model-file MODEL-FILE --feature-map FEATURE-MAP-FILE --feature-mode ${MODE-NUM} --align-mode ${ALIGN} --data-type ${TYPE} --tune-parameters true --select-threshold true

This command will train a classifier and save it to the provided path. The last two parameters controls whether you want to tune hyper parameters and select the threshold, or use the defaults.

### feature-mode numbers
The feature-mode are numbers ranging from `[0, 17]`.

Mingpipe always combine the simplified Chinese characters and one pinyin representation to compose features. For `feature-mode < 6`, it combines simplified characters with string-pinyin, for `6 <= feature-mode < 12`, it combines with character-pinyin and for `12 <= feature-mode < 18`, it combines with pronunciation-pinyin. Other value of feature-mode will raise a ValueError.

Then, different value of feature-mode % 6 represents different combination of features, see mingpipe.features.py function extract() for more information.

Our experiments on wiki redirect data shows the feature-mode 11 and 17 performed the best.

## Data Format
The command line methods take data in one of two formats.

### 1) Three column format

The three tab separated columns are:
> name1  name2  true/false

Each line of the data file is a pair of names with the boolean true/false indicating if these names match. This is useful for providing explicit negative examples.

To use this format, pass the flag `--data-type paired`.

### 2) Two column format
The two tab separated columns are:
> name1   name2

This is equivalent to each line containing the label `true`. mingpipe will generate negative examples automatically.

To use this format, pass the flag `--data-type raw`.

## How to cite
If you use mingpipe, please cite it as follows:

Nanyun Peng, Mo Yu, Mark Dredze. An Empirical Study of Chinese Name Matching and Applications. Association for Computational Linguistics (ACL) (short paper), 2015.

> @inproceedings{Peng:2015db,
> 
> 	Author = {Nanyun Peng and Mo Yu and Mark Dredze},
> 
> 	Booktitle = {Association for Computational Linguistics (ACL) (short paper)},
> 
> 	Title = {An Empirical Study of Chinese Name Matching and Applications},
> 
> 	Year = {2015}}

