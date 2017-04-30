
#

```
$ brew --version
if not brew exists
	$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
$ brew update && brew upgrade
$ brew install pyenv
$ vi ~/.zshrc
```

```
## pyenv
#
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

```
$ source ~/.zshrc

$ pyenv --version
> pyenv 1.0.7

# python 3.6 = anaconda3-4.2.0 (対応していないライブラリが多いv3)
# python 3.5 = anaconda3-2.5.0 (opencvなど適度に対応してるv3)
# python 2.7 = anaconda2-4.2.0 (ほとんど対応しているが、今から廃れるv2)

$ pyenv install anaconda3-2.5.0
$ pyenv global anaconda3-2.5.0
$ python --version
> Python 3.5.3 :: Anaconda 2.5.0 (x86_64)
```

[TF_PYTHON_URL](https://www.tensorflow.org/install/install_mac#TF_PYTHON_URL)

```
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.1.0-py3-none-any.whl
$ pip install --ignore-installed --upgrade $TF_BINARY_URL
$ python
>> import tensorflow as tf
>> tf.__version__
'1.1.0'
```


```
$ git clone https://github.com/arakawamoriyuki/meetup024.git
$ cd meetup024
$ pip install -r ./requirements.txt
```

```
$ jupyter notebook
```
