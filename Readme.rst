------------
Text-As-Data Workshop
------------

**Uni Exeter Econ, June 22, 2022**

This Github page contains resources for use in a text-as-data workshop in economics at the University of Exeter on June 22.

For this workshop I'll be working through some tools in Python 3, specifically Python 3.9. If you haven't used Python before, I like using the PyCharm IDE (integrated development environment). The community version is freely available, and has lots of nice easy-to-use features. It's available here: https://www.jetbrains.com/pycharm/download/

This workshop is aimed at people who aren't familiar with text-as-data tools yet, and who ideally have a bit of experience with Python, but even beginners to Python will be able to follow along. The goal is to introduce some of the most common Natural Language Processing tools and give a few examples that can be a jumping off point for applications in your own work.

**Some additional instructions I should have added for a first-time install**
When you open PyCharm for the first time, you may need to install Python. If you click "New Project" from the home page, it will ask you what interpreter you want to use. The simplest option is probably to select "Previously Configured Interpreter" then choose "System Interpreter" and then select "Python 3.9" from the drop-down menu. You may need to install Python and give authorization for it to download/install. The other way to do this is to select "New environment using Virtualenv" which will create an isolated Python environment where you can install these packages without messing up any other python packages you might have installed (e.g. if you've installed another package elsewhere that requires a specific version of numpy, that might stop working if you install the default). 

Once you've got Python installed and ready to go, there are a few packages we'll go through that you need to install. There are a few basic general-use Python libraries you'll need to install:

  #. pandas
  #. numpy
  #. matplotlib
  #. statsmodels

Then some more natural language processing or machine learning specific packages:

  #. sklearn
  #. nltk
  #. vaderSentiment
  #. gensim

If you're using PyCharm installing these is easy to do. In the bottom-right corner of the screen it will say something like "Python 3.9" to tell you which Python interpreter PyCharm is using. If you click on this, then on "Interpreter Settings" in the menu that pops up, it will list the packages installed. At the bottom of this list there will be a "+" symbol that allows you to install new packages. If you click the "+" button, it will bring up a list of available packages, and you can search by name for the packages listed above, and it will automatically install for you.
